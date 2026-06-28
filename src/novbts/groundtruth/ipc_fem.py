#!/usr/bin/env python3
"""
IPC-based FEM ground truth for gel-indentation-shear (PARALLEL to the PhysX
extractor isaac_extract_shear.py -- PhysX path is left untouched).

Motivation: the PhysX deformable solver uses penalty/regularized friction whose
effective behaviour is mesh-coupled, so its TANGENTIAL field does not converge
under mesh refinement (res-24/32/40 tangential rel-L2 ~0.7-0.9, non-monotonic,
and PhysX blows up past ~res-48).  Incremental Potential Contact (IPC,
Li et al. 2020) provides a *convergent* friction model (a smoothed Coulomb law
with a controllable, mesh-independent velocity threshold eps_v -> 0) with a
guaranteed intersection-free barrier.  We assemble a standard nonlinear FEM
(Neo-Hookean elasticity, large deformation) around the `ipctk` (IPC Toolkit)
contact/friction potentials and a projected-Newton solver with CCD-filtered
line search -- i.e. what IPC labs do.

This file builds up in stages; each stage is validated by a self-test in
__main__ before the next is added:
  Stage 1  box tet mesh                       (positive volumes)
  Stage 2  Neo-Hookean elasticity (torch AD)  (FD grad check + uniaxial energy)
  Stage 3  IPC contact vs rigid sphere        (Hertz contact radius)   [next]
  Stage 4  friction + tangential load steps                            [next]
  Stage 5  mesh-convergence harness vs PhysX                           [next]

Run self-tests:  .venv-gate2/bin/python -m novbts.groundtruth.ipc_fem --selftest
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

torch.set_default_dtype(torch.float64)   # FEM needs double precision
DEV = "cpu"                              # elasticity assembly on CPU (sparse); small meshes


# --------------------------------------------------------------------------- #
# Stage 1 -- box tetrahedral mesh
# --------------------------------------------------------------------------- #
# 6-tet decomposition of a unit cube (node-local indices 0..7), consistent
# orientation so every tet has positive volume for a right-handed grid.
_CUBE_TETS = np.array([
    [0, 5, 1, 7], [0, 1, 3, 7], [0, 3, 2, 7],
    [0, 2, 6, 7], [0, 6, 4, 7], [0, 4, 5, 7],
], dtype=np.int64)


def build_box_mesh(Lx=0.05, Ly=0.05, Lz=0.02, res=8):
    """Structured tet mesh of the gel block occupying
    x,y in [-L/2, L/2], z in [0, Lz].  `res` = #cells along the longest XY edge;
    z-divisions scaled to keep elements roughly cubic.

    Returns dict with X (N,3) rest nodes, tets (T,4), and boolean node masks
    bottom (z==0, clamped) and top (z==Lz, the sensor/marker surface).
    """
    nx = max(2, int(round(res)))
    ny = max(2, int(round(res * Ly / Lx)))
    nz = max(2, int(round(res * Lz / Lx)))
    xs = np.linspace(-Lx / 2, Lx / 2, nx + 1)
    ys = np.linspace(-Ly / 2, Ly / 2, ny + 1)
    zs = np.linspace(0.0, Lz, nz + 1)
    # node grid, index = ((iz)*(ny+1) + iy)*(nx+1) + ix
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")  # (nx+1,ny+1,nz+1)
    X = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    # X was flattened from meshgrid(indexing="ij").reshape(-1,3): C-order over
    # (nx+1, ny+1, nz+1) -> iz is the fastest-varying index.
    def nid(ix, iy, iz):
        return (ix * (ny + 1) + iy) * (nz + 1) + iz

    tets = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # 8 corners of the cell in local order 0..7 (x fastest, then y, then z)
                c = [nid(ix + (k & 1), iy + ((k >> 1) & 1), iz + ((k >> 2) & 1)) for k in range(8)]
                for t in _CUBE_TETS:
                    tets.append([c[t[0]], c[t[1]], c[t[2]], c[t[3]]])
    tets = np.array(tets, dtype=np.int64)

    bottom = np.isclose(X[:, 2], 0.0)
    top = np.isclose(X[:, 2], Lz)
    return dict(X=X, tets=tets, bottom=bottom, top=top, dims=(Lx, Ly, Lz),
                grid=(nx, ny, nz))


def _orient_tets(X, tets):
    """Flip any inverted tet so all rest volumes are positive."""
    p = X[tets]
    d1 = p[:, 1] - p[:, 0]; d2 = p[:, 2] - p[:, 0]; d3 = p[:, 3] - p[:, 0]
    vol6 = np.einsum("ij,ij->i", np.cross(d1, d2), d3)
    flip = vol6 < 0
    tets = tets.copy()
    tets[flip] = tets[flip][:, [0, 1, 3, 2]]
    return tets


# --------------------------------------------------------------------------- #
# Stage 2 -- Neo-Hookean elasticity (compressible, log-J form)
# --------------------------------------------------------------------------- #
def lame_from_E_nu(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def precompute_rest(X, tets):
    """Rest-shape inverse Dm^{-1} (T,3,3) and rest volume W (T,)."""
    P = X[tets]                                   # (T,4,3)
    Dm = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], axis=-1)  # (T,3,3) cols
    W = np.abs(np.linalg.det(Dm)) / 6.0
    Dm_inv = np.linalg.inv(Dm)
    return torch.as_tensor(Dm_inv), torch.as_tensor(W)


def neohookean_energy(x, tets, Dm_inv, W, mu, lam):
    """Total compressible Neo-Hookean energy.
    Psi = mu/2 (I_C - 3) - mu ln J + lam/2 (ln J)^2,  per unit rest volume.
    x: (N,3) tensor (current positions).  Returns scalar tensor.
    """
    p = x[tets]                                   # (T,4,3)
    Ds = torch.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], dim=-1)  # (T,3,3)
    F = Ds @ Dm_inv
    J = torch.linalg.det(F)
    IC = (F * F).sum(dim=(-1, -2))
    logJ = torch.log(J.clamp_min(1e-9))
    Psi = 0.5 * mu * (IC - 3.0) - mu * logJ + 0.5 * lam * logJ * logJ
    return (Psi * W).sum()


# --------------------------------------------------------------------------- #
# Stage 3 -- IPC contact vs a rigid ANALYTIC sphere (SDF) + projected Newton
# --------------------------------------------------------------------------- #
# Sphere is rigid with prescribed centre c(t); for a gel node p the unsigned
# distance to the sphere surface is d = ||p - c|| - R.  We use the IPC log
# barrier b(d) on d<dhat (convergent, intersection-free) -- exact for a sphere,
# so no mesh/CCD is needed.  Friction (smoothed Coulomb) is added in Stage 4.

def ipc_barrier_vals(d, dhat):
    """IPC barrier b(d) = -(d-dhat)^2 ln(d/dhat) and its 1st/2nd derivatives,
    for 0<d<dhat (else 0).  d: (K,) tensor."""
    r = d / dhat
    logr = torch.log(r)
    dm = d - dhat
    b = -(dm * dm) * logr
    bp = -2.0 * dm * logr - (dm * dm) / d
    bpp = -2.0 * logr - 4.0 * dm / d + (dm * dm) / (d * d)
    active = (d < dhat) & (d > 0)
    z = torch.zeros_like(d)
    return torch.where(active, b, z), torch.where(active, bp, z), torch.where(active, bpp, z)


def _det3(F):
    """Explicit 3x3 determinant -- clean higher-order derivatives under torch.func
    (torch.linalg.det's 2nd derivative yields NaN through functorch)."""
    return (F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1])
            - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
            + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]))


def _elastic_local(x, tets, Dm_inv, W, mu, lam):
    """Per-tet gradient (T,12) and PSD-projected hessian (T,12,12) via torch.func."""
    from torch.func import vmap, hessian, grad as fgrad

    def te(p12, Dmi, w):
        p = p12.reshape(4, 3)
        Ds = torch.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], dim=-1)
        F = Ds @ Dmi
        J = _det3(F)
        IC = (F * F).sum()
        logJ = torch.log(J.clamp_min(1e-9))
        return (0.5 * mu * (IC - 3.0) - mu * logJ + 0.5 * lam * logJ * logJ) * w

    p12 = x[tets].reshape(-1, 12)
    g = vmap(fgrad(te), in_dims=(0, 0, 0))(p12, Dm_inv, W)        # (T,12)
    H = vmap(hessian(te), in_dims=(0, 0, 0))(p12, Dm_inv, W)      # (T,12,12)
    return g, _project_psd(H)


def _project_psd(H):
    """Clamp eigenvalues to >=0 (per matrix in a batch). Uses numpy's LAPACK
    eigh (robust to the 6 repeated zero rigid modes of rest-state tets, which
    make torch's batched eigh fail with error code 11)."""
    Hs = 0.5 * (H + H.transpose(-1, -2)).numpy()
    w, V = np.linalg.eigh(Hs)                      # batched, robust
    w = np.clip(w, 1e-12, None)
    Hp = (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)
    return torch.as_tensor(Hp)


def _scatter_grad(gloc, tets, N):
    g = torch.zeros(N * 3, dtype=gloc.dtype)
    idx = (tets[:, :, None] * 3 + torch.arange(3)).reshape(tets.shape[0], 12)  # (T,12)
    g.index_add_(0, idx.reshape(-1), gloc.reshape(-1))
    return g


def _scatter_hess(Hloc, tets, N):
    import scipy.sparse as sp
    T = tets.shape[0]
    idx = (tets[:, :, None] * 3 + torch.arange(3)).reshape(T, 12)  # (T,12)
    rows = idx[:, :, None].expand(T, 12, 12).reshape(-1).numpy()
    cols = idx[:, None, :].expand(T, 12, 12).reshape(-1).numpy()
    vals = Hloc.reshape(-1).numpy()
    return sp.coo_matrix((vals, (rows, cols)), shape=(N * 3, N * 3)).tocsr()


def _contact_terms(x, center, R, dhat, kappa, N):
    """Contact gradient (3N,) and sparse PSD hessian from analytic sphere barrier."""
    import scipy.sparse as sp
    rel = x - center                                   # (N,3)
    dist = torch.linalg.norm(rel, dim=1)               # (N,)
    d = dist - R
    n = rel / dist.unsqueeze(1).clamp_min(1e-12)       # outward normals (N,3)
    b, bp, bpp = ipc_barrier_vals(d, dhat)
    act = (d < dhat) & (d > 0)
    g = torch.zeros(N * 3, dtype=x.dtype)
    rows, cols, vals = [], [], []
    ai = torch.nonzero(act).flatten()
    for i in ai.tolist():
        ni = n[i]
        gi = kappa * bp[i] * ni                        # dE/dp_i
        g[3 * i:3 * i + 3] = gi
        # hessian: kappa*(b'' n n^T + b' (I - n n^T)/dist)
        Hi = kappa * (bpp[i] * torch.outer(ni, ni)
                      + bp[i] * (torch.eye(3, dtype=x.dtype) - torch.outer(ni, ni)) / dist[i])
        w, V = torch.linalg.eigh(0.5 * (Hi + Hi.T)); w = w.clamp_min(0.0)
        Hi = (V * w) @ V.T
        for a in range(3):
            for bb in range(3):
                rows.append(3 * i + a); cols.append(3 * i + bb); vals.append(float(Hi[a, bb]))
    Hc = sp.coo_matrix((vals, (rows, cols)), shape=(N * 3, N * 3)).tocsr() if vals \
        else sp.csr_matrix((N * 3, N * 3))
    return g, Hc, (float(d[act].min()) if act.any() else float("inf")), int(act.sum())


def solve_static(X, tets, Dm_inv, W, mu, lam, center, R, dhat, kappa,
                 fixed_mask, x_init=None, max_iter=60, tol=1e-9, verbose=False):
    """Projected-Newton minimise elastic + contact energy. Bottom nodes (fixed_mask)
    are Dirichlet-clamped at rest. Returns equilibrium positions x (N,3)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    N = X.shape[0]
    tets = torch.as_tensor(np.asarray(tets)).long()     # ensure torch index tensor
    x = (torch.as_tensor(X.copy()) if x_init is None else x_init.clone())
    center = torch.as_tensor(np.asarray(center, dtype=float))
    free = ~fixed_mask
    fdof = np.repeat(free, 3)                            # (3N,) bool
    fdof_idx = np.nonzero(fdof)[0]

    def total_energy(xq):
        e = neohookean_energy(xq, tets, Dm_inv, W, mu, lam)
        rel = xq - center; dist = torch.linalg.norm(rel, dim=1); dd = dist - R
        bvals, _, _ = ipc_barrier_vals(dd, dhat)
        return float(e + kappa * bvals.sum())

    for it in range(max_iter):
        gl, Hl = _elastic_local(x, tets, Dm_inv, W, mu, lam)
        g = _scatter_grad(gl, tets, N)
        H = _scatter_hess(Hl, tets, N)
        gc, Hc, dmin, nc = _contact_terms(x, center, R, dhat, kappa, N)
        g = g + gc; H = H + Hc
        gfree = g.numpy()[fdof_idx]
        gnorm = np.linalg.norm(gfree)
        if verbose:
            print(f"   it{it:2d} |g|={gnorm:.3e} contacts={nc} dmin={dmin:.3e}")
        if gnorm < tol:
            break
        Hff = H[fdof_idx][:, fdof_idx] + sp.eye(len(fdof_idx)) * 1e-8
        dx = spsolve(Hff.tocsc(), -gfree)
        step = torch.zeros(N * 3, dtype=x.dtype)
        step[fdof_idx] = torch.as_tensor(dx)
        step = step.reshape(N, 3)
        # line search: keep all distances > 0 (no penetration) + energy decrease
        e0 = total_energy(x); alpha = 1.0
        for _ in range(40):
            xn = x + alpha * step
            dnew = torch.linalg.norm(xn - center, dim=1) - R
            if (dnew > 1e-9).all() and total_energy(xn) <= e0 + 1e-12:
                break
            alpha *= 0.5
        x = x + alpha * step
    return x, gnorm, dmin, nc


def indent_sphere(X, tets, Dm_inv, W, mu, lam, R, depth, dhat, kappa, fixed_mask,
                  Lz, x_init=None, verbose=True):
    """Quasi-static indentation by an analytic sphere, descending from JUST-TOUCHING
    in increments capped by the current gap so a node is never placed inside the
    sphere (the IPC barrier is only valid for d>0)."""
    x = torch.as_tensor(X.copy()) if x_init is None else x_init.clone()
    cz_start = Lz + R + dhat                 # sphere bottom dhat above the flat surface
    cz_final = Lz + R - depth
    cz = cz_start
    step_cap = 0.3 * dhat
    it = 0
    while cz > cz_final + 1e-9:
        # current minimum gap to the sphere at the present (equilibrated) state
        d_now = (torch.linalg.norm(x - torch.tensor([0.0, 0.0, cz]), dim=1) - R)
        gap = float(d_now.min())
        dz = min(step_cap, max(0.2 * dhat, 0.8 * gap)) if gap > 0 else step_cap
        cz = max(cz - dz, cz_final)
        x, gn, dmin, nc = solve_static(X, tets, Dm_inv, W, mu, lam,
                                       (0.0, 0.0, cz), R, dhat, kappa,
                                       fixed_mask, x_init=x)
        it += 1
        if verbose:
            print(f"  [{it:3d}] cz-Lz-R={cz-Lz-R:+.4e} |g|={gn:.2e} contacts={nc} dmin={dmin:.2e}")
        if it > 400:
            print("  (step cap hit)"); break
    return x


def _hertz_test(res=10, depth=0.002, R=0.02, E=1.0e5, nu=0.45, dhat=3e-4, kappa=None):
    print(f"== Stage 3: indentation vs Hertz (res={res}, depth={depth*1000:.1f}mm, R={R*1000:.0f}mm) ==")
    m = build_box_mesh(res=res); m["tets"] = _orient_tets(m["X"], m["tets"])
    X, tets, Lz = m["X"], m["tets"], m["dims"][2]
    Dm_inv, W = precompute_rest(X, tets)
    mu, lam = lame_from_E_nu(E, nu)
    if kappa is None:
        kappa = 1.0e8                                    # barrier stiffness (tuned so sphere bites)
    print(f"  nodes={len(X)} tets={len(tets)} kappa={kappa:.1e} dhat={dhat}")
    x = indent_sphere(X, tets, Dm_inv, W, mu, lam, R, depth, dhat, kappa,
                      m["bottom"], Lz, verbose=True)
    # contact radius: top-surface nodes pushed down by > 5% of max indentation
    u = (x.numpy() - X)
    top = m["top"]
    uz_top = -u[top, 2]                                 # downward displacement
    rad = np.linalg.norm(X[top, :2], axis=1)
    peak = uz_top.max()
    in_contact = uz_top > 0.5 * peak                    # uz(a)=0.5 uz(0) (Hertz edge)
    a_est = rad[in_contact].max() if in_contact.any() else 0.0
    a_hertz = np.sqrt(R * depth)
    print(f"  peak uz={peak*1000:.3f}mm  a_est={a_est*1000:.3f}mm  "
          f"a_hertz=sqrt(R*d)={a_hertz*1000:.3f}mm  rel.err={abs(a_est-a_hertz)/a_hertz:.1%}")
    return a_est, a_hertz


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    print("== Stage 1: box mesh ==")
    m = build_box_mesh(res=6)
    m["tets"] = _orient_tets(m["X"], m["tets"])
    X, tets = m["X"], m["tets"]
    print(f"  nodes={len(X)} tets={len(tets)} grid={m['grid']} "
          f"top_nodes={m['top'].sum()} bottom_nodes={m['bottom'].sum()}")
    Dm_inv, W = precompute_rest(X, tets)
    assert (W > 0).all(), "non-positive rest volumes!"
    vol_mesh = float(W.sum()); vol_true = np.prod(m["dims"])
    print(f"  sum(tet vol)={vol_mesh:.6e}  box vol={vol_true:.6e}  "
          f"rel.err={abs(vol_mesh-vol_true)/vol_true:.2e}")
    assert abs(vol_mesh - vol_true) / vol_true < 1e-9, "volume mismatch"

    print("== Stage 2a: rest state has zero energy & zero force ==")
    E, nu = 1.0e5, 0.45
    mu, lam = lame_from_E_nu(E, nu)
    x0 = torch.tensor(X.copy(), requires_grad=True)
    e0 = neohookean_energy(x0, tets, Dm_inv, W, mu, lam)
    (g0,) = torch.autograd.grad(e0, x0)
    print(f"  E(rest)={float(e0):.3e} (expect ~0)  max|force|={float(g0.abs().max()):.3e}")
    assert abs(float(e0)) < 1e-12 and float(g0.abs().max()) < 1e-9

    print("== Stage 2b: finite-difference gradient check (random perturbation) ==")
    x = torch.tensor(X + 1e-3 * np.random.randn(*X.shape), requires_grad=True)
    e = neohookean_energy(x, tets, Dm_inv, W, mu, lam)
    (g,) = torch.autograd.grad(e, x)
    eps = 1e-7
    idx = [(0, 0), (5, 1), (len(X) // 2, 2)]
    for (i, j) in idx:
        xp = x.detach().clone(); xp[i, j] += eps
        xm = x.detach().clone(); xm[i, j] -= eps
        ep = float(neohookean_energy(xp, tets, Dm_inv, W, mu, lam))
        em = float(neohookean_energy(xm, tets, Dm_inv, W, mu, lam))
        fd = (ep - em) / (2 * eps)
        print(f"  node{i},dim{j}: autograd={float(g[i,j]):+.6e}  FD={fd:+.6e}  "
              f"rel={abs(fd-float(g[i,j]))/(abs(fd)+1e-12):.2e}")

    print("== Stage 2c: uniaxial homogeneous deformation energy ==")
    # apply uniform stretch s in z to ALL nodes -> F = diag(1,1,s) everywhere;
    # total energy must equal Psi(F)*box_volume (homogeneous).
    for s in (0.9, 1.1, 0.7):
        xs = X.copy(); xs[:, 2] *= s
        e = float(neohookean_energy(torch.as_tensor(xs), tets, Dm_inv, W, mu, lam))
        IC = 2.0 + s * s; logJ = np.log(s)
        psi = 0.5 * mu * (IC - 3) - mu * logJ + 0.5 * lam * logJ ** 2
        e_true = psi * vol_true
        print(f"  s={s}: E_mesh={e:.6e}  E_analytic={e_true:.6e}  "
              f"rel.err={abs(e-e_true)/abs(e_true):.2e}")
        assert abs(e - e_true) / abs(e_true) < 1e-9, "uniaxial energy mismatch"
    print("\nALL STAGE 1-2 SELF-TESTS PASSED.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hertz", action="store_true", help="Stage 3 indentation vs Hertz")
    ap.add_argument("--res", type=int, default=10)
    ap.add_argument("--depth", type=float, default=0.002)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--dhat", type=float, default=3e-4)
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.hertz:
        _hertz_test(res=args.res, depth=args.depth, kappa=args.kappa, dhat=args.dhat)
    else:
        print("use --selftest or --hertz")
