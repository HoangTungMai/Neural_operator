#!/usr/bin/env python3
"""Apples-to-apples bake-off: our neural operator (FNO) vs the marker-motion
models of representative VBTS (vision-based tactile sensor) simulators, all
trained and evaluated on the SAME PhysX-FEM ground truth.

Why reimplement rather than cite published numbers: cross-paper accuracy/fps is
not comparable (different sensors, output modalities -- RGB image vs our marker
displacement field, different contacts). The fair test is to give each method's
*marker-motion model* the same FEM training data and the same metric.

Representative VBTS simulators and the marker-motion model we reimplement:
  - TACTO  (Wang et al., RA-L 2022): physics/rendering sim on PyBullet. Its
    marker motion is KINEMATIC -- markers follow the contact surface; tangential
    motion is a rigid drag of the contact patch, with NO friction / stick-slip
    model. -> `TactoKinematic`.
  - Taxim  (Si & Yuan, RA-L 2022) and FOTS (Zhao et al., 2023): example-based /
    fast optical sims whose marker field is a LINEAR ELASTIC SUPERPOSITION model
    (a calibrated linear/Green's-function map from contact load to marker
    displacement). -> `LinearSuperposition` (a single linear, shift-invariant
    conv = the discretised superposition kernel, no nonlinearity).
  - per-point MLP: learned-but-LOCAL lower bound (no global context).
  - FNO (ours): non-local spectral operator.

We reimplement the marker-motion CORE of each simulator (not the optical
renderer) and fit its free parameters on our FEM train split, exactly like we
train the operators. If FNO beats the linear superposition model, that isolates
the value of modelling the *nonlinear, non-local* stick-slip field that the
linear/kinematic VBTS sims cannot represent.

Usage:
  python -m novbts.operator.vbts_baselines --data data/fem/shear_fine_swept_normaug.npz
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from novbts.operator.field2field import (
    FNOField, PerPointMLP, train_operator, predict_raw, throughput,
    rel_l2_per_mode, tangential_dir_error, count_parameters, DEV,
)
from novbts.operator.fem_benchmark import load, norm_from
from novbts.groundtruth.hertz_mindlin import MODE_NAMES, hertz_mindlin_field
from novbts.paths import FEM, RUNS, ensure


class TactoKinematic(nn.Module):
    """TACTO-style marker motion: kinematic and friction-less.

    uz (out-of-plane) <- penetration; in-plane <- (a) rigid shear drag of the
    contact patch + (b) radial push by the penetration surface-slope. There is
    NO stick-slip model: the in-plane field is a uniform rigid drag, missing the
    stuck-centre / slipping-annulus structure FEM produces. The few free scalars
    are fit by the same MSE training loop as the operators.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))   # uz   <- penetration
        self.b = nn.Parameter(torch.tensor(0.0))   # in-plane <- penetration gradient (radial)
        self.g = nn.Parameter(torch.tensor(1.0))   # in-plane <- rigid shear drag

    def forward(self, field, scal):
        pen = field[:, 0:1]
        sx, sy = field[:, 1:2], field[:, 2:3]
        gx = torch.zeros_like(pen); gy = torch.zeros_like(pen)
        gx[..., :, 1:-1] = (pen[..., :, 2:] - pen[..., :, :-2]) * 0.5
        gy[..., 1:-1, :] = (pen[..., 2:, :] - pen[..., :-2, :]) * 0.5
        ux = self.g * sx + self.b * gx
        uy = self.g * sy + self.b * gy
        uz = self.a * pen
        return torch.cat([ux, uy, uz], 1)


class LinearSuperposition(nn.Module):
    """Taxim/FOTS-style linear elastic superposition marker model.

    output field = (input field + broadcast mu,E) convolved with a single learned
    linear, shift-invariant kernel -- the discretised elastic Green's function.
    No activation => the map is exactly linear in the contact input, so it cannot
    represent the nonlinear stick->partial->full slip transition.
    """
    def __init__(self, ksize=31, in_ch=5, out_ch=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, padding=ksize // 2, bias=True)

    def forward(self, field, scal):
        b, c, h, w = field.shape
        sc = scal[:, :, None, None].expand(b, scal.shape[1], h, w)
        return self.conv(torch.cat([field, sc], 1))


# ---------------------------------------------------------------------------
# Newer / more advanced architectures (is FNO still best among modern nets?)
# ---------------------------------------------------------------------------

class DeepONetField(nn.Module):
    """DeepONet (Lu et al., Nat. Mach. Intell. 2021): the principal neural-operator
    paradigm ALTERNATIVE to FNO. Branch net encodes the input function (contact
    field + scalars) into p coefficients per channel; trunk net maps query coords
    to p basis functions; output = <branch, trunk>. Global but via a learned basis
    rather than Fourier modes."""
    def __init__(self, side, in_ch=3, out_ch=3, p=128, hidden=256):
        super().__init__()
        self.side, self.out_ch, self.p = side, out_ch, p
        self.branch = nn.Sequential(
            nn.Linear(in_ch * side * side + 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, p * out_ch))
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p), nn.GELU())
        xs = torch.linspace(-1, 1, side)
        gy, gx = torch.meshgrid(xs, xs, indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy], -1).reshape(-1, 2))
        self.b0 = nn.Parameter(torch.zeros(out_ch))

    def forward(self, field, scal):
        B = field.shape[0]
        coef = self.branch(torch.cat([field.reshape(B, -1), scal], 1)).reshape(B, self.out_ch, self.p)
        basis = self.trunk(self.coords)                                  # [HW, p]
        out = torch.einsum("bop,np->bon", coef, basis).reshape(B, self.out_ch, self.side, self.side)
        return out + self.b0[None, :, None, None]


class UNetField(nn.Module):
    """U-Net (Ronneberger 2015): encoder-decoder CNN with skip connections -- the
    dense-prediction backbone of most recent LEARNED GelSight/marker simulators.
    Local convolutions + skips, no spectral global mixing."""
    def __init__(self, in_ch=3, out_ch=3, w=32):
        super().__init__()
        def C(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GELU())
        self.e1, self.e2, self.e3 = C(in_ch + 2, w), C(w, w * 2), C(w * 2, w * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.d2 = nn.ConvTranspose2d(w * 4, w * 2, 2, 2), C(w * 4, w * 2)
        self.up1, self.d1 = nn.ConvTranspose2d(w * 2, w, 2, 2), C(w * 2, w)
        self.head = nn.Conv2d(w, out_ch, 1)

    def forward(self, field, scal):
        b, c, h, wd = field.shape
        sc = scal[:, :, None, None].expand(b, scal.shape[1], h, wd)
        e1 = self.e1(torch.cat([field, sc], 1))
        e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))         # 32 -> 16 -> 8
        d2 = self.d2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


class GalerkinBlock(nn.Module):
    """Linear (Galerkin-type) attention block: LayerNorm on K,V then Q(KᵀV) -- O(N)
    in tokens, the operator-learning attention of Cao (2021)."""
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))
        self.proj = nn.Linear(dim, dim)
        self.lnk, self.lnv = nn.LayerNorm(self.dh), nn.LayerNorm(self.dh)
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        B, N, D = x.shape
        xn = self.ln1(x)
        q = self.q(xn).view(B, N, self.h, self.dh).transpose(1, 2)
        k = self.lnk(self.k(xn).view(B, N, self.h, self.dh).transpose(1, 2))
        v = self.lnv(self.v(xn).view(B, N, self.h, self.dh).transpose(1, 2))
        kv = torch.einsum("bhnd,bhne->bhde", k, v) / N
        att = torch.einsum("bhnd,bhde->bhne", q, kv).transpose(1, 2).reshape(B, N, D)
        x = x + self.proj(att)
        return x + self.ff(self.ln2(x))


class GalerkinOperator(nn.Module):
    """Galerkin-attention Transformer operator (Cao 2021; OFormer, Li 2023): the
    transformer-based neural-operator family -- global mixing via linear attention
    over grid tokens instead of Fourier modes. A newer paradigm than spectral FNO."""
    def __init__(self, side, in_ch=3, out_ch=3, dim=96, heads=4, layers=4):
        super().__init__()
        self.side = side
        self.inp = nn.Linear(in_ch + 2 + 2, dim)                          # +2 pos +2 scalar
        self.blocks = nn.ModuleList([GalerkinBlock(dim, heads) for _ in range(layers)])
        self.out = nn.Linear(dim, out_ch)
        xs = torch.linspace(-1, 1, side)
        gy, gx = torch.meshgrid(xs, xs, indexing="ij")
        self.register_buffer("pos", torch.stack([gx, gy], -1).reshape(-1, 2))

    def forward(self, field, scal):
        b, c, h, w = field.shape
        tok = field.permute(0, 2, 3, 1).reshape(b, h * w, c)
        pos = self.pos[None].expand(b, -1, -1)
        sc = scal[:, None, :].expand(b, h * w, scal.shape[1])
        x = self.inp(torch.cat([tok, pos, sc], -1))
        for blk in self.blocks:
            x = blk(x)
        return self.out(x).reshape(b, h, w, -1).permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FEM / "shear_fine_swept_normaug.npz"))
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ksize", type=int, default=31)
    args = ap.parse_args()

    D = load(args.data)
    side, N, nt = D["side"], D["inp"].shape[0], args.n_test
    print(f"device={DEV}  VBTS bake-off  data={args.data}  N={N} side={side}  test=last {nt}")
    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1
    ).astype(np.float32)).to(DEV)
    inp, out, scal, mode = (D[k].to(DEV) for k in ("inp", "out", "scal", "mode"))

    tr = torch.arange(0, N - nt, device=DEV)
    te = torch.arange(N - nt, N, device=DEV)
    im, istd, om, ostd, sm, sstd = norm_from(inp[tr], out[tr], scal[tr])
    nin = lambda t: (t - im) / istd
    nout = lambda t: (t - om) / ostd
    nsc = lambda t: (t - sm) / sstd

    def fit(model, is_mlp=False):
        torch.manual_seed(0)
        model = model.to(DEV)
        secs, _ = train_operator(model, nin(inp[tr]), nout(out[tr]), nsc(scal[tr]),
                                 mode[tr], cg, args.epochs, args.lr, is_mlp=is_mlp)
        return model, secs

    def evaluate(model, is_mlp=False):
        pred = predict_raw(model, nin(inp[te]), nsc(scal[te]), cg, ostd, om, is_mlp=is_mlp)
        return {"relative_l2": rel_l2_per_mode(pred, out[te], mode[te]),
                "tangential_dir_error_deg": tangential_dir_error(pred, out[te], mode[te]),
                "throughput_fps": throughput(model, nin(inp[te]), nsc(scal[te]), cg, is_mlp=is_mlp),
                "params": count_parameters(model)}

    # representative VBTS marker-motion models + our operator (+ MLP lower bound)
    specs = [
        ("tacto_kinematic", lambda: TactoKinematic(), False,
         "TACTO-style: kinematic, friction-less (rigid shear drag, no stick-slip)"),
        ("taxim_fots_linear", lambda: LinearSuperposition(args.ksize), False,
         "Taxim/FOTS-style: linear elastic superposition (one linear conv kernel)"),
        ("mlp_perpoint", lambda: PerPointMLP(), True,
         "per-point MLP: learned but local (lower bound)"),
        ("fno_ours", lambda: FNOField(modes=args.modes), False,
         "ours: non-local spectral neural operator"),
    ]

    # newer / more advanced neural architectures (is FNO still best among modern nets?)
    adv_specs = [
        ("deeponet", lambda: DeepONetField(side), False,
         "DeepONet (Lu 2021): branch-trunk operator, the alternative operator paradigm"),
        ("unet", lambda: UNetField(), False,
         "U-Net (2015): CNN encoder-decoder w/ skips, backbone of recent learned tactile sims"),
        ("galerkin_transformer", lambda: GalerkinOperator(side), False,
         "Galerkin-attention Transformer operator (Cao 2021 / OFormer 2023): transformer-based SOTA"),
    ]

    results = {}
    for key, ctor, is_mlp, desc in specs + adv_specs:
        model, secs = fit(ctor(), is_mlp=is_mlp)
        r = evaluate(model, is_mlp=is_mlp)
        r["train_s"] = round(secs, 1)
        r["desc"] = desc
        results[key] = r
        print(f"[{key:18s}] trained {secs:5.0f}s  params={r['params']:>9d}  "
              f"overall={r['relative_l2']['overall']:.3f}  tang_dir={r['tangential_dir_error_deg']:5.1f}°")

    # ---- first-principles physics baseline: Cattaneo-Mindlin analytic contact model ----
    # The textbook contact-mechanics field (Hertz normal + Cattaneo-Mindlin partial
    # slip) underlying analytical tactile sims. Its amplitude is in its own units, so
    # we give it the FAIREST shot: a per-channel affine (scale+bias) fit by least
    # squares on the train split -> isolates whether the analytic field STRUCTURE
    # (not amplitude) matches FEM, i.e. what the neural operator adds over textbook physics.
    coords_np = np.load(args.data, allow_pickle=True)["coords"]
    params_np = D["params"] if isinstance(D["params"], np.ndarray) else np.asarray(D["params"])
    disp_a, _ = hertz_mindlin_field(params_np, coords_np)            # [N,M,3]
    af = torch.tensor(disp_a.reshape(-1, side, side, 3).transpose(0, 3, 1, 2)).to(DEV)
    tr_np, te_np = tr.cpu().numpy(), te.cpu().numpy()
    pred_a = torch.zeros_like(af[te])
    for c in range(3):                                               # per-channel scale+bias
        A = af[tr, c].reshape(-1); Y = out[tr, c].reshape(-1)
        M = torch.stack([A, torch.ones_like(A)], 1)
        wb = torch.linalg.lstsq(M, Y.unsqueeze(1)).solution.squeeze(1)
        pred_a[:, c] = wb[0] * af[te, c] + wb[1]
    import time as _t
    _ = hertz_mindlin_field(params_np[te_np], coords_np)             # warm
    t0 = _t.perf_counter()
    for _ in range(5):
        hertz_mindlin_field(params_np[te_np], coords_np)
    cm_fps = (5 * len(te_np)) / (_t.perf_counter() - t0)
    results["cattaneo_mindlin_analytic"] = {
        "relative_l2": rel_l2_per_mode(pred_a, out[te], mode[te]),
        "tangential_dir_error_deg": tangential_dir_error(pred_a, out[te], mode[te]),
        "throughput_fps": cm_fps, "params": 6,
        "desc": "first-principles Hertz + Cattaneo-Mindlin analytic, per-channel affine-calibrated to FEM"}
    rc = results["cattaneo_mindlin_analytic"]
    print(f"[cattaneo_mindlin   ] analytic (calibrated)  overall={rc['relative_l2']['overall']:.3f}  "
          f"tang_dir={rc['tangential_dir_error_deg']:5.1f}°")

    fno = results["fno_ours"]["relative_l2"]["overall"]
    summary = {
        "gt": args.data, "side": side, "train_frames": int(N - nt), "test_frames": nt,
        "note": ("VBTS-simulator marker-motion models reimplemented and fit on our FEM GT. "
                 "TACTO=kinematic/friction-less; Taxim/FOTS=linear elastic superposition; "
                 "cattaneo_mindlin=first-principles analytic contact physics (affine-calibrated); "
                 "MLP=local lower bound; FNO=ours (non-local). Cross-paper numbers are not "
                 "comparable (RGB image vs marker field), hence this in-house bake-off. Running "
                 "the originals' pre-calibrated weights on our gel would be wrong-sensor, not "
                 "fairer; the fair route-A is fitting each model's form to our data, as done here."),
        "models": results,
        "groups": {
            "vbts_sim_marker_models": ["tacto_kinematic", "cattaneo_mindlin_analytic",
                                       "taxim_fots_linear", "mlp_perpoint", "fno_ours"],
            "advanced_neural_architectures": ["deeponet", "unet", "galerkin_transformer", "fno_ours"],
        },
        "fno_advantage_x": {k: round(results[k]["relative_l2"]["overall"] / fno, 2)
                            for k in results if k != "fno_ours"},
    }
    phase_dir = RUNS / "phase3_fem"
    ensure(phase_dir)
    out_path = phase_dir / "vbts_baselines.json"
    json.dump(summary, open(out_path, "w"), indent=2)

    def print_table(title, order):
        print(f"\n=== {title} (rel L2 on FEM GT, lower=better) ===")
        print(f"{'method':22s} {'overall':>8s} {'normal':>7s} {'stick':>7s} {'partial':>8s} "
              f"{'full':>6s} {'tang°':>6s} {'fps':>7s} {'params':>9s}")
        for k in order:
            r = results[k]; L = r["relative_l2"]
            print(f"{k:22s} {L['overall']:8.3f} {L['normal']:7.3f} {L['stick']:7.3f} "
                  f"{L['partial_slip']:8.3f} {L['full_slip']:6.3f} "
                  f"{r['tangential_dir_error_deg']:6.1f} {r['throughput_fps']:7.0f} {r['params']:9d}")

    print_table("§6d VBTS-sim marker-motion models", summary["groups"]["vbts_sim_marker_models"])
    print_table("§6e newer/advanced neural architectures", summary["groups"]["advanced_neural_architectures"])
    print("\nFNO advantage (their overall rel L2 / ours):")
    for k, v in summary["fno_advantage_x"].items():
        print(f"  {k:22s} {v:.2f}x")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
