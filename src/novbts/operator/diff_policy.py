#!/usr/bin/env python3
"""Phase 4 -- end-to-end policy learning through the differentiable FNO.

Headline question: does backprop through the differentiable FNO train a tactile
CONTROL policy more sample-efficiently / to better final loss than a GRADIENT-FREE
method (evolution strategies) that treats the same frozen FNO as a black box? And,
honestly, does the differentiable gradient degrade on the sharp stick->slip frames
(Suh et al. 2022, "Do Differentiable Simulators Give Better Policy Gradients?")?

The FNO is a STATIC one-step input->field map, so this is a single-step CONTEXTUAL
policy (amortized optimization), NOT multi-step RL with dynamics.

Two paths share one trained+frozen FNO:
  --probe         : sweep depth/|shear| to check the slip-control lever (gates task A)
  --train-policy  : the autograd-vs-ES policy bake-off, --task {servo,antislip}

Task B (servo, primary, robust): policy pi(context + target-field summary) -> action
  (sx,sy); loss = ||FNO(action) - y*||^2 read from the FNO OUTPUT field. It is the
  amortized version of the per-instance inverse in inverse_demo.py.
Task A (antislip, probe-gated): policy pi(mu,R,E,disturbance) -> depth; loss reads
  slip_proxy(FNO field). Only runs if --probe reported depth_is_lever.

  python -m novbts.operator.diff_policy --probe
  python -m novbts.operator.diff_policy --train-policy --task servo --n-seeds 3
"""
import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from novbts.operator.field2field import (
    FNOField, train_operator, params_to_fieldinput, count_parameters, DEV,
)
from novbts.operator.fem_benchmark import load, norm_from
from novbts.groundtruth.hertz_mindlin import hertz_mindlin_field, hertz_scalars
from novbts.paths import FEM, RUNS, ensure


# ---------------------------------------------------------------------------
# shared building blocks
# ---------------------------------------------------------------------------

def contact_mask(params, coords, side):
    """[N,H,W] mask r <= contact radius (sphere: Hertz a; flat: R)."""
    p = np.asarray(params, dtype=np.float64)
    x0, y0, depth, R, sx, sy, mu, E, geom = [p[:, i] for i in range(9)]
    X = coords[:, 0].reshape(side, side); Y = coords[:, 1].reshape(side, side)
    r = np.sqrt((X[None] - x0[:, None, None]) ** 2 + (Y[None] - y0[:, None, None]) ** 2 + 1e-12)
    a, _, _, _ = hertz_scalars(depth, R, E)
    a_eff = np.where(geom > 0.5, R, a)[:, None, None]
    return (r <= a_eff).astype(np.float32)


def slip_proxy(field, mask):
    """field [N,3,H,W] (ux,uy,uz), mask [N,H,W] -> [N] mean |u_tangential| over contact."""
    tmag = torch.sqrt(field[:, 0] ** 2 + field[:, 1] ** 2 + 1e-12)
    return (tmag * mask).sum((1, 2)) / (mask.sum((1, 2)) + 1e-9)


class Setup:
    """Trains the forward FNO once (seed 0) and freezes it; holds data + norms."""
    def __init__(self, args):
        self.D = load(args.data)
        self.side = self.D["side"]
        self.N = self.D["inp"].shape[0]
        self.nt = args.n_test
        self.coords = np.load(args.data, allow_pickle=True)["coords"]
        self.P = self.D["params"]
        inp, out, scal, mode = (self.D[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
        self.inp, self.out, self.scal, self.mode = inp, out, scal, mode
        self.tr = torch.arange(0, self.N - self.nt, device=DEV)
        self.te = torch.arange(self.N - self.nt, self.N, device=DEV)
        self.im, self.istd, self.om, self.ostd, self.sm, self.sstd = norm_from(
            inp[self.tr], out[self.tr], scal[self.tr])
        self.cg = torch.tensor(np.stack(np.meshgrid(
            np.linspace(-1, 1, self.side), np.linspace(-1, 1, self.side), indexing="ij")[::-1], -1
        ).astype(np.float32)).to(DEV)
        print(f"device={DEV}  phase4  data={args.data}  N={self.N} side={self.side}")
        torch.manual_seed(0)
        self.fno = FNOField(modes=args.modes).to(DEV)
        secs, _ = train_operator(self.fno, self.nin(inp[self.tr]), self.nout(out[self.tr]),
                                 self.nsc(scal[self.tr]), mode[self.tr], self.cg,
                                 args.epochs, args.lr)
        self.fno_params = count_parameters(self.fno)      # count BEFORE freezing
        self.fno.eval()
        for p in self.fno.parameters():
            p.requires_grad_(False)                       # frozen: gradients flow to ACTION only
        self.fno_train_s = secs
        print(f"[FNO] trained {secs:.0f}s  ({self.fno_params} params)  [frozen]\n")

    def nin(self, t): return (t - self.im) / self.istd
    def nout(self, t): return (t - self.om) / self.ostd
    def nsc(self, t): return (t - self.sm) / self.sstd


# ---------------------------------------------------------------------------
# differentiable FNO environment (action -> predicted field -> loss)
# ---------------------------------------------------------------------------

def context_tensors(S, idx):
    """For frame indices idx, precompute the action-independent pieces:
      pen[B,H,W], mask[B,H,W], scal[B,2], ystar[B,3,H,W] (raw FEM target),
      params_t[B,9], mode[B]. pen/mask depend on geometry (not on the (sx,sy) action),
      so the env is linear & differentiable in the action."""
    P = S.P[idx.cpu().numpy()]
    inp_np, scal_np = params_to_fieldinput(P, S.coords, S.side)
    pen = torch.tensor(inp_np[:, 0], device=DEV)                 # [B,H,W]
    mask = torch.tensor(contact_mask(P, S.coords, S.side), device=DEV)
    scal = torch.tensor(scal_np, device=DEV)
    ystar = S.out[idx]                                           # raw FEM field [B,3,H,W]
    return {"pen": pen, "mask": mask, "scal": scal, "ystar": ystar,
            "params": torch.tensor(P, dtype=torch.float32, device=DEV), "mode": S.mode[idx]}


def assemble_input(S, action, ctx):
    """action [B,2] (sx,sy) + context -> NORMALISED FNO input [B,3,H,W].
    Differentiable in `action`; pen/mask are constants. Mirrors params_to_fieldinput."""
    pen, mask = ctx["pen"], ctx["mask"]
    ch1 = action[:, 0:1].unsqueeze(-1) * mask                   # [B,1,1]*[B,H,W] -> [B,H,W]
    ch2 = action[:, 1:2].unsqueeze(-1) * mask
    inp = torch.stack([pen, ch1, ch2], 1)                       # [B,3,H,W]
    return (inp - S.im) / S.istd


def fno_field(S, action, ctx):
    """-> predicted RAW marker field [B,3,H,W]. One FNO forward = one env query."""
    pred = S.fno(assemble_input(S, action, ctx), S.nsc(ctx["scal"]))
    return pred * S.ostd + S.om


def servo_loss(pred_field, ystar, action, lam):
    """field-matching loss (reads the FNO OUTPUT field) + optional action effort."""
    mse = ((pred_field - ystar) ** 2).mean(dim=(1, 2, 3))       # [B]
    return mse + lam * (action ** 2).mean(dim=1)


# ---------------------------------------------------------------------------
# policy + context features
# ---------------------------------------------------------------------------

class PolicyMLP(nn.Module):
    """context -> action, tanh-squashed into the physical action box [-a_scale, a_scale]."""
    def __init__(self, ctx_dim, action_dim, a_scale, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )
        self.register_buffer("a_scale", a_scale)

    def forward(self, ctx_feat):
        return torch.tanh(self.net(ctx_feat)) * self.a_scale


def context_features(ctx):
    """policy input: object scalars (mu,E,R,geom) + a COMPACT summary of the target
    field (mean ux,uy,uz over the contact). The policy must learn the summary->action
    map; the loss still reads the full field via the FNO."""
    P, mask, y = ctx["params"], ctx["mask"], ctx["ystar"]
    msum = mask.sum((1, 2)) + 1e-9
    mux = (y[:, 0] * mask).sum((1, 2)) / msum
    muy = (y[:, 1] * mask).sum((1, 2)) / msum
    muz = (y[:, 2] * mask).sum((1, 2)) / msum
    return torch.stack([P[:, 6], P[:, 7], P[:, 3], P[:, 8], mux, muy, muz], 1)   # [B,7]


# ---------------------------------------------------------------------------
# training: autograd vs evolution strategies (same policy, same Adam)
# ---------------------------------------------------------------------------

def _flat(policy):
    return torch.cat([p.data.view(-1) for p in policy.parameters()])


def _set_flat(policy, flat):
    i = 0
    for p in policy.parameters():
        n = p.numel(); p.data.copy_(flat[i:i + n].view_as(p)); i += n


def _val_loss(S, policy, feat, ctx, lam):
    with torch.no_grad():
        a = policy(feat)
        return float(servo_loss(fno_field(S, a, ctx), ctx["ystar"], a, lam).mean())


def train_policy_autograd(S, seed, feat_tr, ctx_tr, feat_va, ctx_va, args):
    torch.manual_seed(seed)
    policy = PolicyMLP(feat_tr.shape[1], 2, args.a_scale).to(DEV)
    opt = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    n = feat_tr.shape[0]; bs = args.bs
    curve, qn = [], 0
    t0 = time.perf_counter()
    for step in range(args.steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        a = policy(feat_tr[idx])
        ctx_b = {k: v[idx] for k, v in ctx_tr.items()}
        loss = servo_loss(fno_field(S, a, ctx_b), ctx_b["ystar"], a, args.lambda_reg).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        qn += 1                                            # 1 forward query / step
        if step % args.log_every == 0 or step == args.steps - 1:
            curve.append((qn, _val_loss(S, policy, feat_va, ctx_va, args.lambda_reg)))
    wall = time.perf_counter() - t0
    return {"policy": policy, "curve": curve, "final_loss": curve[-1][1],
            "forward_evals": qn, "backward_evals": qn, "wall_s": round(wall, 2)}


def train_policy_es(S, seed, feat_tr, ctx_tr, feat_va, ctx_va, args):
    """OpenAI-ES with antithetic sampling; same Adam on the estimated gradient.
    1 step = 2*pop FORWARD queries, 0 backward."""
    torch.manual_seed(seed)
    policy = PolicyMLP(feat_tr.shape[1], 2, args.a_scale).to(DEV)
    opt = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    n = feat_tr.shape[0]; bs = args.bs; sigma, pop = args.es_sigma, args.es_pop
    curve, qn = [], 0
    t0 = time.perf_counter()

    def batch_loss(idx):
        with torch.no_grad():
            a = policy(feat_tr[idx])
            ctx_b = {k: v[idx] for k, v in ctx_tr.items()}
            return float(servo_loss(fno_field(S, a, ctx_b), ctx_b["ystar"], a, args.lambda_reg).mean())

    P = _flat(policy).numel()
    for step in range(args.steps):
        idx = torch.randint(0, n, (bs,), device=DEV)
        theta = _flat(policy)
        g = torch.zeros(P, device=DEV)
        for _ in range(pop):
            eps = torch.randn(P, device=DEV)
            _set_flat(policy, theta + sigma * eps); lp = batch_loss(idx)
            _set_flat(policy, theta - sigma * eps); lm = batch_loss(idx)
            g += (lp - lm) * eps
            qn += 2                                        # antithetic pair = 2 forwards
        g /= (2 * sigma * pop)
        _set_flat(policy, theta)
        i = 0
        for p in policy.parameters():
            nn_ = p.numel(); p.grad = g[i:i + nn_].view_as(p).clone(); i += nn_
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            curve.append((qn, _val_loss(S, policy, feat_va, ctx_va, args.lambda_reg)))
    wall = time.perf_counter() - t0
    return {"policy": policy, "curve": curve, "final_loss": curve[-1][1],
            "forward_evals": qn, "backward_evals": 0, "wall_s": round(wall, 2)}


# ---------------------------------------------------------------------------
# references + diagnostics
# ---------------------------------------------------------------------------

def per_instance_reference(S, feat_va, ctx_va, args):
    """Oracle floor: optimise the action directly per context (inverse_demo style),
    no policy. The best loss the amortized policy could hope to match."""
    a = torch.zeros(ctx_va["ystar"].shape[0], 2, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([a], lr=args.a_scale.max().item() * 0.05)
    for _ in range(args.steps):
        loss = servo_loss(fno_field(S, a, ctx_va), ctx_va["ystar"], a, args.lambda_reg).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        return float(servo_loss(fno_field(S, a, ctx_va), ctx_va["ystar"], a, args.lambda_reg).mean())


def analytic_baseline(S, ctx_tr, ctx_va, args):
    """Degenerate baseline: predict the mean training action for every context."""
    mean_a = ctx_tr["params"][:, 4:6].mean(0, keepdim=True)
    a = mean_a.expand(ctx_va["ystar"].shape[0], 2)
    with torch.no_grad():
        return float(servo_loss(fno_field(S, a, ctx_va), ctx_va["ystar"], a, args.lambda_reg).mean())


def grad_variance_probe(S, feat, ctx, args, n_draws=24):
    """Compare exact autograd gradient vs ES estimates (bias+variance) at a fixed
    random policy, SPLIT by stored slip mode (stick<=1 vs slip>=2) -- the honest
    Suh-2022 check on whether sharp-contact frames hurt the differentiable gradient."""
    torch.manual_seed(0)
    out = {}
    for tag, sel in [("stick", ctx["mode"] <= 1), ("slip", ctx["mode"] >= 2)]:
        if sel.sum() < 8:
            out[tag] = {"n": int(sel.sum()), "note": "too few frames"}
            continue
        f = feat[sel]; c = {k: v[sel] for k, v in ctx.items()}
        policy = PolicyMLP(f.shape[1], 2, args.a_scale).to(DEV)
        # exact autograd gradient (flattened)
        a = policy(f)
        loss = servo_loss(fno_field(S, a, c), c["ystar"], a, args.lambda_reg).mean()
        g_auto = torch.cat([gg.view(-1) for gg in torch.autograd.grad(loss, policy.parameters())])
        # ES estimates
        theta = _flat(policy); P = theta.numel()
        ests = []
        for _ in range(n_draws):
            g = torch.zeros(P, device=DEV)
            for _ in range(args.es_pop):
                eps = torch.randn(P, device=DEV)
                _set_flat(policy, theta + args.es_sigma * eps)
                with torch.no_grad():
                    ap = policy(f); lp = float(servo_loss(fno_field(S, ap, c), c["ystar"], ap, args.lambda_reg).mean())
                _set_flat(policy, theta - args.es_sigma * eps)
                with torch.no_grad():
                    am = policy(f); lm = float(servo_loss(fno_field(S, am, c), c["ystar"], am, args.lambda_reg).mean())
                g += (lp - lm) * eps
            _set_flat(policy, theta)
            ests.append(g / (2 * args.es_sigma * args.es_pop))
        E = torch.stack(ests)
        es_mean = E.mean(0)
        bias = float((es_mean - g_auto).norm() / (g_auto.norm() + 1e-12))
        var = float(((E - es_mean) ** 2).sum(1).mean())
        out[tag] = {"n": int(sel.sum()), "grad_norm_autograd": float(g_auto.norm()),
                    "es_bias_rel": bias, "es_var": var}
    return out


def eval_policy(S, policy, feat, ctx, args):
    """final loss overall + per slip regime (stored mode)."""
    with torch.no_grad():
        a = policy(feat)
        per = servo_loss(fno_field(S, a, ctx), ctx["ystar"], a, args.lambda_reg)   # [B]
    res = {"overall": float(per.mean())}
    for tag, sel in [("stick", ctx["mode"] <= 1), ("slip", ctx["mode"] >= 2)]:
        res[tag] = float(per[sel].mean()) if sel.any() else float("nan")
    return res


def plot_training_curves(results, floor, baseline, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, c in [("autograd", "tab:blue"), ("es", "tab:orange")]:
            cur = np.array(results[name]["curve_mean"])
            ax.plot(cur[:, 0], cur[:, 1], "-o", ms=3, color=c, label=f"{name} (policy)")
        ax.axhline(floor, ls="--", color="green", label="per-instance oracle floor")
        ax.axhline(baseline, ls=":", color="gray", label="mean-action baseline")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cumulative FNO-env forward queries (log)")
        ax.set_ylabel("validation loss (log)")
        ax.set_title("Phase 4: differentiable (autograd) vs gradient-free (ES) policy")
        ax.grid(True, which="both", alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(out_path, dpi=130)
        return str(out_path)
    except Exception as e:
        return f"plot_error: {e}"


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def run_probe(S, args):
    coords, side, P, N, nt = S.coords, S.side, S.P, S.N, S.nt

    @torch.no_grad()
    def fno_proxy(rows):
        inp_np, scal_np = params_to_fieldinput(rows, coords, side)
        pred = S.fno(S.nin(torch.tensor(inp_np, device=DEV)),
                     S.nsc(torch.tensor(scal_np, device=DEV))) * S.ostd + S.om
        return slip_proxy(pred, torch.tensor(contact_mask(rows, coords, side), device=DEV)).cpu().numpy()

    base = np.median(P[:N - nt], 0)
    R_med, mu_med, E_med = base[3], base[6], base[7]
    d_lo, d_hi = np.quantile(P[:, 2], 0.05), np.quantile(P[:, 2], 0.95)
    s_hi = float(np.quantile(np.hypot(P[:, 4], P[:, 5]), 0.95))
    print(f"context: R={R_med*1e3:.1f}mm mu={mu_med:.2f} E={E_med:.0f}  "
          f"depth[{d_lo*1e3:.2f},{d_hi*1e3:.2f}]mm  |shear|[0,{s_hi*1e3:.2f}]mm")
    rep = {"context": {"R": float(R_med), "mu": float(mu_med), "E": float(E_med)}, "sweeps": {}}

    depths = np.linspace(d_lo, d_hi, args.sweep_n)
    rows = np.tile(base, (args.sweep_n, 1)); rows[:, 2] = depths; rows[:, 4] = s_hi; rows[:, 5] = 0.0
    pf = fno_proxy(rows)
    rep["sweeps"]["depth"] = {"depth_mm": (depths * 1e3).tolist(), "fno_proxy": pf.tolist()}
    print("\n--- sweep DEPTH (action), shear fixed high ---")
    for d, a in zip(depths, pf):
        print(f"  depth={d*1e3:6.3f}mm  proxy={a:.5f}")

    shears = np.linspace(0.0, s_hi, args.sweep_n)
    rows2 = np.tile(base, (args.sweep_n, 1)); rows2[:, 4] = shears; rows2[:, 5] = 0.0
    pf2 = fno_proxy(rows2)
    rep["sweeps"]["shear"] = {"shear_mm": (shears * 1e3).tolist(), "fno_proxy": pf2.tolist()}

    depth_resp = abs(pf.max() - pf.min()) / (pf.mean() + 1e-9)
    shear_resp = abs(pf2.max() - pf2.min()) / (pf2.mean() + 1e-9)
    rep["verdict"] = {"depth_response_frac": float(depth_resp), "shear_response_frac": float(shear_resp),
                      "depth_is_lever": bool(depth_resp > 0.05),
                      "note": "proxy rises with depth (contact-area effect, not anti-slip); "
                              "slip_proxy is a confounded slip measure -- see task-A caveat."}
    print(f"\nVERDICT: depth response={depth_resp*100:.0f}%  shear response={shear_resp*100:.0f}%  "
          f"depth_is_lever={rep['verdict']['depth_is_lever']}")
    phase_dir = RUNS / "phase4"; ensure(phase_dir)
    json.dump(rep, open(phase_dir / "probe.json", "w"), indent=2, default=float)
    print(f"Saved {phase_dir / 'probe.json'}")


def run_policy(S, args):
    assert args.task == "servo", "antislip task is probe-gated; see plan (only servo implemented as primary)"
    # action box from data
    s_abs = float(np.abs(S.P[:, 4:6]).max())
    args.a_scale = torch.tensor([s_abs, s_abs], dtype=torch.float32, device=DEV)

    ctx_tr = context_tensors(S, S.tr)
    ctx_va = context_tensors(S, S.te)
    feat_tr_raw = context_features(ctx_tr)
    feat_va_raw = context_features(ctx_va)
    fm = feat_tr_raw.mean(0, keepdim=True); fs = feat_tr_raw.std(0, keepdim=True).clamp_min(1e-6)
    feat_tr = (feat_tr_raw - fm) / fs
    feat_va = (feat_va_raw - fm) / fs

    floor = per_instance_reference(S, feat_va, ctx_va, args)
    base = analytic_baseline(S, ctx_tr, ctx_va, args)
    print(f"references: per-instance oracle floor={floor:.3e}  mean-action baseline={base:.3e}\n")

    methods = {"autograd": train_policy_autograd, "es": train_policy_es}
    agg = {m: {"final": [], "curves": [], "wall": [], "fwd": [], "bwd": [], "eval": []}
           for m in methods}
    for seed in range(args.n_seeds):
        for m, fn in methods.items():
            r = fn(S, seed, feat_tr, ctx_tr, feat_va, ctx_va, args)
            agg[m]["final"].append(r["final_loss"]); agg[m]["curves"].append(r["curve"])
            agg[m]["wall"].append(r["wall_s"]); agg[m]["fwd"].append(r["forward_evals"])
            agg[m]["bwd"].append(r["backward_evals"])
            agg[m]["eval"].append(eval_policy(S, r["policy"], feat_va, ctx_va, args))
        print(f"  seed {seed}: autograd final={agg['autograd']['final'][-1]:.3e}  "
              f"es final={agg['es']['final'][-1]:.3e}")

    def ms(xs):
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs))}

    results = {}
    for m in methods:
        # curves share query x-grid across seeds (same schedule) -> average y
        curves = np.array(agg[m]["curves"])               # [seeds, npts, 2]
        curve_mean = np.stack([curves[0, :, 0], curves[:, :, 1].mean(0)], 1).tolist()
        # queries to reach a target an AMORTIZED policy can realistically hit:
        # close 40% of the baseline->oracle-floor gap (per-instance floor is the
        # un-amortizable ideal, so 1.5x-floor is unreachable by one network).
        target = base - 0.4 * (base - floor)
        q2t = []
        for s in range(args.n_seeds):
            hit = [q for q, l in agg[m]["curves"][s] if l <= target]
            q2t.append(hit[0] if hit else float("inf"))
        ev = {k: ms([e[k] for e in agg[m]["eval"]]) for k in agg[m]["eval"][0]}
        results[m] = {"final_loss": ms(agg[m]["final"]), "wall_s": ms(agg[m]["wall"]),
                      "forward_evals": int(np.mean(agg[m]["fwd"])), "backward_evals": int(np.mean(agg[m]["bwd"])),
                      "queries_to_target": [None if np.isinf(q) else int(q) for q in q2t],
                      "curve_mean": curve_mean, "eval_by_regime": ev}

    gradvar = grad_variance_probe(S, feat_va, ctx_va, args)

    # verdict
    a_q = [q for q in results["autograd"]["queries_to_target"] if q is not None]
    e_q = [q for q in results["es"]["queries_to_target"] if q is not None]
    a_fin, e_fin = results["autograd"]["final_loss"]["mean"], results["es"]["final_loss"]["mean"]
    if a_q and e_q:
        ratio = np.mean(e_q) / max(np.mean(a_q), 1)
        eff = (f"autograd reaches target in {np.mean(a_q):.0f} fwd-queries vs ES {np.mean(e_q):.0f} "
               f"({ratio:.0f}x fewer); wall {results['autograd']['wall_s']['mean']:.0f}s vs "
               f"{results['es']['wall_s']['mean']:.0f}s")
    elif a_q and not e_q:
        eff = f"autograd reaches target in {np.mean(a_q):.0f} fwd-queries; ES never reaches it within budget"
    else:
        eff = "neither reached target within budget"
    win = "WINS" if (a_fin < e_fin) else ("LOSES" if a_fin > e_fin * 1.05 else "MIXED")
    verdict = {"differentiable": win, "sample_efficiency": eff,
               "autograd_final": a_fin, "es_final": e_fin, "floor": floor, "baseline": base}

    out = {"task": args.task, "device": str(DEV), "train_frames": int(S.N - S.nt),
           "side": S.side, "n_seeds": args.n_seeds,
           "fno": {"train_s": round(S.fno_train_s, 1), "params": S.fno_params},
           "policy_params": count_parameters(PolicyMLP(feat_tr.shape[1], 2, args.a_scale)),
           "es_cfg": {"pop": args.es_pop, "sigma": args.es_sigma},
           "references": {"per_instance_floor": floor, "mean_action_baseline": base},
           "autograd": results["autograd"], "es": results["es"],
           "grad_variance": gradvar, "verdict": verdict}

    phase_dir = RUNS / "phase4"; ensure(phase_dir)
    out["plot"] = plot_training_curves(results, floor, base, phase_dir / f"policy_{args.task}_curve.png")
    json.dump(out, open(phase_dir / f"policy_{args.task}.json", "w"), indent=2, default=float)

    # ---- printout ----
    print("\n=== policy training: differentiable (autograd) vs gradient-free (ES) ===")
    print(f"{'method':12s} {'final_loss':>20s} {'queries(fwd)':>13s} {'bwd':>8s} {'wall_s':>9s}")
    for m in ["autograd", "es"]:
        r = results[m]
        print(f"{m:12s} {r['final_loss']['mean']:.3e}±{r['final_loss']['std']:.1e}   "
              f"{r['forward_evals']:13d} {r['backward_evals']:8d} {r['wall_s']['mean']:9.2f}")
    print(f"{'per_instance':12s} {floor:.3e}  (oracle floor)")
    print(f"{'mean_action':12s} {base:.3e}  (baseline)")
    print(f"\nsample efficiency: {eff}")
    print("\n=== counter-thesis (Suh 2022) gradient probe, by stored slip mode ===")
    for tag in ["stick", "slip"]:
        g = gradvar.get(tag, {})
        if "es_bias_rel" in g:
            print(f"  {tag:6s} n={g['n']:4d}  ||g_auto||={g['grad_norm_autograd']:.2e}  "
                  f"ES bias={g['es_bias_rel']:.2f}  ES var={g['es_var']:.2e}")
        else:
            print(f"  {tag:6s} {g.get('note','')}")
    print("\n=== eval by regime (final policy loss) ===")
    for m in ["autograd", "es"]:
        ev = results[m]["eval_by_regime"]
        print(f"  {m:9s} overall={ev['overall']['mean']:.3e}  "
              f"stick={ev['stick']['mean']:.3e}  slip={ev['slip']['mean']:.3e}")
    print(f"\nVERDICT: differentiable {win}: {eff}")
    print(f"Saved {phase_dir / f'policy_{args.task}.json'} + {args.task}_curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FEM / "shear_fine_swept_normaug.npz"))
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--sweep-n", type=int, default=12)
    ap.add_argument("--train-policy", action="store_true")
    ap.add_argument("--task", default="servo", choices=["servo", "antislip"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--policy-lr", type=float, default=1e-2)
    ap.add_argument("--lambda-reg", type=float, default=0.0)
    ap.add_argument("--es-pop", type=int, default=32)
    ap.add_argument("--es-sigma", type=float, default=0.02)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()

    S = Setup(args)
    if args.probe:
        run_probe(S, args)
    if args.train_policy:
        run_policy(S, args)
    if not args.probe and not args.train_policy:
        print("nothing to do: pass --probe and/or --train-policy")


if __name__ == "__main__":
    main()
