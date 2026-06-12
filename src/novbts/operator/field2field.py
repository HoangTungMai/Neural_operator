#!/usr/bin/env python3
"""
Phase 3 — FULL pipeline in the field->field operator framing.

This promotes the proof-of-concept in phase3_field2field.py into the complete
RQ1/RQ2/RQ3 experiment (the same questions the param->field pipeline answers in
phase3_train.py + phase3_eval_rq.py), but in the framing where the operator is
actually justified:

  * param->field (old headline): each grid point is fed the full 9-param vector,
    so a per-point coordinate MLP solves locally and BEATS the FNO -> the operator
    looks pointless (artifact of the framing).
  * field->field (this file): the input is an indentation/penetration FIELD on the
    grid; a point's displacement is a NON-LOCAL function of the whole contact
    (elastic Green's function), so a per-point MLP cannot see where contact is and
    FNO's global spectral mixing wins. This is the operator-learning setup that
    belongs in the paper headline.

Input field (3 ch on HxW):  ch0 penetration, ch1 shear_x*mask, ch2 shear_y*mask.
Scalars (2):                 mu, E.
Output field (3 ch):         displacement (ux, uy, uz)  [Hertz-Mindlin GT].

Splits MIRROR data/phase3_gt exactly (same RANGES / OOD overrides / mode balance),
built via sample_params from phase3_data_gen so the two framings are comparable.

Models:
  * FNO     field->field  (operator, main)
  * MLP     per-point     (mandatory local baseline)
  * FNO + multi-task slip head (a)   — pooled latent -> 4-mode logits
  * Separate slip classifier   (b)   — CNN reading the FNO field output

Outputs: runs/phase3_f2f_full/results.json + fidelity_speed.png + printed tables.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from novbts.models import SpectralConv2d, count_parameters
from novbts.groundtruth.hertz_mindlin import hertz_mindlin_field, hertz_scalars, MODE_NAMES
from novbts.groundtruth.data_gen import sample_params, make_grid
from novbts.paths import RUNS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# real PhysX-FEM solver throughput measured in Phase 3 (data/phase3_gt_fem),
# used as the RQ3 reference instead of the (too-fast) analytic formula.
FEM_SOLVER_FPS = 7.2


# ---------------------------------------------------------------------------
# Field-input construction (params -> penetration/shear field) + GT output
# ---------------------------------------------------------------------------

def params_to_fieldinput(params, coords, side):
    """[N,9] params -> input field [N,3,H,W] + scalars [N,2] (mu,E).

    Penetration profile depends on geometry, so the flat-punch OOD split is a
    genuinely different input field shape the operator has not seen:
      sphere (geom=0): pen = max(0, depth - r^2/(2R)),  contact radius = Hertz a
      flat   (geom=1): pen = depth inside the punch,    contact radius = R
    """
    p = np.asarray(params, dtype=np.float64)
    x0, y0, depth, R, sx, sy, mu, E, geom = [p[:, i] for i in range(9)]
    X = coords[:, 0].reshape(side, side)
    Y = coords[:, 1].reshape(side, side)
    dx = X[None] - x0[:, None, None]
    dy = Y[None] - y0[:, None, None]
    r2 = dx ** 2 + dy ** 2
    r = np.sqrt(r2 + 1e-12)

    a, _, _, _ = hertz_scalars(depth, R, E)
    is_flat = geom[:, None, None] > 0.5
    a_eff = np.where(geom > 0.5, R, a)[:, None, None]

    pen_sphere = np.clip(depth[:, None, None] - r2 / (2.0 * R[:, None, None]), 0.0, None)
    pen_flat = depth[:, None, None] * (r <= R[:, None, None])
    pen = np.where(is_flat, pen_flat, pen_sphere)
    mask = (r <= a_eff).astype(np.float64)

    inp = np.zeros((p.shape[0], 3, side, side), np.float32)
    inp[:, 0] = pen
    inp[:, 1] = sx[:, None, None] * mask
    inp[:, 2] = sy[:, None, None] * mask
    scal = np.stack([mu, E], -1).astype(np.float32)
    return inp, scal


def gen_split(params, coords, side):
    """-> input [N,3,H,W], output disp [N,3,H,W], scalars [N,2], mode [N]."""
    inp, scal = params_to_fieldinput(params, coords, side)
    disp, mode = hertz_mindlin_field(params, coords)        # [N,M,3]
    out = disp.reshape(-1, side, side, 3).transpose(0, 3, 1, 2).astype(np.float32)
    return (torch.tensor(inp), torch.tensor(out),
            torch.tensor(scal), torch.tensor(mode.astype(np.int64)))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FNOField(nn.Module):
    """field -> field FNO with an optional multi-task 4-mode slip head (a)."""
    def __init__(self, in_ch=3, width=48, modes=12, out_ch=3, with_slip_head=False):
        super().__init__()
        self.with_slip_head = with_slip_head
        self.fc0 = nn.Conv2d(in_ch + 2, width, 1)            # +2 scalar channels
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes, modes) for _ in range(4)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Conv2d(width, 96, 1)
        self.fc2 = nn.Conv2d(96, out_ch, 1)
        if with_slip_head:
            self.slip_head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 4))

    def forward(self, field, scal):
        b, c, h, w = field.shape
        sc = scal[:, :, None, None].expand(b, scal.shape[1], h, w)
        x = self.fc0(torch.cat([field, sc], 1))
        for spec, pw in zip(self.spectral, self.pointwise):
            x = torch.nn.functional.gelu(spec(x) + pw(x))
        latent = x
        y = torch.nn.functional.gelu(self.fc1(latent))
        out = self.fc2(y)
        if self.with_slip_head:
            return out, self.slip_head(latent.mean(dim=(2, 3)))
        return out


class PerPointMLP(nn.Module):
    """Local baseline: [input 3ch @ point, x, y, mu, E] -> disp. No global context."""
    def __init__(self, in_ch=3, hidden=256, out_ch=3):
        super().__init__()
        d = in_ch + 2 + 2
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_ch),
        )

    def forward(self, field, scal, coord_grid):
        b, c, h, w = field.shape
        f = field.permute(0, 2, 3, 1)
        cg = coord_grid[None].expand(b, h, w, 2)
        sc = scal[:, None, None, :].expand(b, h, w, 2)
        x = torch.cat([f, cg, sc], -1)
        return self.net(x).permute(0, 3, 1, 2)


class SlipClassifierField(nn.Module):
    """Slip head (b): CNN classifier reading a (frozen) FNO's field output [N,3,H,W]."""
    def __init__(self, in_ch=3, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 4),
        )

    def forward(self, field):
        return self.net(field)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rel_l2_per_mode(pred, tgt, mode):
    d = (pred - tgt).reshape(pred.shape[0], -1)
    t = tgt.reshape(tgt.shape[0], -1)
    rel = torch.linalg.norm(d, dim=1) / (torch.linalg.norm(t, dim=1) + 1e-8)
    out = {"overall": float(rel.mean())}
    for i, n in enumerate(MODE_NAMES):
        m = mode == i
        out[n] = float(rel[m].mean()) if m.any() else float("nan")
    return out


def tangential_dir_error(pred, tgt, mode):
    """Mean angle (deg) between predicted & GT in-plane vectors on slip frames,
    weighted by GT tangential magnitude. pred/tgt: [N,3,H,W]."""
    slip = mode >= 2
    if not slip.any():
        return float("nan")
    p = pred[slip][:, :2].permute(0, 2, 3, 1).reshape(slip.sum(), -1, 2)
    t = tgt[slip][:, :2].permute(0, 2, 3, 1).reshape(slip.sum(), -1, 2)
    tmag = torch.linalg.norm(t, dim=-1)
    w = tmag / (tmag.sum(dim=-1, keepdim=True) + 1e-9)
    cos = torch.nn.functional.cosine_similarity(p, t, dim=-1).clamp(-1, 1)
    ang = torch.arccos(cos) * 180.0 / math.pi
    return float((ang * w).sum(dim=-1).mean())


def macro_f1(pred_labels, true_labels, n_cls=4):
    f1s = []
    for c in range(n_cls):
        tp = ((pred_labels == c) & (true_labels == c)).sum().item()
        fp = ((pred_labels == c) & (true_labels != c)).sum().item()
        fn = ((pred_labels != c) & (true_labels == c)).sum().item()
        f1s.append(2 * tp / (2 * tp + fp + fn + 1e-9))
    ps, ts = (pred_labels >= 2), (true_labels >= 2)
    tp = (ps & ts).sum().item(); fp = (ps & ~ts).sum().item(); fn = (~ps & ts).sum().item()
    slip_f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return float(np.mean(f1s)), f1s, slip_f1


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def field_loss(pred, tgt, lambda_dir=0.1):
    mse = (pred - tgt).square().mean()
    pt = pred[:, :2].permute(0, 2, 3, 1).reshape(pred.shape[0], -1, 2)
    tt = tgt[:, :2].permute(0, 2, 3, 1).reshape(tgt.shape[0], -1, 2)
    tmag = torch.linalg.norm(tt, dim=-1)
    mask = tmag > 1e-4
    if mask.any():
        cos = torch.nn.functional.cosine_similarity(pt[mask], tt[mask], dim=-1)
        dir_loss = (1.0 - cos).mean()
    else:
        dir_loss = torch.zeros((), device=pred.device)
    return mse + lambda_dir * dir_loss


def train_operator(model, inp, out, scal, mode, coord_grid, epochs, lr,
                   is_mlp=False, multitask=False, lambda_cls=0.1, bs=64):
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    n = inp.shape[0]
    if DEV.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(n, device=inp.device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            if is_mlp:
                pred = model(inp[idx], scal[idx], coord_grid)
                loss = field_loss(pred, out[idx])
            elif multitask:
                pred, logits = model(inp[idx], scal[idx])
                loss = field_loss(pred, out[idx]) + lambda_cls * ce(logits, mode[idx])
            else:
                pred = model(inp[idx], scal[idx])
                loss = field_loss(pred, out[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if DEV.type == "cuda" else None
    return secs, vram


def train_separate_clf(clf, fno, inp, scal, mode, ostd, om, epochs, lr, bs=64):
    """Train (b) on the frozen FNO's NORMALISED field output."""
    opt = torch.optim.AdamW(clf.parameters(), lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    fno.eval()
    n = inp.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=inp.device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            with torch.no_grad():
                out = fno(inp[idx], scal[idx])
                field = out[0] if isinstance(out, tuple) else out
            logits = clf(field)
            loss = ce(logits, mode[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


@torch.no_grad()
def predict_raw(model, inp, scal, coord_grid, ostd, om, is_mlp=False, multitask=False, bs=128):
    preds = []
    for s in range(0, inp.shape[0], bs):
        e = s + bs
        if is_mlp:
            p = model(inp[s:e], scal[s:e], coord_grid)
        else:
            o = model(inp[s:e], scal[s:e])
            p = o[0] if multitask else o
        preds.append(p * ostd + om)
    return torch.cat(preds)


@torch.no_grad()
def throughput(model, inp, scal, coord_grid, is_mlp=False, multitask=False, iters=5):
    _ = model(inp[:32], scal[:32], coord_grid) if is_mlp else model(inp[:32], scal[:32])
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter(); n = 0
    for _ in range(iters):
        for s in range(0, inp.shape[0], 256):
            e = s + 256
            if is_mlp:
                o = model(inp[s:e], scal[s:e], coord_grid)
            else:
                oo = model(inp[s:e], scal[s:e]); o = oo[0] if multitask else oo
            n += o.shape[0]
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


# ---------------------------------------------------------------------------

OOD_SPLITS = [
    ("small_radius",  {"radius": (0.03, 0.07)}),
    ("large_radius",  {"radius": (0.34, 0.50)}),
    ("deep_indent",   {"depth": (1.01, 1.60)}),
    ("soft_material", {"stiffness": (0.05, 0.45)}),
    ("low_friction",  {"mu": (0.05, 0.28)}),
    ("flat_geom",     {"geom": (0, 1)}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=32)
    ap.add_argument("--train-per-mode", type=int, default=4000)
    ap.add_argument("--test-per-mode", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--clf-epochs", type=int, default=20)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-cls", type=float, default=0.1)
    ap.add_argument("--out-dir", default=str(RUNS / "phase3_f2f_full"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"device={DEV}  field->field FULL  {args.side}x{args.side}")
    torch.manual_seed(7)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    coords = make_grid(args.side)
    coord_grid = torch.tensor(coords.reshape(args.side, args.side, 2)).to(DEV)

    # ---- splits mirroring data/phase3_gt ----
    print("generating splits (field->field)...")
    tr = gen_split(sample_params(rng, args.train_per_mode), coords, args.side)
    va = gen_split(sample_params(rng, args.test_per_mode), coords, args.side)
    ti = gen_split(sample_params(rng, args.test_per_mode), coords, args.side)
    ts = gen_split(sample_params(rng, args.test_per_mode, modes=(2, 3)), coords, args.side)
    tr = [t.to(DEV) for t in tr]; va = [t.to(DEV) for t in va]
    ti = [t.to(DEV) for t in ti]; ts = [t.to(DEV) for t in ts]

    # per-channel normalisation from train
    im = tr[0].mean((0, 2, 3), keepdim=True); istd = tr[0].std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    om = tr[1].mean((0, 2, 3), keepdim=True); ostd = tr[1].std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    sm = tr[2].mean(0, keepdim=True); sstd = tr[2].std(0, keepdim=True).clamp_min(1e-6)
    nin = lambda t: (t - im) / istd
    nout = lambda t: (t - om) / ostd
    nsc = lambda t: (t - sm) / sstd

    tr_in, tr_out, tr_sc, tr_md = nin(tr[0]), nout(tr[1]), nsc(tr[2]), tr[3]

    summary = {"framing": "field->field", "device": str(DEV), "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
               "train_frames": int(tr[0].shape[0]), "side": args.side,
               "lambda_cls": args.lambda_cls, "models": {},
               "RQ1": {}, "RQ2": {}, "RQ3": {}}

    # ===== train models =====
    torch.manual_seed(0)
    mlp = PerPointMLP().to(DEV)
    s_mlp, v_mlp = train_operator(mlp, tr_in, tr_out, tr_sc, tr_md, coord_grid,
                                  args.epochs, args.lr, is_mlp=True)
    print(f"[MLP] {s_mlp:.0f}s")

    torch.manual_seed(0)
    fno = FNOField(modes=args.modes).to(DEV)
    s_fno, v_fno = train_operator(fno, tr_in, tr_out, tr_sc, tr_md, coord_grid,
                                  args.epochs, args.lr)
    print(f"[FNO] {s_fno:.0f}s")

    torch.manual_seed(0)
    fno_mt = FNOField(modes=args.modes, with_slip_head=True).to(DEV)
    s_mt, v_mt = train_operator(fno_mt, tr_in, tr_out, tr_sc, tr_md, coord_grid,
                                args.epochs, args.lr, multitask=True, lambda_cls=args.lambda_cls)
    print(f"[FNO+slip a] {s_mt:.0f}s")

    torch.manual_seed(0)
    clf = SlipClassifierField().to(DEV)
    train_separate_clf(clf, fno, tr_in, tr_sc, tr_md, ostd, om, args.clf_epochs, args.lr)
    print(f"[slip clf b] done")

    for nm, mdl, sc, vr in [("mlp", mlp, s_mlp, v_mlp), ("fno", fno, s_fno, v_fno),
                            ("fno_multitask_a", fno_mt, s_mt, v_mt)]:
        summary["models"][nm] = {"params": count_parameters(mdl),
                                 "train_s": round(sc, 1), "peak_vram_gb": vr}
    summary["models"]["slip_classifier_b"] = {"params": count_parameters(clf)}

    # ===== RQ1 accuracy (test_id) =====
    def eval_acc(model, split, is_mlp=False, multitask=False):
        inp, gt, scal, md = split
        pred = predict_raw(model, nin(inp), nsc(scal), coord_grid, ostd, om,
                           is_mlp=is_mlp, multitask=multitask)
        return {"relative_l2": rel_l2_per_mode(pred, gt, md),
                "tangential_dir_error_deg": tangential_dir_error(pred, gt, md)}

    summary["RQ1"]["mlp"] = eval_acc(mlp, ti, is_mlp=True)
    summary["RQ1"]["fno"] = eval_acc(fno, ti)
    summary["RQ1"]["fno_mt_a"] = eval_acc(fno_mt, ti, multitask=True)
    summary["RQ1"]["fno_on_test_slip"] = eval_acc(fno, ts)

    # slip heads on test_id
    with torch.no_grad():
        _, logits_a = fno_mt(nin(ti[0]), nsc(ti[2]))
        field_b = fno(nin(ti[0]), nsc(ti[2]))
        logits_b = clf(field_b)
    for tag, logits in [("slip_head_a_multitask", logits_a), ("slip_head_b_classifier", logits_b)]:
        mf1, per, slipf1 = macro_f1(logits.argmax(-1), ti[3])
        summary["RQ1"][tag] = {"macro_f1": mf1, "per_class_f1": dict(zip(MODE_NAMES, per)),
                               "slip_f1": slipf1}

    # ===== RQ2 generalisation (FNO) =====
    id_l2 = summary["RQ1"]["fno"]["relative_l2"]["overall"]
    for name, ov in OOD_SPLITS:
        sp = gen_split(sample_params(rng, args.test_per_mode, overrides=ov), coords, args.side)
        sp = [t.to(DEV) for t in sp]
        l2 = eval_acc(fno, sp)["relative_l2"]["overall"]
        summary["RQ2"][name] = {"relative_l2_overall": l2, "degradation_x": l2 / id_l2}
    # resolution OOD (different grid -> FNO needs grid >= modes)
    base_p = sample_params(rng, args.test_per_mode)
    for res in (16, 64):
        cg = make_grid(res)
        sp = gen_split(base_p, cg, res)
        sp = [t.to(DEV) for t in sp]
        cgt = torch.tensor(cg.reshape(res, res, 2)).to(DEV)
        try:
            inp, gt, scal, md = sp
            pred = predict_raw(fno, (inp - im) / istd, nsc(scal), cgt, ostd, om)
            l2 = rel_l2_per_mode(pred, gt, md)["overall"]
            summary["RQ2"][f"res{res}"] = {"relative_l2_overall": l2, "degradation_x": l2 / id_l2}
        except Exception as e:
            summary["RQ2"][f"res{res}"] = {"error": f"{type(e).__name__}: not evaluable "
                                           f"(grid below FNO mode resolution)"}

    # ===== RQ3 speed =====
    speeds = {
        "mlp": throughput(mlp, nin(ti[0]), nsc(ti[2]), coord_grid, is_mlp=True),
        "fno": throughput(fno, nin(ti[0]), nsc(ti[2]), coord_grid),
        "fno_mt_a": throughput(fno_mt, nin(ti[0]), nsc(ti[2]), coord_grid, multitask=True),
    }
    # analytic GT throughput (reference; too fast to be the real baseline)
    pn, cn = sample_params(rng, args.test_per_mode), coords
    t0 = time.perf_counter()
    for _ in range(3):
        hertz_mindlin_field(pn, cn)
    speeds["gt_solver_analytic"] = (3 * pn.shape[0]) / (time.perf_counter() - t0)
    speeds["physx_fem_solver"] = FEM_SOLVER_FPS
    summary["RQ3"]["throughput_fps"] = speeds
    summary["RQ3"]["fno_speedup_vs_fem"] = speeds["fno"] / FEM_SOLVER_FPS
    summary["RQ3"]["note"] = (f"physx_fem_solver={FEM_SOLVER_FPS} fps is the real measured "
                              "PhysX-FEM throughput (data/phase3_gt_fem); analytic GT is far "
                              "faster but not a fair solver baseline.")

    # fidelity-speed plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        pts = {
            "MLP (per-point)": (speeds["mlp"], summary["RQ1"]["mlp"]["relative_l2"]["overall"]),
            "FNO (operator)": (speeds["fno"], summary["RQ1"]["fno"]["relative_l2"]["overall"]),
            "FNO+slip(a)": (speeds["fno_mt_a"], summary["RQ1"]["fno_mt_a"]["relative_l2"]["overall"]),
        }
        for lbl, (x, y) in pts.items():
            ax.scatter(x, y, s=90); ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 6))
        ax.axvline(FEM_SOLVER_FPS, ls="--", color="crimson")
        ax.annotate("PhysX-FEM solver (7.2 fps)", (FEM_SOLVER_FPS, ax.get_ylim()[1]),
                    color="crimson", rotation=90, va="top", fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("throughput (frames/s, log)")
        ax.set_ylabel("relative L2 (lower = more accurate)")
        ax.set_title("Phase 3 fidelity-speed (field->field framing)")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "fidelity_speed.png", dpi=130)
        summary["RQ3"]["plot"] = str(out / "fidelity_speed.png")
    except Exception as e:
        summary["RQ3"]["plot_error"] = str(e)

    (out / "results.json").write_text(json.dumps(summary, indent=2))

    # ===== print =====
    print("\n================ RQ1 accuracy (test_id, field->field) ================")
    for m in ["mlp", "fno", "fno_mt_a"]:
        r = summary["RQ1"][m]["relative_l2"]
        print(f"  {m:9s} relL2={r['overall']:.4f} | "
              + " ".join(f"{n}={r[n]:.3f}" for n in MODE_NAMES)
              + f" | dirErr={summary['RQ1'][m]['tangential_dir_error_deg']:.1f}deg")
    print(f"  fno_on_test_slip relL2={summary['RQ1']['fno_on_test_slip']['relative_l2']['overall']:.4f}")
    print("  --- slip detection (macro-F1 / slip-binary-F1) ---")
    for t in ["slip_head_a_multitask", "slip_head_b_classifier"]:
        print(f"  {t:24s} macroF1={summary['RQ1'][t]['macro_f1']:.4f} "
              f"slipF1={summary['RQ1'][t]['slip_f1']:.4f}")
    print("\n================ RQ2 generalisation (FNO) ================")
    print(f"  in-distribution relL2 = {id_l2:.4f}")
    for k, v in summary["RQ2"].items():
        if "error" in v:
            print(f"  {k:16s} {v['error']}")
        else:
            print(f"  {k:16s} relL2={v['relative_l2_overall']:.4f}  ({v['degradation_x']:.2f}x)")
    print("\n================ RQ3 speed ================")
    for k, v in speeds.items():
        print(f"  {k:20s} {v:10.1f} frames/s")
    print(f"  FNO speedup vs PhysX-FEM solver: {summary['RQ3']['fno_speedup_vs_fem']:.1f}x")
    fno_l2 = summary["RQ1"]["fno"]["relative_l2"]["overall"]
    mlp_l2 = summary["RQ1"]["mlp"]["relative_l2"]["overall"]
    print(f"\nVERDICT: FNO {'BEATS' if fno_l2 < mlp_l2 else 'LOSES TO'} MLP "
          f"(FNO {fno_l2:.4f} vs MLP {mlp_l2:.4f}, {mlp_l2/fno_l2:.1f}x)")
    print(f"Saved {out/'results.json'} + fidelity_speed.png")


if __name__ == "__main__":
    main()
