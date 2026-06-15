#!/usr/bin/env python3
"""Transfer-learning benchmark: pretrain FNO on analytic GT, finetune on FEM.

Compares two conditions evaluated on the same FEM test set:
  baseline  -- FNO trained on FEM only  (same hyperparams as fem_benchmark.py)
  transfer  -- FNO pretrained on analytic GT, then finetuned on FEM (lower lr)

Both receive identical FEM epochs so the comparison is fair on FEM compute.
The pretrain phase is extra cost but uses the free analytic dataset.

Usage:
  python -m novbts.operator.transfer_benchmark
  python -m novbts.operator.transfer_benchmark --pretrain-epochs 60 --finetune-lr 3e-4
"""
import argparse
import json
import numpy as np
import torch

from novbts.operator.field2field import (
    FNOField,
    params_to_fieldinput, train_operator, predict_raw,
    rel_l2_per_mode, tangential_dir_error, macro_f1, count_parameters, DEV,
)
from novbts.groundtruth.hertz_mindlin import MODE_NAMES
from novbts.paths import ANALYTIC, FEM, RUNS, ensure


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_npz(path, n_limit=None):
    """Load a params/coords/disp/mode npz into normalised tensors.

    Returns: inp [N,3,H,W], out [N,3,H,W], scal [N,2], mode [N], side
    """
    d = np.load(path, allow_pickle=True)
    params = d["params"]
    coords = d["coords"]
    if n_limit is not None:
        params = params[:n_limit]
    side = int(round(np.sqrt(coords.shape[0])))
    inp, scal = params_to_fieldinput(params, coords, side)
    disp = d["disp"][:len(params)]
    out = disp.reshape(-1, side, side, 3).transpose(0, 3, 1, 2).astype(np.float32)
    mode = d["mode"][:len(params)].astype(np.int64)
    return (torch.tensor(inp), torch.tensor(out),
            torch.tensor(scal), torch.tensor(mode), side)


def norm_stats(inp, out, scal):
    im = inp.mean((0, 2, 3), keepdim=True)
    istd = inp.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    om = out.mean((0, 2, 3), keepdim=True)
    ostd = out.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    sm = scal.mean(0, keepdim=True)
    sstd = scal.std(0, keepdim=True).clamp_min(1e-6)
    return im, istd, om, ostd, sm, sstd


def normalise(inp, out, scal, stats):
    im, istd, om, ostd, sm, sstd = stats
    return (inp - im) / istd, (out - om) / ostd, (scal - sm) / sstd


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, inp, out, scal, mode, fem_stats, cg):
    im, istd, om, ostd, sm, sstd = fem_stats
    pred = predict_raw(model, (inp - im) / istd, (scal - sm) / sstd, cg, ostd, om)
    return {
        "relative_l2": rel_l2_per_mode(pred, out, mode),
        "tangential_dir_error_deg": tangential_dir_error(pred, out, mode),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fem-data", default=str(FEM / "shear_fine_swept.npz"))
    ap.add_argument("--analytic-data", default=str(ANALYTIC / "train.npz"))
    ap.add_argument("--n-analytic", type=int, default=None,
                    help="cap analytic frames (default: all 16k)")
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    ap.add_argument("--finetune-epochs", type=int, default=80,
                    help="FEM epochs for transfer (matches baseline)")
    ap.add_argument("--baseline-epochs", type=int, default=80)
    ap.add_argument("--pretrain-lr", type=float, default=1e-3)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(RUNS / "phase3_transfer"))
    args = ap.parse_args()

    # ---- load datasets ----
    print("Loading FEM data...")
    fem_inp, fem_out, fem_scal, fem_mode, side = load_npz(args.fem_data)
    N, nt = fem_inp.shape[0], args.n_test
    print(f"device={DEV}  side={side}  FEM N={N}  test={nt}")

    print("Loading analytic data...")
    an_inp, an_out, an_scal, an_mode, _ = load_npz(args.analytic_data, args.n_analytic)
    print(f"Analytic frames: {an_inp.shape[0]}")

    # ---- splits ----
    tr_idx = torch.arange(0, N - nt)
    te_idx = torch.arange(N - nt, N)

    def to_dev(tensors): return tuple(t.to(DEV) for t in tensors)

    fem_tr = to_dev((fem_inp[tr_idx], fem_out[tr_idx], fem_scal[tr_idx], fem_mode[tr_idx]))
    fem_te = to_dev((fem_inp[te_idx], fem_out[te_idx], fem_scal[te_idx], fem_mode[te_idx]))
    an_all = to_dev((an_inp, an_out, an_scal, an_mode))

    # ---- normalization stats (computed from respective train sets) ----
    fem_stats = norm_stats(*fem_tr[:3])
    an_stats  = norm_stats(*an_all[:3])

    fem_tr_n = normalise(*fem_tr[:3], fem_stats) + (fem_tr[3],)
    an_n     = normalise(*an_all[:3], an_stats)  + (an_all[3],)

    # coord grid (only used by MLP, not FNO — passed for API compatibility)
    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1
    ).astype(np.float32)).to(DEV)

    # ---- baseline: FEM-only ----
    print(f"\n[baseline] FEM-only FNO  epochs={args.baseline_epochs}  lr={args.lr}")
    torch.manual_seed(args.seed)
    fno_base = FNOField(modes=args.modes).to(DEV)
    s_base, _ = train_operator(
        fno_base, *fem_tr_n[:3], fem_tr_n[3], cg,
        args.baseline_epochs, args.lr,
    )
    print(f"[baseline] done  {s_base:.0f}s")

    # ---- transfer: pretrain analytic -> finetune FEM ----
    print(f"\n[transfer] pretrain analytic  epochs={args.pretrain_epochs}  lr={args.pretrain_lr}")
    torch.manual_seed(args.seed)
    fno_tl = FNOField(modes=args.modes).to(DEV)
    s_pre, _ = train_operator(
        fno_tl, *an_n[:3], an_n[3], cg,
        args.pretrain_epochs, args.pretrain_lr,
    )
    print(f"[transfer] pretrain done  {s_pre:.0f}s")
    print(f"[transfer] finetune FEM  epochs={args.finetune_epochs}  lr={args.finetune_lr}")
    s_ft, _ = train_operator(
        fno_tl, *fem_tr_n[:3], fem_tr_n[3], cg,
        args.finetune_epochs, args.finetune_lr,
    )
    print(f"[transfer] finetune done  {s_ft:.0f}s  (total {s_pre + s_ft:.0f}s)")

    # ---- evaluate both on FEM test set ----
    res_base = evaluate(fno_base, *fem_te[:3], fem_te[3], fem_stats, cg)
    res_tl   = evaluate(fno_tl,   *fem_te[:3], fem_te[3], fem_stats, cg)

    # ---- report ----
    print("\n=== Transfer Learning vs FEM-only (FEM test set) ===")
    print(f"{'model':12s} {'relL2':>8s} {'tang_dir°':>10s}")
    for tag, r in [("baseline", res_base), ("transfer", res_tl)]:
        print(f"{tag:12s} {r['relative_l2']['overall']:8.3f}"
              f"  {r['tangential_dir_error_deg']:8.1f}°")
    dl2  = res_base["relative_l2"]["overall"] - res_tl["relative_l2"]["overall"]
    ddir = res_base["tangential_dir_error_deg"] - res_tl["tangential_dir_error_deg"]
    print(f"\nΔ relL2    {dl2:+.3f}  ({'transfer better ✓' if dl2 > 0 else 'no gain'})")
    print(f"Δ tang_dir {ddir:+.1f}°  ({'transfer better ✓' if ddir > 0 else 'no gain'})")
    print(f"\nper-mode relL2:")
    print(f"{'mode':14s} {'baseline':>10s} {'transfer':>10s}")
    for m in MODE_NAMES:
        b = res_base["relative_l2"].get(m, float("nan"))
        t = res_tl["relative_l2"].get(m, float("nan"))
        sym = "✓" if t < b else " "
        print(f"{m:14s} {b:10.3f} {t:10.3f}  {sym}")

    # ---- save ----
    summary = {
        "gt": "physx_fem_shear_swept",
        "device": str(DEV), "side": side,
        "n_train_fem": int(N - nt), "n_test": int(nt),
        "n_analytic_pretrain": int(an_inp.shape[0]),
        "baseline": {
            "epochs_fem": args.baseline_epochs,
            "lr": args.lr,
            "train_s": round(s_base, 1),
            **res_base,
        },
        "transfer": {
            "pretrain_epochs_analytic": args.pretrain_epochs,
            "pretrain_lr": args.pretrain_lr,
            "finetune_epochs_fem": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "pretrain_s": round(s_pre, 1),
            "finetune_s": round(s_ft, 1),
            **res_tl,
        },
        "delta": {
            "rel_l2": round(dl2, 4),
            "tangential_dir_deg": round(ddir, 2),
        },
    }
    ensure(args.out_dir)
    out_path = f"{args.out_dir}/results.json"
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
