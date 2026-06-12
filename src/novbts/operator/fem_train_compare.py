#!/usr/bin/env python3
"""
Train the field->field operator on COARSE-mesh vs FINE(res24)-mesh FEM shear GT,
then evaluate BOTH on the SAME held-out FINE test set.

Question: does an under-resolved (coarse) FEM ground truth teach the operator a
worse model than the converged (fine) GT?  The convergence study showed the
coarse default mesh under-measures tangential displacement ~37%; if that bias
propagates, the coarse-trained operator should systematically under-predict
tangential motion on the fine test set.

Datasets are PAIRED (same geometry/seed, only mesh differs), so the comparison
isolates mesh resolution.  Reuses the field->field machinery from
phase3_field2field_full.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from novbts.operator.field2field import (
    FNOField, params_to_fieldinput, train_operator, predict_raw,
    rel_l2_per_mode, tangential_dir_error, DEV,
)
from novbts.groundtruth.hertz_mindlin import MODE_NAMES
from novbts.paths import FEM, RUNS, ensure


def load_fem_f2f(npz_path):
    """FEM npz -> field->field tensors (input [N,3,H,W], out [N,3,H,W], scal, mode)."""
    d = np.load(npz_path, allow_pickle=True)
    params, coords, disp, mode = d["params"], d["coords"], d["disp"], d["mode"]
    side = int(round(np.sqrt(coords.shape[0])))
    inp, scal = params_to_fieldinput(params, coords, side)
    out = disp.reshape(-1, side, side, 3).transpose(0, 3, 1, 2).astype(np.float32)
    return (torch.tensor(inp), torch.tensor(out), torch.tensor(scal),
            torch.tensor(mode.astype(np.int64)), side)


def tangential_rell2(pred, tgt):
    """rel L2 on the in-plane (tangential) channels only [N,0:2,H,W]."""
    p, t = pred[:, :2], tgt[:, :2]
    d = (p - t).reshape(p.shape[0], -1)
    tt = t.reshape(t.shape[0], -1)
    return float((torch.linalg.norm(d, 1) / (torch.linalg.norm(tt, 1) + 1e-8)).mean())


def mean_tang_mag(field):
    return float(torch.linalg.norm(field[:, :2], dim=1).mean())


def train_one(label, tr, coord_grid, epochs, lr, modes):
    """tr = (inp,out,scal,mode) RAW tensors on device. Returns (model, norm)."""
    inp, out, scal, mode = tr
    im = inp.mean((0, 2, 3), keepdim=True); istd = inp.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    om = out.mean((0, 2, 3), keepdim=True); ostd = out.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    sm = scal.mean(0, keepdim=True); sstd = scal.std(0, keepdim=True).clamp_min(1e-6)
    torch.manual_seed(0)
    model = FNOField(modes=modes).to(DEV)
    secs, _ = train_operator(model, (inp - im) / istd, (out - om) / ostd,
                             (scal - sm) / sstd, mode, coord_grid, epochs, lr)
    print(f"  [{label}] trained {secs:.0f}s")
    return model, (im, istd, om, ostd, sm, sstd)


def eval_on(model, norm, test, coord_grid):
    """Evaluate model (trained with its own norm) on RAW test tensors."""
    im, istd, om, ostd, sm, sstd = norm
    inp, out, scal, mode = test
    pred = predict_raw(model, (inp - im) / istd, (scal - sm) / sstd, coord_grid, ostd, om)
    return {
        "rel_l2_overall": rel_l2_per_mode(pred, out, mode)["overall"],
        "rel_l2_tangential": tangential_rell2(pred, out),
        "dir_err_deg": tangential_dir_error(pred, out, mode),
        "pred_mean_tang_mm": mean_tang_mag(pred) * 1000,
        "gt_mean_tang_mm": mean_tang_mag(out) * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", default=str(FEM / "shear_fine.npz"))
    ap.add_argument("--coarse", default=str(FEM / "shear_coarse.npz"))
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--modes", type=int, default=12)
    args = ap.parse_args()

    fine = load_fem_f2f(args.fine)
    coarse = load_fem_f2f(args.coarse)
    side = fine[4]
    assert coarse[4] == side, "coarse/fine marker grids differ"
    N = fine[0].shape[0]
    nt = args.n_test
    print(f"frames={N} side={side}  test=last {nt} (FINE)  train=first {N-nt}")

    coord_grid = torch.tensor(
        np.stack(np.meshgrid(np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1)
        .astype(np.float32)).to(DEV)

    def split(ds, lo, hi):
        return tuple(t[lo:hi].to(DEV) for t in ds[:4])

    fine_tr = split(fine, 0, N - nt);  fine_te = split(fine, N - nt, N)
    coarse_tr = split(coarse, 0, N - nt)

    print("\nTraining on COARSE-mesh GT ...")
    m_coarse, n_coarse = train_one("coarse", coarse_tr, coord_grid, args.epochs, args.lr, args.modes)
    print("Training on FINE(res24)-mesh GT ...")
    m_fine, n_fine = train_one("fine", fine_tr, coord_grid, args.epochs, args.lr, args.modes)

    print("\n=== Evaluate BOTH on the SAME held-out FINE test set ===")
    r_coarse = eval_on(m_coarse, n_coarse, fine_te, coord_grid)
    r_fine = eval_on(m_fine, n_fine, fine_te, coord_grid)
    gt_tang = r_fine["gt_mean_tang_mm"]

    print(f"\n{'metric':28s} {'COARSE-trained':>16s} {'FINE-trained':>14s}")
    for k in ["rel_l2_overall", "rel_l2_tangential", "dir_err_deg", "pred_mean_tang_mm"]:
        print(f"  {k:26s} {r_coarse[k]:16.4f} {r_fine[k]:14.4f}")
    print(f"  {'gt_mean_tang_mm (ref)':26s} {gt_tang:16.4f} {gt_tang:14.4f}")

    print("\nVERDICT:")
    print(f"  FINE-trained rel L2 overall = {r_fine['rel_l2_overall']:.4f} vs "
          f"COARSE-trained = {r_coarse['rel_l2_overall']:.4f}")
    c_bias = (r_coarse['pred_mean_tang_mm'] - gt_tang) / gt_tang * 100
    f_bias = (r_fine['pred_mean_tang_mm'] - gt_tang) / gt_tang * 100
    print(f"  tangential-magnitude bias vs FINE GT: coarse-trained {c_bias:+.0f}%, "
          f"fine-trained {f_bias:+.0f}%")
    import json
    out_dir = RUNS / "phase3_fem"
    ensure(out_dir)
    json.dump({"coarse_trained": r_coarse, "fine_trained": r_fine},
              open(out_dir / "compare.json", "w"), indent=2)
    print(f"\nSaved {out_dir / 'compare.json'}")


if __name__ == "__main__":
    main()
