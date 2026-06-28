#!/usr/bin/env python3
"""Full RQ1-RQ3 + FNO-vs-MLP + slip-F1 benchmark on the PhysX-FEM ground truth.

Mirrors the analytic field->field benchmark (novbts.operator.field2field.main)
but trains/evaluates on the swept fine-mesh FEM data
(data/fem/shear_fine_swept.npz) -- closing the Gate-3 requirement to run the
benchmark on REAL-physics GT rather than the analytic Hertz-Mindlin proxy.

  RQ1  per-mode accuracy; FNO vs per-point MLP (the non-locality claim on real physics)
  RQ2  generalisation to held-out parameter tails (extrapolate high R / mu / E)
  RQ3  FNO/MLP inference throughput vs the real PhysX-FEM shear solver
  slip per-mode macro-F1 from multitask head (a) and separate classifier (b)

Honest scope note: FEM data is in-box (R 15-25mm, mu 0.4-0.8, E 0.5-2e5).
RQ2 here is *extrapolation to the high tail of each parameter* (train on the
lower 80%, test on the upper 20%), not the wider out-of-range OOD the analytic
split could afford -- we cannot cheaply synthesise FEM outside the box.
"""
import argparse
import glob
import json
import os
import numpy as np
import torch

from novbts.operator.field2field import (
    FNOField, PerPointMLP, SlipClassifierField,
    params_to_fieldinput, train_operator, train_separate_clf, predict_raw,
    throughput, rel_l2_per_mode, tangential_dir_error, macro_f1, count_parameters, DEV,
)
from novbts.groundtruth.hertz_mindlin import MODE_NAMES
from novbts.paths import FEM, RUNS, ensure


def load(npz):
    d = np.load(npz, allow_pickle=True)
    params, coords, disp, mode = d["params"], d["coords"], d["disp"], d["mode"]
    side = int(round(np.sqrt(coords.shape[0])))
    inp, scal = params_to_fieldinput(params, coords, side)
    out = disp.reshape(-1, side, side, 3).transpose(0, 3, 1, 2).astype(np.float32)
    return dict(params=params, inp=torch.tensor(inp), out=torch.tensor(out),
                scal=torch.tensor(scal), mode=torch.tensor(mode.astype(np.int64)), side=side)


def fem_shear_solver_fps():
    """Mean PhysX-FEM shear solve time across the swept combos -> fps."""
    ts = []
    for f in glob.glob(str(FEM / "sweep" / "combo_*" / "fem_gt_shear.npz")):
        z = np.load(f, allow_pickle=True)
        if "solve_time_s" in z.files:
            ts.append(np.asarray(z["solve_time_s"]))
    return (1.0 / float(np.concatenate(ts).mean())) if ts else 1.0 / 3.0


def norm_from(inp, out, scal):
    im = inp.mean((0, 2, 3), keepdim=True); istd = inp.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    om = out.mean((0, 2, 3), keepdim=True); ostd = out.std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    sm = scal.mean(0, keepdim=True); sstd = scal.std(0, keepdim=True).clamp_min(1e-6)
    return im, istd, om, ostd, sm, sstd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FEM / "shear_fine_swept.npz"))
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--clf-epochs", type=int, default=40)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-cls", type=float, default=0.1)
    args = ap.parse_args()

    D = load(args.data)
    side, N, nt = D["side"], D["inp"].shape[0], args.n_test
    print(f"device={DEV}  FEM benchmark  N={N} side={side}  test_id=last {nt}")
    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1
    ).astype(np.float32)).to(DEV)
    inp, out, scal, mode = (D[k].to(DEV) for k in ("inp", "out", "scal", "mode"))

    def slice_(idx):
        return inp[idx], out[idx], scal[idx], mode[idx]

    # ---- main split: train = first N-nt, test_id = last nt (already shuffled in file) ----
    tr_idx = torch.arange(0, N - nt, device=DEV)
    te_idx = torch.arange(N - nt, N, device=DEV)
    tri, tro, trs, trm = slice_(tr_idx)
    im, istd, om, ostd, sm, sstd = norm_from(tri, tro, trs)
    nin = lambda t: (t - im) / istd
    nout = lambda t: (t - om) / ostd
    nsc = lambda t: (t - sm) / sstd

    def train(kind):
        torch.manual_seed(0)
        if kind == "mlp":
            m = PerPointMLP().to(DEV)
            s, v = train_operator(m, nin(tri), nout(tro), nsc(trs), trm, cg, args.epochs, args.lr, is_mlp=True)
        elif kind == "fno":
            m = FNOField(modes=args.modes).to(DEV)
            s, v = train_operator(m, nin(tri), nout(tro), nsc(trs), trm, cg, args.epochs, args.lr)
        elif kind == "fno_mt":
            m = FNOField(modes=args.modes, with_slip_head=True).to(DEV)
            s, v = train_operator(m, nin(tri), nout(tro), nsc(trs), trm, cg, args.epochs, args.lr,
                                  multitask=True, lambda_cls=args.lambda_cls)
        return m, s

    mlp, s_mlp = train("mlp");        print(f"[MLP] {s_mlp:.0f}s")
    fno, s_fno = train("fno");        print(f"[FNO] {s_fno:.0f}s")
    fno_mt, s_mt = train("fno_mt");   print(f"[FNO+slip a] {s_mt:.0f}s")
    torch.manual_seed(0)
    clf = SlipClassifierField().to(DEV)
    train_separate_clf(clf, fno, nin(tri), nsc(trs), trm, ostd, om, args.clf_epochs, args.lr)
    print("[slip clf b] done")

    summary = {"gt": os.path.basename(args.data), "device": str(DEV),
               "train_frames": int(N - nt), "test_frames": nt, "side": side,
               "param_box": {"R_mm": [15, 25], "mu": [0.4, 0.8], "E_Pa": [5e4, 2e5]},
               "models": {}, "RQ1": {}, "RQ2": {}, "RQ3": {}}

    def acc(model, idx, is_mlp=False, mt=False):
        i, g, s, md = slice_(idx)
        pred = predict_raw(model, nin(i), nsc(s), cg, ostd, om, is_mlp=is_mlp, multitask=mt)
        return {"relative_l2": rel_l2_per_mode(pred, g, md),
                "tangential_dir_error_deg": tangential_dir_error(pred, g, md)}

    # ===== RQ1 =====
    summary["RQ1"]["mlp"] = acc(mlp, te_idx, is_mlp=True)
    summary["RQ1"]["fno"] = acc(fno, te_idx)
    summary["RQ1"]["fno_mt_a"] = acc(fno_mt, te_idx, mt=True)
    with torch.no_grad():
        _, logits_a = fno_mt(nin(inp[te_idx]), nsc(scal[te_idx]))
        logits_b = clf(fno(nin(inp[te_idx]), nsc(scal[te_idx])))
    for tag, lg in [("slip_head_a_multitask", logits_a), ("slip_head_b_classifier", logits_b)]:
        mf1, per, slipf1 = macro_f1(lg.argmax(-1), mode[te_idx])
        summary["RQ1"][tag] = {"macro_f1": mf1, "per_class_f1": dict(zip(MODE_NAMES, per)), "slip_f1": slipf1}
    for nm, mdl, sc in [("mlp", mlp, s_mlp), ("fno", fno, s_fno), ("fno_multitask_a", fno_mt, s_mt)]:
        summary["models"][nm] = {"params": count_parameters(mdl), "train_s": round(sc, 1)}
    summary["models"]["slip_classifier_b"] = {"params": count_parameters(clf)}

    # ===== RQ2: extrapolate to the high tail of each parameter (train low-80%, test high-20%) =====
    P = D["params"]
    for pname, col in [("radius", 3), ("mu", 6), ("E", 7)]:
        vals = P[:, col]
        thr = np.quantile(vals, 0.8)
        lo = np.where(vals <= thr)[0]; hi = np.where(vals > thr)[0]
        rng = np.random.default_rng(0); rng.shuffle(lo)
        nval = max(40, len(lo) // 10)
        tr2, val2 = lo[nval:], lo[:nval]                  # train on low-80% minus a val slice
        tr2 = torch.tensor(tr2, device=DEV)
        i2, o2, s2, m2 = slice_(tr2)
        im2, istd2, om2, ostd2, sm2, sstd2 = norm_from(i2, o2, s2)
        torch.manual_seed(0)
        f2 = FNOField(modes=args.modes).to(DEV)
        train_operator(f2, (i2 - im2) / istd2, (o2 - om2) / ostd2, (s2 - sm2) / sstd2, m2, cg, args.epochs, args.lr)

        def l2_on(idx_np):
            idx = torch.tensor(idx_np, device=DEV)
            i, g, s, md = slice_(idx)
            pred = predict_raw(f2, (i - im2) / istd2, (s - sm2) / sstd2, cg, ostd2, om2)
            return rel_l2_per_mode(pred, g, md)["overall"]

        l2_id, l2_ood = l2_on(val2), l2_on(hi)
        summary["RQ2"][f"high_{pname}"] = {
            "threshold": float(thr), "n_ood": int(len(hi)),
            "l2_in_dist": l2_id, "l2_extrapolated": l2_ood,
            "degradation_x": round(l2_ood / max(l2_id, 1e-9), 2)}
        print(f"[RQ2 high_{pname}] in-dist {l2_id:.3f} -> extrapolated {l2_ood:.3f} "
              f"({l2_ood/max(l2_id,1e-9):.2f}x)")

    # ===== RQ3: throughput vs the real PhysX-FEM shear solver =====
    fem_fps = fem_shear_solver_fps()
    speeds = {
        "mlp": throughput(mlp, nin(inp[te_idx]), nsc(scal[te_idx]), cg, is_mlp=True),
        "fno": throughput(fno, nin(inp[te_idx]), nsc(scal[te_idx]), cg),
        "fno_mt_a": throughput(fno_mt, nin(inp[te_idx]), nsc(scal[te_idx]), cg, multitask=True),
        "physx_fem_shear_solver": fem_fps,
    }
    summary["RQ3"]["throughput_fps"] = speeds
    summary["RQ3"]["fno_speedup_vs_fem"] = round(speeds["fno"] / fem_fps, 1)
    summary["RQ3"]["note"] = (f"physx_fem_shear_solver={fem_fps:.3f} fps measured from the swept "
                              "combos' solve_time_s (res-24, 50x50x20mm gel).")

    # ===== print =====
    r1 = summary["RQ1"]
    print(f"\n=== RQ1 accuracy (test_id, FEM GT) ===")
    print(f"{'model':18s} {'overall':>8s} {'tang_dir°':>10s}")
    for nm in ["mlp", "fno", "fno_mt_a"]:
        print(f"{nm:18s} {r1[nm]['relative_l2']['overall']:8.3f} {r1[nm]['tangential_dir_error_deg']:10.1f}")
    print(f"  FNO beats MLP overall: {r1['mlp']['relative_l2']['overall']/r1['fno']['relative_l2']['overall']:.2f}x")
    print(f"per-mode rel L2 (FNO): " + "  ".join(
        f"{m}={r1['fno']['relative_l2'].get(m, float('nan')):.3f}" for m in MODE_NAMES))
    print(f"slip macro-F1: head_a={r1['slip_head_a_multitask']['macro_f1']:.3f}  "
          f"head_b={r1['slip_head_b_classifier']['macro_f1']:.3f}")
    print(f"\n=== RQ3 speed ===  FNO {speeds['fno']:.0f} fps vs FEM {fem_fps:.3f} fps "
          f"=> {summary['RQ3']['fno_speedup_vs_fem']:.0f}x")

    ensure(RUNS / "phase3_fem")
    json.dump(summary, open(RUNS / "phase3_fem" / "benchmark.json", "w"), indent=2)
    print(f"\nSaved {RUNS / 'phase3_fem' / 'benchmark.json'}")


if __name__ == "__main__":
    main()
