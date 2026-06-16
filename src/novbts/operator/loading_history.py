#!/usr/bin/env python3
"""Phase 6c -- does LOADING HISTORY carry information beyond the endpoint params?

The tangential-ceiling note (input-representation-tangential) hypothesised that the
irreducible component is path-dependent. That is only testable if the loading PATH
varies for a fixed endpoint -- which the generator now produces with --mix-loads
(linear / ortho / reverse paths to the SAME (sx,sy)).

Test: predict the FINAL marker displacement with an FNO whose input is
  baseline : endpoint only (pen, sx*mask, sy*mask) + (mu,E)          [in_ch=3]
  +loadmode: baseline + the load-path label as 3 one-hot channels    [in_ch=6]
If +loadmode reduces error (esp. on ortho/reverse frames), the final state is
genuinely path-dependent -> loading history is a real lever. If not, path-dependence
is weak in this shallow-contact FEM (an honest null) and the ceiling is elsewhere.

  python -m novbts.operator.loading_history --data data/fem/traj_mix/fem_gt_shear.npz
"""
import argparse
import json

import numpy as np
import torch

from novbts.operator.field2field import (
    FNOField, train_operator, predict_raw, rel_l2_per_mode, tangential_dir_error,
    count_parameters, DEV,
)
from novbts.operator.fem_benchmark import load, norm_from
from novbts.paths import RUNS, ensure

LOAD_MODES = ["linear", "ortho", "reverse"]


def path_dependence_modelfree(disp, params, lm, k=14):
    """FNO-independent, DISTANCE-MATCHED test of path dependence. For each frame take
    its k nearest endpoints (depth,sx,sy) REGARDLESS of load mode, then split that one
    neighbourhood into same-mode vs cross-mode and compare the final-disp difference.
    Both subsets are drawn from the same k-nearest set, so endpoint distance is
    controlled (avoids the larger-cross-pool bias). cross/same > 1 => path-dependent."""
    N = disp.shape[0]
    ep = np.asarray(params[:, [2, 4, 5]], dtype=np.float64)         # depth, sx, sy
    ep = (ep - ep.mean(0)) / (ep.std(0) + 1e-9)
    Dm = np.sqrt(((ep[:, None] - ep[None]) ** 2).sum(-1))           # [N,N] endpoint distance
    np.fill_diagonal(Dm, np.inf)
    flat = disp.reshape(N, -1)
    nrm = np.linalg.norm(flat, axis=1) + 1e-9
    lm = np.asarray(lm)
    rs, rc, ds, dc = [], [], [], []
    for i in range(N):
        nn = np.argsort(Dm[i])[:k]                                  # k nearest endpoints, any mode
        for j in nn:
            r = np.linalg.norm(flat[i] - flat[j]) / nrm[i]
            if lm[j] == lm[i]:
                rs.append(r); ds.append(Dm[i, j])
            else:
                rc.append(r); dc.append(Dm[i, j])
    rs_m, rc_m = float(np.mean(rs)), float(np.mean(rc))
    return {"same_mode_resid": rs_m, "cross_mode_resid": rc_m,
            "cross_over_same": rc_m / (rs_m + 1e-9),
            "same_mode_ep_dist": float(np.mean(ds)), "cross_mode_ep_dist": float(np.mean(dc))}


def per_subset_rel_l2(pred, tgt, sel):
    """aggregate rel-L2 over a boolean subset of frames."""
    if not sel.any():
        return float("nan")
    d = (pred[sel] - tgt[sel]).reshape(int(sel.sum()), -1)
    t = tgt[sel].reshape(int(sel.sum()), -1)
    return float(torch.linalg.norm(d) / (torch.linalg.norm(t) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fem/traj_mix/fem_gt_shear.npz")
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    D = load(args.data)
    side, N, nt = D["side"], D["inp"].shape[0], args.n_test
    z = np.load(args.data, allow_pickle=True)
    if "load_mode" not in z.files:
        raise SystemExit(f"{args.data} has no load_mode -- regenerate with --mix-loads --save-trajectory")
    lm = torch.tensor(z["load_mode"].astype(np.int64), device=DEV)
    print(f"device={DEV}  loading-history  data={args.data}  N={N} side={side}")
    print(f"load-mode dist: " + "  ".join(f"{LOAD_MODES[i]}={int((lm==i).sum())}" for i in range(3)))

    inp3, out, scal, mode = (D[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    oh = torch.eye(3, device=DEV)[lm][:, :, None, None].expand(N, 3, side, side)   # load-path one-hot field
    inp6 = torch.cat([inp3, oh], 1)

    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij")[::-1], -1
    ).astype(np.float32)).to(DEV)
    tr = torch.arange(0, N - nt, device=DEV); te = torch.arange(N - nt, N, device=DEV)

    def run(tag, inp, in_ch):
        im, istd, om, ostd, sm, sstd = norm_from(inp[tr], out[tr], scal[tr])
        nin = lambda t: (t - im) / istd
        torch.manual_seed(0)
        m = FNOField(in_ch=in_ch, modes=args.modes).to(DEV)
        train_operator(m, nin(inp[tr]), (out[tr] - om) / ostd, (scal[tr] - sm) / sstd,
                       mode[tr], cg, args.epochs, args.lr)
        pred = predict_raw(m, nin(inp[te]), (scal[te] - sm) / sstd, cg, ostd, om)
        rl = rel_l2_per_mode(pred, out[te], mode[te])
        td = tangential_dir_error(pred, out[te], mode[te])
        per_lm = {LOAD_MODES[i]: per_subset_rel_l2(pred, out[te], lm[te] == i) for i in range(3)}
        print(f"[{tag:14s}] overall={rl['overall']:.3f}  tang_dir={td:.1f}deg  "
              f"per-path: " + " ".join(f"{k}={v:.3f}" for k, v in per_lm.items()))
        return {"relative_l2": rl, "tangential_dir_error_deg": td, "per_load_mode": per_lm,
                "params": count_parameters(m)}

    res = {"baseline_endpoint_3ch": run("baseline", inp3, 3),
           "loadmode_6ch": run("loadmode", inp6, 6)}

    # model-free corroboration (independent of FNO training quality)
    mf = path_dependence_modelfree(D["disp"] if "disp" in D else out.cpu().numpy(),
                                   z["params"], z["load_mode"])
    print(f"\nmodel-free kNN (distance-matched): same-mode resid={mf['same_mode_resid']:.3f} "
          f"(ep-dist {mf['same_mode_ep_dist']:.3f})  cross-mode resid={mf['cross_mode_resid']:.3f} "
          f"(ep-dist {mf['cross_mode_ep_dist']:.3f})  cross/same={mf['cross_over_same']:.2f} "
          f"(>1 => path-dependent)")

    b = res["baseline_endpoint_3ch"]["relative_l2"]["overall"]
    h = res["loadmode_6ch"]["relative_l2"]["overall"]
    gain = (b - h) / b * 100
    verdict = ("loading history HELPS -> final state is path-dependent" if gain > 3 else
               "loading history ~NULL -> path-dependence weak in this FEM; ceiling is elsewhere")
    print(f"\nVERDICT: baseline {b:.3f} -> +loadmode {h:.3f}  ({gain:+.1f}%)  => {verdict}")

    out_dir = RUNS / "phase6"; ensure(out_dir)
    json.dump({"data": args.data, "n_test": nt, "epochs": args.epochs,
               "load_mode_dist": {LOAD_MODES[i]: int((lm == i).sum()) for i in range(3)},
               "models": res, "model_free": mf, "gain_pct": gain, "verdict": verdict},
              open(out_dir / "loading_history.json", "w"), indent=2, default=float)
    print(f"saved {out_dir/'loading_history.json'}")


if __name__ == "__main__":
    main()
