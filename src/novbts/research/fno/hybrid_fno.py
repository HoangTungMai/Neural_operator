#!/usr/bin/env python3
"""Multi-seed hybrid global/local operator benchmark on the corrected-BC IPC GT.

Official Track-B run for the hybrid direction
(docs/kse2026/hybrid_fno_direction_report.md). Compares, under the exact
fem_benchmark split/budget/normalisation:

  fno                 global-only spectral trunk (paper baseline)
  unet                local-only conv encoder-decoder (strongest Table-1 rival)
  local_refined_fno   FNO trunk + lightweight local CNN refinement head
  ufno                U-Net-style encoder-decoder with a spectral trunk
                      (U-FNO family, Wen et al. 2022 -- application, not a new
                      architecture claim)

Reports per-seed and mean/std per-mode rel-L2, tangential direction error,
params, train time, inference throughput, and paired per-seed deltas vs the
FNO baseline (the 3-seed probe showed FNO seed variance comparable to the
FNO-vs-U-Net gap, so paired multi-seed evidence is required before any paper
claim).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn

from novbts.research.models import SpectralConv2d, count_parameters
from novbts.research.fno.fem_benchmark import load, norm_from
from novbts.research.fno.field2field import (
    DEV, FNOField, predict_raw, rel_l2_per_mode, tangential_dir_error,
    throughput, train_operator,
)
from novbts.research.fno.vbts_baselines import UNetField
from novbts.paths import RUNS, ensure


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GELU(),
    )


class LocalRefinedFNO(nn.Module):
    """FNO trunk plus local CNN refinement head.

    Tests whether local contact-detail correction helps without replacing the
    spectral global-coupling core.
    """

    def __init__(self, in_ch=3, out_ch=3, width=48, modes=12, with_slip_head=False):
        super().__init__()
        self.with_slip_head = with_slip_head
        self.lift = nn.Conv2d(in_ch + 2, width, 1)
        self.local_stem = conv_block(in_ch + 2, width)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes, modes) for _ in range(4)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.base_head = nn.Sequential(nn.Conv2d(width, 96, 1), nn.GELU(), nn.Conv2d(96, out_ch, 1))
        self.refine = nn.Sequential(
            nn.Conv2d(width * 2 + in_ch + 2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, out_ch, 1),
        )
        if with_slip_head:
            self.slip_head = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 4))

    def forward(self, field, scal):
        b, _, h, w = field.shape
        sc = scal[:, :, None, None].expand(b, scal.shape[1], h, w)
        x0 = torch.cat([field, sc], 1)
        local = self.local_stem(x0)
        h0 = self.lift(x0) + local
        h1 = h0
        for spec, pw in zip(self.spectral, self.pointwise):
            h1 = torch.nn.functional.gelu(spec(h1) + pw(h1))
        out = self.base_head(h1) + self.refine(torch.cat([h1, local, x0], 1))
        if self.with_slip_head:
            return out, self.slip_head(h1.mean(dim=(2, 3)))
        return out


class UFNOField(nn.Module):
    """Small U-Net/FNO hybrid: full-res spectral trunk with U-Net local skips."""

    def __init__(self, in_ch=3, out_ch=3, width=32, modes=12):
        super().__init__()
        c0 = in_ch + 2
        self.e1 = conv_block(c0, width)
        self.e2 = conv_block(width, width * 2)
        self.e3 = conv_block(width * 2, width * 4)
        self.pool = nn.MaxPool2d(2)
        self.fno_lift = nn.Conv2d(c0, width * 2, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width * 2, modes, modes) for _ in range(4)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width * 2, width * 2, 1) for _ in range(4)])
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, 2)
        self.d2 = conv_block(width * 6, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, 2)
        self.d1 = conv_block(width * 4, width)
        self.head = nn.Conv2d(width, out_ch, 1)

    def forward(self, field, scal):
        b, _, h, w = field.shape
        sc = scal[:, :, None, None].expand(b, scal.shape[1], h, w)
        x0 = torch.cat([field, sc], 1)
        e1 = self.e1(x0)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))

        f = self.fno_lift(x0)
        for spec, pw in zip(self.spectral, self.pointwise):
            f = torch.nn.functional.gelu(spec(f) + pw(f))
        f2 = torch.nn.functional.avg_pool2d(f, 2)

        d2 = self.d2(torch.cat([self.up2(e3), e2, f2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1, f], 1))
        return self.head(d1)


def make_field_model(name: str, modes: int = 12, with_slip_head: bool = False) -> nn.Module:
    """Factory for the main reported field->field surrogate (paper model switch)."""
    if name == "fno":
        return FNOField(modes=modes, with_slip_head=with_slip_head)
    if name == "lr_fno":
        return LocalRefinedFNO(modes=modes, with_slip_head=with_slip_head)
    raise ValueError(f"unknown field model {name!r} (expected 'fno' or 'lr_fno')")


def _stats(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "min": float(a.min()), "max": float(a.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/uipc/shear_res24_avg_swept_REALISTIC_BC.npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    D = load(args.data)
    side, n, nt = D["side"], D["inp"].shape[0], args.n_test
    print(f"device={DEV} hybrid multiseed data={args.data} N={n} side={side} seeds={args.seeds}",
          flush=True)
    cg = torch.tensor(np.stack(np.meshgrid(
        np.linspace(-1, 1, side), np.linspace(-1, 1, side), indexing="ij"
    )[::-1], -1).astype(np.float32)).to(DEV)
    inp, out, scal, mode = (D[k].to(DEV) for k in ("inp", "out", "scal", "mode"))
    tr = torch.arange(0, n - nt, device=DEV)
    te = torch.arange(n - nt, n, device=DEV)
    im, istd, om, ostd, sm, sstd = norm_from(inp[tr], out[tr], scal[tr])
    nin = lambda t: (t - im) / istd
    nsc = lambda t: (t - sm) / sstd
    nout = lambda t: (t - om) / ostd

    specs = [
        ("fno", lambda: FNOField(modes=args.modes)),
        ("unet", lambda: UNetField()),
        ("local_refined_fno", lambda: LocalRefinedFNO(modes=args.modes)),
        ("ufno", lambda: UFNOField(modes=args.modes)),
    ]
    results = {
        "gt": os.path.basename(args.data), "gt_path": args.data,
        "device": str(DEV), "seeds": args.seeds, "epochs": args.epochs,
        "train_frames": int(n - nt), "test_frames": nt, "models": {},
    }
    for name, ctor in specs:
        rows = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = ctor().to(DEV)
            t0 = time.perf_counter()
            secs, _ = train_operator(model, nin(inp[tr]), nout(out[tr]), nsc(scal[tr]),
                                     mode[tr], cg, args.epochs, args.lr)
            pred = predict_raw(model, nin(inp[te]), nsc(scal[te]), cg, ostd, om)
            rel = rel_l2_per_mode(pred, out[te], mode[te])
            direction = tangential_dir_error(pred, out[te], mode[te])
            rows.append({"seed": seed, "train_s": secs,
                         "wall_s": time.perf_counter() - t0,
                         "params": count_parameters(model),
                         "rel_l2": rel, "dir_deg": direction})
            print(f"{name:18s} seed={seed} rel={rel['overall']:.4f} "
                  f"full={rel['full_slip']:.4f} dir={direction:.2f} "
                  f"params={rows[-1]['params']/1e6:.2f}M train_s={secs:.1f}", flush=True)
        fps = throughput(model, nin(inp[te]), nsc(scal[te]), cg)
        overall = [r["rel_l2"]["overall"] for r in rows]
        results["models"][name] = {
            "runs": rows,
            "throughput_fps": fps,
            "params": rows[-1]["params"],
            "rel_l2_overall": _stats(overall),
            "dir_deg": _stats([r["dir_deg"] for r in rows]),
            "rel_l2_by_mode": {
                m: _stats([r["rel_l2"][m] for r in rows])
                for m in ("normal", "stick", "partial_slip", "full_slip")
            },
        }
        print(f"{name:18s} mean={np.mean(overall):.4f} std={np.std(overall):.4f} "
              f"fps={fps:.0f}", flush=True)

    # Paired per-seed deltas vs the FNO baseline (positive = model beats FNO).
    fno_overall = {r["seed"]: r["rel_l2"]["overall"] for r in results["models"]["fno"]["runs"]}
    for name in ("unet", "local_refined_fno", "ufno"):
        deltas = [fno_overall[r["seed"]] - r["rel_l2"]["overall"]
                  for r in results["models"][name]["runs"]]
        d = np.asarray(deltas)
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 else float("nan")
        results["models"][name]["paired_vs_fno"] = {
            "delta_rel_l2": deltas, "mean_delta": float(d.mean()),
            "all_positive": bool((d > 0).all()), "paired_t": t,
        }

    ensure(RUNS / "phase3_fem")
    out_path = RUNS / "phase3_fem" / "hybrid_multiseed.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved {out_path}", flush=True)
    print("HYBRID_MULTISEED_DONE", flush=True)


if __name__ == "__main__":
    main()
