#!/usr/bin/env python3
"""
Experiment #2 — field -> field operator test (the proper operator-learning setup).

The param-vector -> field framing in phase3_train.py inherently favours a
coordinate MLP (each point gets the full param vector and solves locally).  The
NATURAL operator-learning setup is field -> field: feed an indentation/penetration
FIELD on the grid and predict the displacement FIELD.  Here a point's
displacement depends on the WHOLE contact (elastic Green's function), so a purely
local per-point MLP should lose to FNO's global spectral mixing.  This is the
decisive test of whether the operator earns its keep.

Input field  (3 ch on HxW grid):
  ch0 = penetration(x,y) = max(0, d - r^2/(2R))   (sphere indentation profile)
  ch1 = shear_x * contact_mask
  ch2 = shear_y * contact_mask
Output field (3 ch): displacement (ux,uy,uz) from Hertz-Mindlin GT.

Compares:
  * FNO   (field -> field, global spectral)
  * MLP   (per-point: [3 input-ch @ point, x, y, mu, E] -> disp)   <- local baseline
Reports per-mode relative L2 for both.  If FNO < MLP here, the operator framing
is justified (unlike the param-vector setup).
"""
import argparse
import math
import time

import numpy as np
import torch
from torch import nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate3_operator_slip_check import SpectralConv2d, count_parameters
from gt_hertz_mindlin import hertz_mindlin_field, MODE_NAMES, hertz_scalars

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_grid(side):
    xs = np.linspace(-1, 1, side)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], -1).astype(np.float32)


def gen_dataset(n, side, rng):
    """Returns input fields [n,3,H,W], output disp [n,3,H,W], scalars [n,2], mode[n]."""
    coords = make_grid(side)                       # [M,2]
    X = coords[:, 0].reshape(side, side)
    Y = coords[:, 1].reshape(side, side)
    inp = np.zeros((n, 3, side, side), np.float32)
    out = np.zeros((n, 3, side, side), np.float32)
    scal = np.zeros((n, 2), np.float32)
    modes = np.zeros(n, np.int32)
    factor_by_mode = [(0.0, 0.02), (0.08, 0.35), (0.48, 0.72), (0.90, 1.30)]
    for i in range(n):
        cx, cy = rng.uniform(-0.5, 0.5, 2)
        depth = rng.uniform(0.1, 1.0)
        R = rng.uniform(0.08, 0.33)
        mu = rng.uniform(0.3, 0.9)
        E = rng.uniform(0.5, 3.5)
        m = i % 4
        theta = rng.uniform(0, 2 * np.pi)
        smag = mu * rng.uniform(*factor_by_mode[m])
        sx, sy = smag * np.cos(theta), smag * np.sin(theta)
        # GT output field via Hertz-Mindlin
        p = np.array([[cx, cy, depth, R, sx, sy, mu, E, 0.0]])
        disp, mode = hertz_mindlin_field(p, coords)   # [1,M,3]
        out[i] = disp[0].reshape(side, side, 3).transpose(2, 0, 1)
        modes[i] = mode[0]
        # input penetration field
        a, _, _, _ = hertz_scalars(np.array([depth]), np.array([R]), np.array([E]))
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        pen = np.maximum(0.0, depth - r2 / (2 * R))
        mask = (np.sqrt(r2) <= a[0]).astype(np.float32)
        inp[i, 0] = pen
        inp[i, 1] = sx * mask
        inp[i, 2] = sy * mask
        scal[i] = [mu, E]
    return (torch.tensor(inp), torch.tensor(out),
            torch.tensor(scal), torch.tensor(modes.astype(np.int64)), coords)


class FNOField(nn.Module):
    """field -> field FNO."""
    def __init__(self, in_ch=3, width=48, modes=12, out_ch=3):
        super().__init__()
        self.width = width
        self.fc0 = nn.Conv2d(in_ch + 2, width, 1)   # +2 scalar channels (mu,E)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes, modes) for _ in range(4)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Conv2d(width, 96, 1)
        self.fc2 = nn.Conv2d(96, out_ch, 1)

    def forward(self, field, scal):
        b, c, h, w = field.shape
        sc = scal[:, :, None, None].expand(b, scal.shape[1], h, w)
        x = torch.cat([field, sc], 1)
        x = self.fc0(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = torch.nn.functional.gelu(spec(x) + pw(x))
        x = torch.nn.functional.gelu(self.fc1(x))
        return self.fc2(x)


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
        f = field.permute(0, 2, 3, 1)                       # [b,h,w,c]
        cg = coord_grid[None].expand(b, h, w, 2)
        sc = scal[:, None, None, :].expand(b, h, w, 2)
        x = torch.cat([f, cg, sc], -1)
        out = self.net(x)                                   # [b,h,w,3]
        return out.permute(0, 3, 1, 2)


def rel_l2_per_mode(pred, tgt, mode):
    d = (pred - tgt).reshape(pred.shape[0], -1)
    t = tgt.reshape(tgt.shape[0], -1)
    rel = torch.linalg.norm(d, dim=1) / (torch.linalg.norm(t, dim=1) + 1e-8)
    out = {"overall": float(rel.mean())}
    for i, n in enumerate(MODE_NAMES):
        m = mode == i
        out[n] = float(rel[m].mean()) if m.any() else float("nan")
    return out


def train(model, inp, out, scal, coord_grid, epochs, is_mlp, bs=64):
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    n = inp.shape[0]
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(n, device=inp.device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            pred = model(inp[idx], scal[idx], coord_grid) if is_mlp else model(inp[idx], scal[idx])
            loss = (pred - out[idx]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return time.perf_counter() - t0


@torch.no_grad()
def evaluate(model, inp, out, scal, coord_grid, mode, is_mlp, bs=128):
    preds = []
    for s in range(0, inp.shape[0], bs):
        e = s + bs
        p = model(inp[s:e], scal[s:e], coord_grid) if is_mlp else model(inp[s:e], scal[s:e])
        preds.append(p)
    return rel_l2_per_mode(torch.cat(preds), out, mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=32)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--modes", type=int, default=12)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    print(f"device={DEV}  field->field {args.side}x{args.side}")

    tr = gen_dataset(args.train, args.side, rng)
    te = gen_dataset(args.test, args.side, rng)
    coord_grid = torch.tensor(make_grid(args.side).reshape(args.side, args.side, 2)).to(DEV)
    tri = [t.to(DEV) for t in tr[:4]]
    tei = [t.to(DEV) for t in te[:4]]

    # normalise input/output per-channel (from train)
    im = tri[0].mean((0, 2, 3), keepdim=True); istd = tri[0].std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    om = tri[1].mean((0, 2, 3), keepdim=True); ostd = tri[1].std((0, 2, 3), keepdim=True).clamp_min(1e-6)
    sm = tri[2].mean(0, keepdim=True); sstd = tri[2].std(0, keepdim=True).clamp_min(1e-6)
    def npz(t, m, s): return (t - m) / s
    tri_n = (npz(tri[0], im, istd), npz(tri[1], om, ostd), npz(tri[2], sm, sstd), tri[3])
    tei_n = (npz(tei[0], im, istd), tei[1], npz(tei[2], sm, sstd), tei[3])  # eval on RAW out

    results = {}
    for name, build, is_mlp in [
        ("FNO_field2field", lambda: FNOField(modes=args.modes).to(DEV), False),
        ("MLP_perpoint",    lambda: PerPointMLP().to(DEV), True),
    ]:
        torch.manual_seed(0)
        model = build()
        secs = train(model, tri_n[0], tri_n[1], tri_n[2], coord_grid, args.epochs, is_mlp)
        # eval: denormalise pred
        @torch.no_grad()
        def pred_raw(inp, scal):
            preds = []
            for s in range(0, inp.shape[0], 128):
                e = s + 128
                p = model(inp[s:e], scal[s:e], coord_grid) if is_mlp else model(inp[s:e], scal[s:e])
                preds.append(p * ostd + om)
            return torch.cat(preds)
        pr = pred_raw(tei_n[0], tei_n[2])
        rel = rel_l2_per_mode(pr, tei[1], tei[3])
        results[name] = {"params": count_parameters(model), "train_s": round(secs, 1), "rel_l2": rel}
        print(f"{name:18s} params={count_parameters(model):>8} relL2={rel['overall']:.4f}  "
              + " ".join(f"{k}={rel[k]:.3f}" for k in MODE_NAMES) + f"  {secs:.0f}s")

    import json
    Path("runs/phase3_f2f").mkdir(parents=True, exist_ok=True)
    json.dump(results, open("runs/phase3_f2f/results.json", "w"), indent=2)
    fno = results["FNO_field2field"]["rel_l2"]["overall"]
    mlp = results["MLP_perpoint"]["rel_l2"]["overall"]
    print(f"\nVERDICT: FNO {'BEATS' if fno < mlp else 'LOSES TO'} MLP  "
          f"(FNO {fno:.4f} vs MLP {mlp:.4f})  -> operator framing "
          f"{'justified' if fno < mlp else 'still not justified'}")


if __name__ == "__main__":
    main()
