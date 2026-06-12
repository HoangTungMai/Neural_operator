#!/usr/bin/env python3
"""
Phase 3 training.

Models:
  * FNO  (main operator)            — field regression
  * MLP  (mandatory coordinate baseline)
  * FNO + multi-task slip head (a)  — shared trunk, extra 4-class head
  * Separate slip classifier   (b)  — reads a trained FNO's field output

Slip heads (a) and (b) are trained BOTH so RQ1/ablation can compare them.

Loss for the operator:
  field MSE  +  lambda_dir * (1 - cos)  tangential-direction loss
Multi-task adds:
  + lambda_cls * cross-entropy(4-mode)

Saves checkpoints + a metrics json to runs/phase3/.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from novbts.models import CoordinateMLP, SpectralConv2d, count_parameters
from novbts.paths import ANALYTIC, RUNS

MODE_NAMES = ["normal", "stick", "partial_slip", "full_slip"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(path, device):
    d = np.load(path, allow_pickle=True)
    params = torch.tensor(d["params"], dtype=torch.float32, device=device)
    coords = torch.tensor(d["coords"], dtype=torch.float32, device=device)
    disp = torch.tensor(d["disp"], dtype=torch.float32, device=device)
    mode = torch.tensor(d["mode"].astype(np.int64), dtype=torch.long, device=device)
    return params, coords, disp, mode


# --- normalisation (params have very different scales; disp spans wide range) ---

def compute_norm(params, disp):
    pm = params.mean(0, keepdim=True)
    ps = params.std(0, keepdim=True)
    # Constant-in-train columns (e.g. geom=0 for sphere-only training) have
    # std~0.  Normalising them makes ANY OOD value (geom=1) explode to ~1/eps.
    # Pass such columns through (mean=0, std=1) so OOD inputs stay sane.
    const = ps < 1e-3
    pm = torch.where(const, torch.zeros_like(pm), pm)
    ps = torch.where(const, torch.ones_like(ps), ps)
    dm = disp.reshape(-1, disp.shape[-1]).mean(0)            # per-channel
    ds = disp.reshape(-1, disp.shape[-1]).std(0).clamp_min(1e-6)
    return pm, ps, dm, ds


def save_norm(path, pm, ps, dm, ds):
    np.savez(path, pm=pm.cpu().numpy(), ps=ps.cpu().numpy(),
             dm=dm.cpu().numpy(), ds=ds.cpu().numpy())


def load_norm(path, device):
    d = np.load(path)
    t = lambda k: torch.tensor(d[k], dtype=torch.float32, device=device)
    return t("pm"), t("ps"), t("dm"), t("ds")


# ---------------------------------------------------------------------------
# FNO with optional multi-task slip head
# ---------------------------------------------------------------------------

class FNO2dMultiTask(nn.Module):
    """TinyFNO2d + optional 4-class contact-mode head (multi-task slip head a)."""

    def __init__(self, params_dim=9, width=48, modes=12, out_dim=3, with_slip_head=False):
        super().__init__()
        self.width = width
        self.with_slip_head = with_slip_head
        self.fc0 = nn.Linear(params_dim + 2, width)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes, modes) for _ in range(4)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Linear(width, 96)
        self.fc2 = nn.Linear(96, out_dim)
        if with_slip_head:
            # pooled latent -> 4-mode logits
            self.slip_head = nn.Sequential(
                nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 4)
            )

    def forward(self, params, coords):
        batch, markers, _ = coords.shape
        side = int(math.sqrt(markers))
        params_grid = params[:, None, :].expand(batch, markers, params.shape[-1])
        x = torch.cat([params_grid, coords], dim=-1)
        x = self.fc0(x).view(batch, side, side, self.width).permute(0, 3, 1, 2)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = torch.nn.functional.gelu(spec(x) + pw(x))
        latent = x  # [B, width, side, side]
        h = latent.permute(0, 2, 3, 1)
        h = torch.nn.functional.gelu(self.fc1(h))
        field = self.fc2(h).reshape(batch, markers, -1)
        if self.with_slip_head:
            pooled = latent.mean(dim=(2, 3))  # [B, width]
            logits = self.slip_head(pooled)
            return field, logits
        return field


class SeparateSlipClassifier(nn.Module):
    """Slip head (b): a CNN classifier reading a (frozen) FNO's field output."""

    def __init__(self, in_ch=3, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, stride=2), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 4),
        )

    def forward(self, field, side):
        b, m, c = field.shape
        x = field.permute(0, 2, 1).reshape(b, c, side, side)
        return self.net(x)


# ---------------------------------------------------------------------------
# Losses & metrics
# ---------------------------------------------------------------------------

def field_loss(pred, target, lambda_dir=0.1):
    mse = torch.mean((pred - target).square())
    # tangential direction loss (cos similarity on in-plane components)
    pt, tt = pred[..., :2], target[..., :2]
    tmag = torch.linalg.norm(tt, dim=-1)
    mask = tmag > 1e-4
    if mask.any():
        cos = torch.nn.functional.cosine_similarity(pt[mask], tt[mask], dim=-1)
        dir_loss = (1.0 - cos).mean()
    else:
        dir_loss = torch.zeros((), device=pred.device)
    return mse + lambda_dir * dir_loss, mse.detach(), dir_loss.detach()


def relative_l2_per_mode(pred, target, mode):
    diff = pred - target
    rel = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=-1) / (
        torch.linalg.norm(target.reshape(target.shape[0], -1), dim=-1) + 1e-8)
    out = {"overall": float(rel.mean())}
    for i, name in enumerate(MODE_NAMES):
        m = mode == i
        out[name] = float(rel[m].mean()) if m.any() else float("nan")
    return out


def macro_f1(pred_labels, true_labels, n_cls=4):
    f1s = []
    for c in range(n_cls):
        tp = ((pred_labels == c) & (true_labels == c)).sum().item()
        fp = ((pred_labels == c) & (true_labels != c)).sum().item()
        fn = ((pred_labels != c) & (true_labels == c)).sum().item()
        f1s.append(2 * tp / (2 * tp + fp + fn + 1e-9))
    binary = {  # slip (partial+full) vs non-slip
        "slip_f1": _binary_slip_f1(pred_labels, true_labels)
    }
    return float(np.mean(f1s)), f1s, binary


def _binary_slip_f1(pred, true):
    ps, ts = (pred >= 2), (true >= 2)
    tp = (ps & ts).sum().item(); fp = (ps & ~ts).sum().item(); fn = (~ps & ts).sum().item()
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def make_loader(params, coords, disp, mode, bs, shuffle):
    coords_b = coords[None].expand(params.shape[0], -1, -1).contiguous()
    return DataLoader(TensorDataset(params, coords_b, disp, mode), batch_size=bs, shuffle=shuffle)


def train_operator(model, loader, device, epochs, lr, multitask=False, lambda_cls=0.5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for params, coords_b, disp, mode in loader:
            if multitask:
                pred, logits = model(params, coords_b)
                floss, _, _ = field_loss(pred, disp)
                loss = floss + lambda_cls * ce(logits, mode)
            else:
                pred = model(params, coords_b)
                loss, _, _ = field_loss(pred, disp)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
    return secs, vram


def train_separate_classifier(clf, fno, loader, device, epochs, lr, side):
    """loader yields NORMALISED params; fno outputs normalised field — fine,
    the classifier just needs a consistent representation."""
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    fno.eval()
    for ep in range(epochs):
        clf.train()
        for params, coords_b, disp, mode in loader:
            with torch.no_grad():
                out = fno(params, coords_b)
                field = out[0] if isinstance(out, tuple) else out
            logits = clf(field, side)
            loss = ce(logits, mode)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


@torch.no_grad()
def eval_operator(model, params, coords, disp, mode, bs, norm, multitask=False):
    """params/disp are RAW; norm=(pm,ps,dm,ds). Feeds normalised params,
    denormalises predictions, scores against RAW disp."""
    pm, ps, dm, ds = norm
    model.eval()
    pnorm = (params - pm) / ps
    coords_b = coords[None].expand(params.shape[0], -1, -1).contiguous()
    preds, logits_all = [], []
    for s in range(0, params.shape[0], bs):
        e = s + bs
        out = model(pnorm[s:e], coords_b[s:e])
        if multitask:
            preds.append(out[0]); logits_all.append(out[1])
        else:
            preds.append(out)
    pred = torch.cat(preds) * ds + dm  # denormalise
    rel = relative_l2_per_mode(pred, disp, mode)
    res = {"relative_l2": rel}
    if multitask:
        lab = torch.cat(logits_all).argmax(-1)
        mf1, per, binm = macro_f1(lab, mode)
        res["slip_head_a"] = {"macro_f1": mf1, "per_class_f1": per, **binm}
    return res, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ANALYTIC))
    ap.add_argument("--out-dir", default=str(RUNS / "phase3"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--clf-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fno-modes", type=int, default=12)
    ap.add_argument("--lambda-cls", type=float, default=0.1,
                    help="weight of slip CE in multitask (kept low so it does not "
                         "sacrifice field accuracy)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(7)
    data = Path(args.data_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  data={data}")

    tr = load_split(data / "train.npz", device)
    va = load_split(data / "val.npz", device)
    coords = tr[1]
    side = int(math.sqrt(coords.shape[0]))

    # normalise (params: per-dim; disp: per-channel) — big conditioning win
    pm, ps, dm, ds = compute_norm(tr[0], tr[2])
    norm = (pm, ps, dm, ds)
    save_norm(out / "norm.npz", pm, ps, dm, ds)
    tr_pn = (tr[0] - pm) / ps
    tr_dn = (tr[2] - dm) / ds
    loader = make_loader(tr_pn, coords, tr_dn, tr[3], args.batch_size, shuffle=True)

    summary = {"device": str(device), "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "train_frames": int(tr[0].shape[0]), "lambda_cls": args.lambda_cls,
               "normalised": True, "models": {}}

    # ---- MLP baseline ----
    print("\n[MLP] training...")
    mlp = CoordinateMLP(params_dim=9).to(device)
    secs, vram = train_operator(mlp, loader, device, args.epochs, args.lr)
    res, _ = eval_operator(mlp, va[0], coords, va[2], va[3], args.eval_batch_size, norm)
    summary["models"]["mlp"] = {"params": count_parameters(mlp), "train_s": secs,
                                "peak_vram_gb": vram, **res}
    torch.save(mlp.state_dict(), out / "mlp.pt")
    print(f"  rel L2 overall={res['relative_l2']['overall']:.4f}  {secs:.1f}s")

    # ---- FNO main (no head) ----
    print("\n[FNO] training...")
    fno = FNO2dMultiTask(params_dim=9, modes=args.fno_modes, with_slip_head=False).to(device)
    secs, vram = train_operator(fno, loader, device, args.epochs, args.lr)
    res, _ = eval_operator(fno, va[0], coords, va[2], va[3], args.eval_batch_size, norm)
    summary["models"]["fno"] = {"params": count_parameters(fno), "train_s": secs,
                                "peak_vram_gb": vram, **res}
    torch.save(fno.state_dict(), out / "fno.pt")
    print(f"  rel L2 overall={res['relative_l2']['overall']:.4f}  {secs:.1f}s")

    # ---- FNO + multi-task slip head (a) ----
    print("\n[FNO+slip head a] training (multi-task)...")
    fno_mt = FNO2dMultiTask(params_dim=9, modes=args.fno_modes, with_slip_head=True).to(device)
    secs, vram = train_operator(fno_mt, loader, device, args.epochs, args.lr,
                                multitask=True, lambda_cls=args.lambda_cls)
    res, _ = eval_operator(fno_mt, va[0], coords, va[2], va[3], args.eval_batch_size,
                           norm, multitask=True)
    summary["models"]["fno_multitask_a"] = {"params": count_parameters(fno_mt), "train_s": secs,
                                            "peak_vram_gb": vram, **res}
    torch.save(fno_mt.state_dict(), out / "fno_multitask_a.pt")
    print(f"  rel L2 overall={res['relative_l2']['overall']:.4f}  "
          f"slip macro-F1={res['slip_head_a']['macro_f1']:.4f}  {secs:.1f}s")

    # ---- Separate slip classifier (b) on frozen FNO ----
    print("\n[slip classifier b] training on frozen FNO field...")
    clf = SeparateSlipClassifier().to(device)
    train_separate_classifier(clf, fno, loader, device, args.clf_epochs, args.lr, side)
    # eval b
    fno.eval(); clf.eval()
    with torch.no_grad():
        coords_b = coords[None].expand(va[0].shape[0], -1, -1).contiguous()
        field = fno((va[0] - pm) / ps, coords_b)
        labs = clf(field, side).argmax(-1)
    mf1, per, binb = macro_f1(labs, va[3])
    summary["models"]["slip_classifier_b"] = {"params": count_parameters(clf),
                                               "macro_f1": mf1, "per_class_f1": per, **binb}
    torch.save(clf.state_dict(), out / "slip_classifier_b.pt")
    print(f"  slip macro-F1={mf1:.4f}  slip-binary-F1={binb['slip_f1']:.4f}")

    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved checkpoints + summary to {out}")


if __name__ == "__main__":
    main()
