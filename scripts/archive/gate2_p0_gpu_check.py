import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RunResult:
    device: str
    torch_version: str
    cuda_version: str | None
    gpu_name: str | None
    total_vram_gb: float | None
    frames: int
    markers: int
    batch_size: int
    epochs: int
    params_dim: int
    channels: int
    model: str
    parameters: int
    train_seconds: float
    train_seconds_per_epoch: float
    final_train_mse: float
    val_mse: float
    inference_frames_per_second: float
    peak_vram_gb: float | None


def make_marker_grid(side: int, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(-1.0, 1.0, side, device=device)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def synthetic_marker_field(params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    # params: x0, y0, depth, radius, shear_x, shear_y, mu, stiffness
    x0, y0, depth, radius, shear_x, shear_y, mu, stiffness = params.unbind(-1)
    dx = coords[:, 0][None, :] - x0[:, None]
    dy = coords[:, 1][None, :] - y0[:, None]
    r2 = dx.square() + dy.square()
    sigma = radius[:, None].clamp_min(0.05)
    contact = torch.exp(-r2 / (2.0 * sigma.square()))
    radial = torch.sqrt(r2 + 1e-8)
    slip_drive = torch.sqrt(shear_x.square() + shear_y.square()) / (mu + 1e-4)
    slip = torch.sigmoid(12.0 * (slip_drive - 0.65))[:, None]
    stick_decay = torch.exp(-radial / sigma)
    ux = depth[:, None] * 0.15 * contact * dx / (sigma + 1e-4)
    uy = depth[:, None] * 0.15 * contact * dy / (sigma + 1e-4)
    ux = ux + shear_x[:, None] * contact * ((1.0 - slip) * stick_decay + slip)
    uy = uy + shear_y[:, None] * contact * ((1.0 - slip) * stick_decay + slip)
    uz = -depth[:, None] * contact / stiffness[:, None].sqrt()
    return torch.stack([ux, uy, uz], dim=-1)


class CoordinateMLP(nn.Module):
    def __init__(self, params_dim: int, hidden: int = 192, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(params_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        bsz, markers, _ = coords.shape
        params_expanded = params[:, None, :].expand(bsz, markers, params.shape[-1])
        return self.net(torch.cat([params_expanded, coords], dim=-1))


class TinyDeepONet(nn.Module):
    def __init__(self, params_dim: int, width: int = 128, basis: int = 96, out_dim: int = 3):
        super().__init__()
        self.out_dim = out_dim
        self.basis = basis
        self.branch = nn.Sequential(
            nn.Linear(params_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, basis * out_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, basis),
        )
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        branch = self.branch(params).view(params.shape[0], self.out_dim, self.basis)
        trunk = self.trunk(coords)
        return torch.einsum("bck,bmk->bmc", branch, trunk) / math.sqrt(self.basis) + self.bias


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--marker-side", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--model", choices=["mlp", "deeponet"], default="deeponet")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    params_dim = 8
    coords = make_marker_grid(args.marker_side, device)
    params = torch.empty(args.frames, params_dim, device=device).uniform_(0.0, 1.0)
    params[:, 0:2] = params[:, 0:2] * 1.4 - 0.7
    params[:, 2] = params[:, 2] * 0.9 + 0.1
    params[:, 3] = params[:, 3] * 0.35 + 0.08
    params[:, 4:6] = params[:, 4:6] * 0.8 - 0.4
    params[:, 6] = params[:, 6] * 0.8 + 0.2
    params[:, 7] = params[:, 7] * 3.0 + 0.5
    disp = synthetic_marker_field(params, coords)
    coords_all = coords[None, :, :].expand(args.frames, -1, -1).contiguous()

    train_n = int(args.frames * 0.8)
    train_ds = TensorDataset(params[:train_n], coords_all[:train_n], disp[:train_n])
    val_params = params[train_n:]
    val_coords = coords_all[train_n:]
    val_disp = disp[train_n:]
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model: nn.Module
    if args.model == "mlp":
        model = CoordinateMLP(params_dim)
    else:
        model = TinyDeepONet(params_dim)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = 0.0
    for _ in range(args.epochs):
        model.train()
        for batch_params, batch_coords, batch_disp in loader:
            pred = model(batch_params, batch_coords)
            loss = torch.mean((pred - batch_disp).square())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        val_sse = torch.zeros((), device=device)
        val_count = 0
        eval_batch_size = min(args.batch_size, 64)
        for start_idx in range(0, val_params.shape[0], eval_batch_size):
            end_idx = start_idx + eval_batch_size
            val_pred = model(val_params[start_idx:end_idx], val_coords[start_idx:end_idx])
            diff = val_pred - val_disp[start_idx:end_idx]
            val_sse = val_sse + diff.square().sum()
            val_count += diff.numel()
        val_mse = float((val_sse / val_count).cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_start = time.perf_counter()
        repeats = 100
        infer_batch = min(args.batch_size, val_params.shape[0])
        for _ in range(repeats):
            _ = model(val_params[:infer_batch], val_coords[:infer_batch])
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_seconds = time.perf_counter() - infer_start

    result = RunResult(
        device=str(device),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        total_vram_gb=(
            torch.cuda.get_device_properties(0).total_memory / 1024**3
            if device.type == "cuda"
            else None
        ),
        frames=args.frames,
        markers=args.marker_side * args.marker_side,
        batch_size=args.batch_size,
        epochs=args.epochs,
        params_dim=params_dim,
        channels=3,
        model=args.model,
        parameters=count_parameters(model),
        train_seconds=train_seconds,
        train_seconds_per_epoch=train_seconds / args.epochs,
        final_train_mse=final_loss,
        val_mse=val_mse,
        inference_frames_per_second=(infer_batch * repeats) / infer_seconds,
        peak_vram_gb=(
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
        ),
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
