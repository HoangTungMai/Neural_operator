import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODE_NAMES = ["normal", "stick", "partial_slip", "full_slip"]


@dataclass
class ModelResult:
    model: str
    device: str
    torch_version: str
    cuda_version: str | None
    gpu_name: str | None
    frames_train: int
    frames_test: int
    markers: int
    batch_size: int
    epochs: int
    parameters: int
    train_seconds: float
    peak_vram_gb: float | None
    overall_relative_l2: float
    overall_rmse: float
    mode_relative_l2: dict[str, float]
    mode_rmse: dict[str, float]
    slip_binary_f1_from_field: float
    slip_binary_best_f1_from_field: float
    slip_binary_best_threshold: float
    slip_score_mae: float


def make_marker_grid(side: int, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(-1.0, 1.0, side, device=device)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def sample_params_per_mode(n_per_mode: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    params = []
    modes = []
    for mode in range(4):
        p = torch.empty(n_per_mode, 8, device=device).uniform_(0.0, 1.0)
        p[:, 0:2] = p[:, 0:2] * 1.4 - 0.7
        p[:, 2] = p[:, 2] * 0.9 + 0.1
        p[:, 3] = p[:, 3] * 0.25 + 0.08
        p[:, 6] = p[:, 6] * 0.6 + 0.3
        p[:, 7] = p[:, 7] * 3.0 + 0.5
        theta = torch.empty(n_per_mode, device=device).uniform_(0.0, 2.0 * math.pi)
        if mode == 0:
            shear_mag = torch.empty(n_per_mode, device=device).uniform_(0.0, 0.02)
        elif mode == 1:
            shear_mag = p[:, 6] * torch.empty(n_per_mode, device=device).uniform_(0.08, 0.35)
        elif mode == 2:
            shear_mag = p[:, 6] * torch.empty(n_per_mode, device=device).uniform_(0.48, 0.72)
        else:
            shear_mag = p[:, 6] * torch.empty(n_per_mode, device=device).uniform_(0.90, 1.30)
        p[:, 4] = shear_mag * torch.cos(theta)
        p[:, 5] = shear_mag * torch.sin(theta)
        params.append(p)
        modes.append(torch.full((n_per_mode,), mode, device=device, dtype=torch.long))
    params_all = torch.cat(params, dim=0)
    modes_all = torch.cat(modes, dim=0)
    perm = torch.randperm(params_all.shape[0], device=device)
    return params_all[perm], modes_all[perm]


def marker_field(params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    x0, y0, depth, radius, shear_x, shear_y, mu, stiffness = params.unbind(-1)
    dx = coords[:, 0][None, :] - x0[:, None]
    dy = coords[:, 1][None, :] - y0[:, None]
    r2 = dx.square() + dy.square()
    sigma = radius[:, None].clamp_min(0.04)
    contact = torch.exp(-r2 / (2.0 * sigma.square()))
    radial = torch.sqrt(r2 + 1e-8)

    shear_mag = torch.sqrt(shear_x.square() + shear_y.square() + 1e-8)
    drive = shear_mag / (mu + 1e-5)
    partial_center = torch.sigmoid(18.0 * (drive - 0.48))[:, None]
    full_slip = torch.sigmoid(18.0 * (drive - 0.86))[:, None]
    stick = 1.0 - partial_center

    # Stick decays smoothly; partial slip introduces a sharp annular transition;
    # full slip makes tangential motion closer to rigid translation in the contact patch.
    stick_profile = torch.exp(-radial / (sigma + 1e-5))
    annulus = torch.sigmoid(80.0 * (radial - 0.58 * sigma))
    partial_profile = (1.0 - annulus) * stick_profile + annulus * 0.92
    full_profile = torch.ones_like(stick_profile)
    tangent_profile = stick * stick_profile + (1.0 - stick) * (
        (1.0 - full_slip) * partial_profile + full_slip * full_profile
    )

    ux = shear_x[:, None] * contact * tangent_profile
    uy = shear_y[:, None] * contact * tangent_profile
    radial_push = 0.12 * depth[:, None] * contact / (sigma + 1e-5)
    ux = ux + radial_push * dx
    uy = uy + radial_push * dy
    uz = -depth[:, None] * contact / stiffness[:, None].sqrt()
    return torch.stack([ux, uy, uz], dim=-1)


class CoordinateMLP(nn.Module):
    def __init__(self, params_dim: int = 8, hidden: int = 192, out_dim: int = 3):
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
        params_expanded = params[:, None, :].expand(params.shape[0], coords.shape[1], params.shape[-1])
        return self.net(torch.cat([params_expanded, coords], dim=-1))


class TinyDeepONet(nn.Module):
    def __init__(self, params_dim: int = 8, width: int = 256, basis: int = 256, out_dim: int = 3):
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


class SpectralConv2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (width * width)
        self.weights1 = nn.Parameter(scale * torch.randn(width, width, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(width, width, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch,
            self.width,
            x.shape[-2],
            x.shape[-1] // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))


class TinyFNO2d(nn.Module):
    def __init__(self, params_dim: int = 8, width: int = 48, modes: int = 12, out_dim: int = 3):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(params_dim + 2, width)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes, modes) for _ in range(4)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Linear(width, 96)
        self.fc2 = nn.Linear(96, out_dim)

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        batch, markers, _ = coords.shape
        side = int(math.sqrt(markers))
        params_grid = params[:, None, :].expand(batch, markers, params.shape[-1])
        x = torch.cat([params_grid, coords], dim=-1)
        x = self.fc0(x).view(batch, side, side, self.width).permute(0, 3, 1, 2)
        for spec, pw in zip(self.spectral, self.pointwise):
            x = torch.nn.functional.gelu(spec(x) + pw(x))
        x = x.permute(0, 2, 3, 1)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.reshape(batch, markers, -1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def slip_score(field: torch.Tensor, coords: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    tangential = torch.linalg.norm(field[..., :2], dim=-1)
    dx = coords[:, 0][None, :] - params[:, 0:1]
    dy = coords[:, 1][None, :] - params[:, 1:2]
    r_norm = torch.sqrt(dx.square() + dy.square()) / (params[:, 3:4] + 1e-6)
    center_weight = torch.exp(-((r_norm / 0.45).square()))
    outer_weight = torch.exp(-(((r_norm - 0.85) / 0.22).square()))
    center = (tangential * center_weight).sum(dim=-1) / (center_weight.sum(dim=-1) + 1e-6)
    outer = (tangential * outer_weight).sum(dim=-1) / (outer_weight.sum(dim=-1) + 1e-6)
    return outer / (center + 1e-6)


def evaluate(model: nn.Module, params: torch.Tensor, coords: torch.Tensor, target: torch.Tensor, modes: torch.Tensor, batch: int):
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, params.shape[0], batch):
            end = start + batch
            preds.append(model(params[start:end], coords[start:end]))
    pred = torch.cat(preds, dim=0)
    diff = pred - target
    rel = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=-1) / (
        torch.linalg.norm(target.reshape(target.shape[0], -1), dim=-1) + 1e-8
    )
    rmse_frame = torch.sqrt(diff.square().mean(dim=(1, 2)))
    mode_rel = {}
    mode_rmse = {}
    for idx, name in enumerate(MODE_NAMES):
        mask = modes == idx
        mode_rel[name] = float(rel[mask].mean().detach().cpu())
        mode_rmse[name] = float(rmse_frame[mask].mean().detach().cpu())

    gt_score = slip_score(target, coords[0], params)
    pred_score = slip_score(pred, coords[0], params)
    gt_slip = modes >= 2
    pred_slip = pred_score > 0.68
    tp = torch.logical_and(gt_slip, pred_slip).sum().float()
    fp = torch.logical_and(~gt_slip, pred_slip).sum().float()
    fn = torch.logical_and(gt_slip, ~pred_slip).sum().float()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    thresholds = torch.linspace(
        float(pred_score.min().detach().cpu()),
        float(pred_score.max().detach().cpu()),
        100,
        device=pred_score.device,
    )
    best_f1 = torch.zeros((), device=pred_score.device)
    best_threshold = thresholds[0]
    for threshold in thresholds:
        pred_slip_at_t = pred_score > threshold
        tp_t = torch.logical_and(gt_slip, pred_slip_at_t).sum().float()
        fp_t = torch.logical_and(~gt_slip, pred_slip_at_t).sum().float()
        fn_t = torch.logical_and(gt_slip, ~pred_slip_at_t).sum().float()
        f1_t = 2 * tp_t / (2 * tp_t + fp_t + fn_t + 1e-8)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = threshold
    return {
        "overall_relative_l2": float(rel.mean().detach().cpu()),
        "overall_rmse": float(rmse_frame.mean().detach().cpu()),
        "mode_relative_l2": mode_rel,
        "mode_rmse": mode_rmse,
        "slip_binary_f1_from_field": float(f1.detach().cpu()),
        "slip_binary_best_f1_from_field": float(best_f1.detach().cpu()),
        "slip_binary_best_threshold": float(best_threshold.detach().cpu()),
        "slip_score_mae": float(torch.mean(torch.abs(pred_score - gt_score)).detach().cpu()),
    }


def run_model(model_name: str, args, device: torch.device, coords: torch.Tensor, train, test) -> ModelResult:
    train_params, train_modes, train_disp = train
    test_params, test_modes, test_disp = test
    train_coords = coords[None, :, :].expand(train_params.shape[0], -1, -1).contiguous()
    test_coords = coords[None, :, :].expand(test_params.shape[0], -1, -1).contiguous()
    loader = DataLoader(
        TensorDataset(train_params, train_coords, train_disp),
        batch_size=args.batch_size,
        shuffle=True,
    )
    if model_name == "mlp":
        model = CoordinateMLP()
    elif model_name == "deeponet":
        model = TinyDeepONet()
    else:
        model = TinyFNO2d()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        for params, batch_coords, disp in loader:
            pred = model(params, batch_coords)
            loss = torch.mean((pred - disp).square())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    metrics = evaluate(model, test_params, test_coords, test_disp, test_modes, args.eval_batch_size)
    return ModelResult(
        model=model_name,
        device=str(device),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        frames_train=train_params.shape[0],
        frames_test=test_params.shape[0],
        markers=coords.shape[0],
        batch_size=args.batch_size,
        epochs=args.epochs,
        parameters=count_parameters(model),
        train_seconds=train_seconds,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None,
        **metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-mode", type=int, default=4000)
    parser.add_argument("--test-per-mode", type=int, default=1000)
    parser.add_argument("--marker-side", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--models", nargs="+", choices=["mlp", "deeponet", "fno"], default=["mlp", "deeponet"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(7)
    coords = make_marker_grid(args.marker_side, device)
    train_params, train_modes = sample_params_per_mode(args.train_per_mode, device)
    test_params, test_modes = sample_params_per_mode(args.test_per_mode, device)
    train_disp = marker_field(train_params, coords)
    test_disp = marker_field(test_params, coords)
    train = (train_params, train_modes, train_disp)
    test = (test_params, test_modes, test_disp)
    results = [asdict(run_model(name, args, device, coords, train, test)) for name in args.models]
    print(json.dumps({"mode_names": MODE_NAMES, "results": results}, indent=2))


if __name__ == "__main__":
    main()
