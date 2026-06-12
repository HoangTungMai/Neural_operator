"""Neural-operator and baseline model definitions, shared across the package.

Extracted from the original gate-3 operator/slip check so every live training
and eval module imports its models from one place.  Contains:
  * CoordinateMLP   — per-point MLP baseline (param-vector -> displacement)
  * TinyDeepONet    — DeepONet baseline (negative ablation)
  * SpectralConv2d  — 2-D Fourier layer
  * TinyFNO2d       — the FNO operator (param->field framing)
  * count_parameters

The field->field FNO (FNOField) lives in novbts.operator.field2field; the
param->field multitask FNO (FNO2dMultiTask) lives in
novbts.operator.param2field.
"""
import math

import torch
from torch import nn


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
