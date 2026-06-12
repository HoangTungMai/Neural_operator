import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class CostResult:
    model: str
    device: str
    torch_version: str
    cuda_version: str | None
    gpu_name: str | None
    total_vram_gb: float | None
    frames: int
    marker_side: int
    markers: int
    load_patches: int
    batch_size: int
    channels: int
    generate_seconds: float
    frames_per_second: float
    marker_vectors_per_second: float
    dataset_size_mb: float
    peak_vram_gb: float | None
    output_path: str | None


def marker_grid(side: int, device: torch.device) -> torch.Tensor:
    xs = torch.linspace(-1.0, 1.0, side, device=device)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def sample_loads(batch: int, patches: int, device: torch.device) -> torch.Tensor:
    # x, y, normal force, shear_x, shear_y, radius, poisson, young
    loads = torch.empty(batch, patches, 8, device=device).uniform_(0.0, 1.0)
    loads[..., 0:2] = loads[..., 0:2] * 1.6 - 0.8
    loads[..., 2] = loads[..., 2] * 1.5 + 0.1
    loads[..., 3:5] = loads[..., 3:5] * 0.8 - 0.4
    loads[..., 5] = loads[..., 5] * 0.18 + 0.03
    loads[..., 6] = loads[..., 6] * 0.15 + 0.35
    loads[..., 7] = loads[..., 7] * 4.0 + 0.5
    return loads


def elastic_halfspace_field(loads: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    # Physics proxy: linear elastic half-space Green's function with Gaussian load patches.
    # It is not a FEM/GIPC/MPM substitute, but it gives a physically grounded displacement field.
    x0 = loads[..., 0]
    y0 = loads[..., 1]
    fn = loads[..., 2]
    fx = loads[..., 3]
    fy = loads[..., 4]
    radius = loads[..., 5].clamp_min(0.02)
    nu = loads[..., 6]
    young = loads[..., 7]

    dx = coords[:, 0][None, None, :] - x0[..., None]
    dy = coords[:, 1][None, None, :] - y0[..., None]
    r2 = dx.square() + dy.square() + radius[..., None].square()
    r = torch.sqrt(r2)
    gaussian = torch.exp(-(dx.square() + dy.square()) / (2.0 * radius[..., None].square()))

    coeff_normal = (1.0 - nu.square())[..., None] / (torch.pi * young[..., None])
    coeff_tangent = (2.0 - 2.0 * nu)[..., None] / (torch.pi * young[..., None])

    uz = -coeff_normal * fn[..., None] * gaussian / r
    ux = coeff_tangent * fx[..., None] * gaussian / r
    uy = coeff_tangent * fy[..., None] * gaussian / r

    # Add radial coupling from normal load so normal indentation creates lateral marker motion.
    radial_coeff = 0.12 * coeff_normal * fn[..., None] * gaussian / (r2 + 1e-6)
    ux = ux + radial_coeff * dx
    uy = uy + radial_coeff * dy

    return torch.stack([ux.sum(dim=1), uy.sum(dim=1), uz.sum(dim=1)], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=20000)
    parser.add_argument("--marker-side", type=int, default=32)
    parser.add_argument("--load-patches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    coords = marker_grid(args.marker_side, device)
    all_params = []
    all_disp = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    remaining = args.frames
    while remaining > 0:
        batch = min(args.batch_size, remaining)
        loads = sample_loads(batch, args.load_patches, device)
        disp = elastic_halfspace_field(loads, coords)
        if args.save is not None:
            all_params.append(loads.detach().cpu().numpy().astype(np.float32))
            all_disp.append(disp.detach().cpu().numpy().astype(np.float32))
        remaining -= batch
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    output_path = None
    dataset_size_mb = args.frames * args.marker_side * args.marker_side * 3 * 4 / 1024**2
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save,
            loads=np.concatenate(all_params, axis=0),
            coords=coords.detach().cpu().numpy().astype(np.float32),
            disp=np.concatenate(all_disp, axis=0),
            model="elastic_halfspace_green_function_proxy",
        )
        output_path = str(args.save)

    result = CostResult(
        model="elastic_halfspace_green_function_proxy",
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
        marker_side=args.marker_side,
        markers=args.marker_side * args.marker_side,
        load_patches=args.load_patches,
        batch_size=args.batch_size,
        channels=3,
        generate_seconds=seconds,
        frames_per_second=args.frames / seconds,
        marker_vectors_per_second=args.frames * args.marker_side * args.marker_side / seconds,
        dataset_size_mb=dataset_size_mb,
        peak_vram_gb=(
            torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None
        ),
        output_path=output_path,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
