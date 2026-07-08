#!/usr/bin/env python3
"""Host-side VBTS tactile renderer driven by penetration maps + frozen FNO."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from novbts.operator.field2field import DEV, fieldinput_from_contact_profile
from novbts.paths import RUNS, ensure
from novbts.sensor.markercam import (
    PinholeCamera,
    deformed_marker_xyz,
    render_dots,
    sample_field_to_markers,
    sensor_marker_grid_pixel_even,
)
from novbts.sensor.realism import add_camera_noise
from novbts.sim.fno_export import DEFAULT_OUT as DEFAULT_CKPT, load_checkpoint
from novbts.sim.gel_contact import gel_grid, pen_sphere, pose_to_T


class VBTSSensor:
    """Frozen FNO + marker camera for one gel pad."""

    def __init__(
        self,
        ckpt: str | Path = DEFAULT_CKPT,
        *,
        device: str | torch.device | None = None,
        px: int = 160,
        sensor_side: int = 11,
        working_dist: float = 0.05,
        marker_half_m: float = 0.009,
        sigma: float = 1.35,
        background: float = 0.72,
        contrast: float = 0.58,
    ) -> None:
        self.device = torch.device(device) if device is not None else DEV
        self.model, self.norm, self.meta = load_checkpoint(ckpt, self.device)
        self.side = int(self.meta.get("side", 32))
        self.coords = gel_grid(self.side, marker_half_m).astype(np.float32)
        self.coords_t = torch.tensor(self.coords, device=self.device)
        self.cam = PinholeCamera.from_gel(marker_half_m, px=px, working_dist=working_dist, fill=0.85)
        self.marker_coords = sensor_marker_grid_pixel_even(self.cam, sensor_side=sensor_side, pixel_fill=0.75)
        self.marker_t = torch.tensor(self.marker_coords, device=self.device)
        self.marker_count = self.marker_coords.shape[0]
        self.px = int(px)
        self.sigma = float(sigma)
        self.render_kw = {
            "background": float(background),
            "contrast": float(contrast),
            "polarity": "dark",
            "saturate": True,
        }
        self.pix_rest = self.cam.project(
            deformed_marker_xyz(
                self.marker_t,
                torch.zeros(1, self.marker_count, 3, device=self.device),
            )
        )
        self.rest_img = render_dots(self.pix_rest, self.px, self.px, self.sigma, **self.render_kw)

    @torch.no_grad()
    def render(
        self,
        pen: np.ndarray,
        sx: float,
        sy: float,
        *,
        mu: float = 0.6,
        E: float = 1.0e5,
        noise: bool = True,
        photons: float = 300.0,
        read_noise: float = 0.02,
    ) -> dict:
        pen = np.asarray(pen, dtype=np.float32)
        if pen.shape != (self.side, self.side):
            raise ValueError(f"pen must be {(self.side, self.side)}, got {pen.shape}")
        saturated = bool(np.max(pen) >= 0.00075)
        pen_clipped = np.clip(pen, 0.0, 0.00075).astype(np.float32)
        if float(pen_clipped.max()) <= 0.0:
            img = self.rest_img.clone().clamp(0.0, 1.0)
            if noise:
                img = add_camera_noise(img, photons=photons, read_noise=read_noise)
            return {
                "img": img,
                "flow": torch.zeros(1, self.marker_count, 2, device=self.device),
                "field": torch.zeros(1, 3, self.side, self.side, device=self.device),
                "saturated": saturated,
            }

        params = np.array(
            [[0.0, 0.0, float(pen_clipped.max()), 0.004, float(sx), float(sy),
              float(mu), float(E), 0.0]],
            dtype=np.float32,
        )
        inp_np, scal_np = fieldinput_from_contact_profile(params, pen_clipped[None], self.side)
        inp = torch.tensor(inp_np, device=self.device)
        scal = torch.tensor(scal_np, device=self.device)
        im, istd, om, ostd, sm, sstd = self.norm
        field = self.model((inp - im) / istd, (scal - sm) / sstd) * ostd + om
        markers = sample_field_to_markers(field, self.coords_t, self.marker_t)
        pix = self.cam.project(deformed_marker_xyz(self.marker_t, markers))
        flow = pix - self.pix_rest
        img = render_dots(pix, self.px, self.px, self.sigma, **self.render_kw)
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        if noise:
            img = add_camera_noise(img, photons=photons, read_noise=read_noise)
        return {"img": img, "flow": flow, "field": field, "saturated": saturated}


def _save_gray_png(img: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(2.0, 2.0), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    ax.set_axis_off()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _save_strip(frames: list[np.ndarray], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(1.7 * n, 1.7), squeeze=False)
    for ax, img in zip(axes[0], frames):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_demo(args) -> dict:
    out_dir = Path(args.out)
    ensure(out_dir)
    sensor = VBTSSensor(args.ckpt, px=args.px, sensor_side=args.sensor_side)
    grid = gel_grid(sensor.side, 0.009)
    frames, flows, pens, shears, timings = [], [], [], [], []
    R = 0.004
    depths = np.r_[np.linspace(0.0, 0.00055, 8), np.full(8, 0.00055)]
    shear_x = np.r_[np.zeros(8), np.linspace(0.0, 0.00045, 8)]
    shear_y = np.r_[np.zeros(8), np.linspace(0.0, -0.00025, 8)]
    for i, (depth, sx, sy) in enumerate(zip(depths, shear_x, shear_y)):
        T = pose_to_T((0.0, 0.0, R - float(depth)), (1.0, 0.0, 0.0, 0.0))
        pen = pen_sphere(T, R, grid, clip=0.00075)
        t0 = time.perf_counter()
        out = sensor.render(pen, float(sx), float(sy), mu=args.mu, E=args.E, noise=not args.no_noise)
        timings.append(time.perf_counter() - t0)
        img = out["img"][0, 0].detach().cpu().numpy()
        frames.append(img)
        flows.append(out["flow"][0].detach().cpu().numpy())
        pens.append(pen)
        shears.append([float(sx), float(sy)])
        _save_gray_png(img, out_dir / f"frame_{i:04d}.png")

    strip = out_dir / "press_then_shear_strip.png"
    _save_strip([frames[i] for i in np.linspace(0, len(frames) - 1, 8).round().astype(int)], strip)
    npz = out_dir / "demo_sequence.npz"
    np.savez_compressed(
        npz,
        images=np.stack(frames).astype(np.float32),
        flow=np.stack(flows).astype(np.float32),
        pen=np.stack(pens).astype(np.float32),
        shear=np.asarray(shears, dtype=np.float32),
        timings_s=np.asarray(timings, dtype=np.float32),
        strip=np.array(str(strip)),
    )
    report = {
        "out_dir": str(out_dir),
        "strip": str(strip),
        "npz": str(npz),
        "frames": len(frames),
        "mean_render_ms": float(np.mean(timings) * 1e3),
        "max_flow_px": float(np.linalg.norm(np.stack(flows), axis=-1).max()),
    }
    (out_dir / "demo_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=str(RUNS / "sim_grasp" / "host_demo"))
    ap.add_argument("--px", type=int, default=160)
    ap.add_argument("--sensor-side", type=int, default=11)
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--E", type=float, default=1.0e5)
    ap.add_argument("--no-noise", action="store_true")
    args = ap.parse_args()
    if args.demo:
        rep = run_demo(args)
        print(json.dumps(rep, indent=2))
        print("VBTS_SENSOR_DEMO_OK", rep["strip"])
    else:
        sensor = VBTSSensor(args.ckpt, px=args.px, sensor_side=args.sensor_side)
        print(f"loaded VBTSSensor side={sensor.side} px={sensor.px} markers={sensor.marker_count}")


if __name__ == "__main__":
    main()
