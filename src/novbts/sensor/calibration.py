#!/usr/bin/env python3
"""Phase 6b -- sensor CALIBRATION schema (the hardware-dependent half).

This is the bridge from the SIM marker-dot sensor to a PHYSICAL DIY build. It does NOT
fit anything (no real sensor exists yet); it only pins down, in one serialisable place,
the parameters that must match between sim and hardware so a future sim2real fit has a
target. `SensorCalib.from_camera(...)` snapshots the current SIM configuration;
`SensorCalib.template(...)` writes a blank config with TODO placeholders for the physical
measurements (camera intrinsics from a checkerboard, gel/marker geometry from the build).

When hardware exists, the fit is: measure the physical quantities, fill the template, and
adjust the sim `PinholeCamera`/marker grid so the rendered rest image matches a captured
rest frame (minimising marker-position residual) -- a thin optimisation left as a stub here.

  python -m novbts.sensor.calibration --template            # write a blank physical-build config
  python -m novbts.sensor.calibration --from-sim            # snapshot the current sim sensor
"""
import argparse
import json
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from novbts.sensor.markercam import (
    PinholeCamera, deformed_marker_xyz, sensor_marker_grid_pixel_even, marker_half_extent,
)
from novbts.paths import RUNS, FEM, ensure


@dataclass
class SensorCalib:
    """Everything that must agree between the sim sensor and a physical marker-dot build.

    `None` fields are physical measurements to be filled when hardware exists (the sim
    snapshot fills the optical/geometry block; the noise block is fitted from a captured
    dark/flat frame). Lengths in metres, angles in pixels unless noted.
    """
    # --- camera intrinsics (sim: auto-fit; hardware: checkerboard calibration) ---
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    px_w: Optional[int] = None
    px_h: Optional[int] = None
    working_dist_m: Optional[float] = None       # camera-to-membrane distance
    # --- gel + marker geometry (hardware: measured from the printed dot gel) ---
    gel_half_extent_m: Optional[float] = None
    marker_side: Optional[int] = None            # markers per row of the visible grid
    marker_pitch_m: Optional[float] = None
    gel_thickness_m: Optional[float] = None
    gel_youngs_E_pa: Optional[float] = None
    # --- render / appearance (sim: render kwargs; hardware: from captured frames) ---
    dot_sigma_px: Optional[float] = None
    background: Optional[float] = None
    contrast: Optional[float] = None
    polarity: str = "dark"
    # --- camera noise model (hardware: fit from dark+flat frames; see realism.py) ---
    photons: Optional[float] = None
    read_noise: Optional[float] = None
    blur_px: Optional[float] = None
    # --- bookkeeping ---
    source: str = "template"                     # "sim_snapshot" | "physical_build"
    notes: str = ""
    todo: list = field(default_factory=list)

    @classmethod
    def from_camera(cls, cam: PinholeCamera, *, gel_half_extent_m, marker_side, marker_pitch_m,
                    dot_sigma_px, background, contrast, polarity="dark",
                    photons=300.0, read_noise=0.02, blur_px=0.6,
                    gel_thickness_m=None, gel_youngs_E_pa=None, notes=""):
        """Snapshot the current SIM sensor into a fully-populated calib (the sim 'ground truth')."""
        return cls(
            fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, px_w=cam.px_w, px_h=cam.px_h,
            working_dist_m=cam.working_dist, gel_half_extent_m=float(gel_half_extent_m),
            marker_side=int(marker_side), marker_pitch_m=float(marker_pitch_m),
            gel_thickness_m=gel_thickness_m, gel_youngs_E_pa=gel_youngs_E_pa,
            dot_sigma_px=float(dot_sigma_px), background=float(background), contrast=float(contrast),
            polarity=polarity, photons=float(photons), read_noise=float(read_noise), blur_px=float(blur_px),
            source="sim_snapshot", notes=notes)

    @classmethod
    def template(cls):
        """Blank physical-build config: every measurable field None, with a TODO checklist."""
        return cls(source="physical_build", notes="fill from the physical DIY build", todo=[
            "camera intrinsics fx,fy,cx,cy,px_w,px_h: OpenCV checkerboard calibration",
            "working_dist_m: measured camera-to-membrane distance",
            "gel_half_extent_m, marker_side, marker_pitch_m: measured from the printed dot gel",
            "gel_thickness_m, gel_youngs_E_pa: gel datasheet / indentation test",
            "dot_sigma_px, background, contrast, polarity: from a captured rest frame",
            "photons, read_noise, blur_px: fit from dark + flat frames (Poisson-Gaussian; see realism.py)",
        ])

    def to_camera(self) -> PinholeCamera:
        """Reconstruct a PinholeCamera (requires the intrinsic block to be filled)."""
        if any(v is None for v in (self.fx, self.fy, self.cx, self.cy, self.px_w, self.px_h, self.working_dist_m)):
            raise ValueError("intrinsics incomplete -- calibrate the camera first")
        return PinholeCamera(self.fx, self.fy, self.cx, self.cy, self.px_w, self.px_h, self.working_dist_m)

    def missing(self):
        """Physical fields still unmeasured (for hardware configs)."""
        opt = ["fx", "fy", "cx", "cy", "px_w", "px_h", "working_dist_m", "gel_half_extent_m",
               "marker_side", "marker_pitch_m", "dot_sigma_px", "background", "contrast",
               "photons", "read_noise", "blur_px"]
        return [k for k in opt if getattr(self, k) is None]

    def save(self, path):
        ensure(path.parent)
        json.dump(asdict(self), open(path, "w"), indent=2, default=float)
        return path

    @classmethod
    def load(cls, path):
        return cls(**json.load(open(path)))


def fit_to_rest_frame(calib: SensorCalib, rest_frame_img):  # pragma: no cover - hardware stub
    """STUB (needs hardware): refine the sim camera/marker grid so the rendered rest image
    matches a captured physical rest frame, by minimising the marker-position residual.
    Implemented when a real sensor and a captured rest frame exist."""
    raise NotImplementedError(
        "sim2real fit requires a captured physical rest frame; fill SensorCalib.template() first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true", help="write a blank physical-build config")
    ap.add_argument("--from-sim", action="store_true", help="snapshot the current sim sensor config")
    ap.add_argument("--data", default=str(FEM / "traj_mix" / "fem_gt_shear.npz"),
                    help="dataset providing the gel/marker geometry for the sim snapshot")
    ap.add_argument("--px", type=int, default=160)
    ap.add_argument("--sensor-marker-side", type=int, default=11)
    ap.add_argument("--marker-pixel-fill", type=float, default=0.75)
    ap.add_argument("--working-dist", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.35)
    ap.add_argument("--background", type=float, default=0.72)
    ap.add_argument("--contrast", type=float, default=0.58)
    args = ap.parse_args()

    out_dir = RUNS / "phase6"; ensure(out_dir)
    if not args.template and not args.from_sim:
        args.from_sim = True  # default: snapshot the sim

    if args.template:
        c = SensorCalib.template()
        p = c.save(out_dir / "calib_template.json")
        print(f"wrote blank physical-build template -> {p}")
        print("TODO to fill from hardware:")
        for t in c.todo:
            print(f"  - {t}")

    if args.from_sim:
        import torch
        z = np.load(args.data, allow_pickle=True)
        coords = z["coords"]
        cam = PinholeCamera.from_gel(marker_half_extent(coords), px=args.px, working_dist=args.working_dist)
        sc = sensor_marker_grid_pixel_even(cam, args.sensor_marker_side, pixel_fill=args.marker_pixel_fill)
        st = torch.tensor(sc)
        pix = cam.project(deformed_marker_xyz(st, torch.zeros(1, sc.shape[0], 3)))[0]
        d = torch.cdist(pix, pix).fill_diagonal_(float("inf"))
        pitch_px = float(d.min(dim=1).values.median())
        # px pitch -> metres via the rest grid spacing
        gxy = np.sort(np.unique(sc[:, 0]))
        pitch_m = float(np.mean(np.diff(gxy))) if len(gxy) > 1 else float("nan")
        c = SensorCalib.from_camera(
            cam, gel_half_extent_m=marker_half_extent(coords), marker_side=args.sensor_marker_side,
            marker_pitch_m=pitch_m, dot_sigma_px=args.sigma, background=args.background,
            contrast=args.contrast, notes=f"sim snapshot from {args.data}; rest pitch={pitch_px:.2f}px")
        p = c.save(out_dir / "calib_sim.json")
        print(f"wrote sim sensor snapshot -> {p}")
        print(f"  intrinsics fx={cam.fx:.1f} cx={cam.cx:.1f}  working_dist={cam.working_dist}m  "
              f"marker pitch={pitch_px:.2f}px ({pitch_m*1e3:.2f}mm)")
        print(f"  missing (none for a sim snapshot): {c.missing()}")


if __name__ == "__main__":
    main()
