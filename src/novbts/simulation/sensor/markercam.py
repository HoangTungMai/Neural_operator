#!/usr/bin/env python3
"""Differentiable marker-dot sensor: pinhole projection + Gaussian-splat dot render.

The gel rest markers sit on a regular grid (`coords`, gel-frame xy, metres). Under
contact each marker moves by the surface displacement `disp=(ux,uy,uz)`. A camera
*below* the membrane looking up the +z axis sees the dots move; that image (and the
2D marker flow tracked from it) is the VBTS observation.

All ops are torch and differentiable in the marker positions -> differentiable in
`disp` -> differentiable in the FNO that predicts `disp`.  Pipeline:

    coords, disp  --deformed_marker_xyz-->  xyz[B,M,3]
                  --PinholeCamera.project->  pix[B,M,2]
                  --render_dots----------->  image[B,1,H,W]
"""
import numpy as np
import torch
import torch.nn.functional as F


class PinholeCamera:
    """Pinhole below the membrane, optical axis through the gel centre (0,0), looking
    up +z. Intrinsics auto-fit so the marker footprint fills `fill` of the frame, so
    dots are always visible regardless of working distance.

      pix_u = fx * x / depth + cx,   pix_v = fy * y / depth + cy
      depth = working_dist + uz      (membrane pressed toward the below-camera -> uz<0 -> magnify)
    """
    def __init__(self, fx, fy, cx, cy, px_w, px_h, working_dist):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.px_w, self.px_h, self.working_dist = px_w, px_h, working_dist

    @classmethod
    def from_gel(cls, marker_half_m, px=80, working_dist=0.05, fill=0.85):
        """marker_half_m: half-extent of the marker grid (m). Maps that to fill*px/2."""
        f = fill * (px / 2.0) * working_dist / float(marker_half_m)
        return cls(f, f, px / 2.0, px / 2.0, px, px, working_dist)

    def project(self, xyz):
        """xyz [..., 3] gel-frame -> pix [..., 2] (u=col, v=row)."""
        depth = self.working_dist + xyz[..., 2]
        depth = torch.clamp(depth, min=1e-4)
        u = self.fx * xyz[..., 0] / depth + self.cx
        v = self.fy * xyz[..., 1] / depth + self.cy
        return torch.stack([u, v], -1)

    def as_dict(self):
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "px_w": self.px_w, "px_h": self.px_h, "working_dist": self.working_dist}


def deformed_marker_xyz(coords, disp, z_rest=0.0):
    """coords [M,2] (rest xy), disp [B,M,3] (ux,uy,uz) -> [B,M,3] deformed gel-frame xyz.
    z_rest is the rest membrane height relative to the camera reference (kept 0: the
    camera `working_dist` already encodes the rest depth; uz is the perturbation)."""
    c = coords if torch.is_tensor(coords) else torch.tensor(coords)
    c = c.to(disp.device, disp.dtype)
    xy = c[None] + disp[..., :2]                       # [B,M,2]
    z = z_rest + disp[..., 2:3]                         # [B,M,1]
    return torch.cat([xy, z], -1)                       # [B,M,3]


def render_dots(
    pix,
    px_h,
    px_w,
    sigma=1.2,
    background=0.0,
    contrast=1.0,
    polarity="bright",
    saturate=False,
):
    """pix [B,M,2] (u,v) -> image [B,1,px_h,px_w]; differentiable Gaussian dots.

    By default this preserves the original bright-dot-on-black render. For a
    GelSight-like tracking-marker image, use polarity="dark", a gray background,
    and saturate=True so each dot behaves like a finite ink spot instead of an
    unbounded Gaussian sum.
    """
    dev, dt = pix.device, pix.dtype
    ys = torch.arange(px_h, device=dev, dtype=dt)
    xs = torch.arange(px_w, device=dev, dtype=dt)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")     # [H,W]
    du = gx[None, None] - pix[:, :, 0, None, None]      # [B,M,H,W]
    dv = gy[None, None] - pix[:, :, 1, None, None]
    g = torch.exp(-(du * du + dv * dv) / (2.0 * sigma * sigma))
    density = g.sum(1, keepdim=True)                    # [B,1,H,W]
    if saturate:
        density = 1.0 - torch.exp(-density)
    if polarity == "dark":
        return background - contrast * density
    return background + contrast * density


def field_to_markers(field):
    """FNO output field [B,3,H,W] -> markers [B,M,3] matching the row-major coords order
    (coords index = i*W + j), the same order params_to_fieldinput/marker_grid use."""
    b, c, h, w = field.shape
    return field.permute(0, 2, 3, 1).reshape(b, h * w, c)


def track_flow_known(pix_rest, pix_def):
    """Ground-truth pixel flow (correspondence known by construction): [B,M,2]."""
    return pix_def - pix_rest


def track_flow_image(img_def, pix_rest, win=3, dark=False):
    """Re-detect each marker from the rendered image by an intensity-weighted centroid in
    a (2*win+1) window around its REST pixel -> tracked deformed pixel [B,M,2]. Used only
    to validate the render+track round-trip (assumes flow < win)."""
    B, _, H, W = img_def.shape
    M = pix_rest.shape[1]
    dev, dt = img_def.device, img_def.dtype
    u0 = pix_rest[:, :, 0].round().long().clamp(win, W - 1 - win)   # [B,M]
    v0 = pix_rest[:, :, 1].round().long().clamp(win, H - 1 - win)
    offs = torch.arange(-win, win + 1, device=dev)
    oy, ox = torch.meshgrid(offs, offs, indexing="ij")              # [k,k]
    k = oy.numel()
    uu = (u0[..., None] + ox.reshape(-1)).clamp(0, W - 1)           # [B,M,k]
    vv = (v0[..., None] + oy.reshape(-1)).clamp(0, H - 1)
    bidx = torch.arange(B, device=dev)[:, None, None].expand(B, M, k)
    img = 1.0 - img_def if dark else img_def
    wgt = img[bidx, 0, vv, uu]                                      # [B,M,k]
    wsum = wgt.sum(-1, keepdim=True) + 1e-9
    cu = (wgt * uu.to(dt)).sum(-1, keepdim=True) / wsum
    cv = (wgt * vv.to(dt)).sum(-1, keepdim=True) / wsum
    return torch.cat([cu, cv], -1)                                  # [B,M,2]


def marker_half_extent(coords):
    """Half-extent (max |x|,|y|) of the marker grid in metres."""
    c = np.asarray(coords)
    return float(np.abs(c).max())


def sensor_marker_grid(coords, sensor_side=11, inset=0.06):
    """Uniform visible marker grid inside the dense FEM/FNO field footprint.

    `coords` describes the dense simulation grid. The returned points describe
    ink dots on the gel surface, so they can be fewer and slightly inset from
    the boundary like a physical tracking-marker gel.
    """
    c = np.asarray(coords)
    xmin, ymin = c.min(axis=0)
    xmax, ymax = c.max(axis=0)
    dx = (xmax - xmin) * float(inset)
    dy = (ymax - ymin) * float(inset)
    xs = np.linspace(xmin + dx, xmax - dx, sensor_side, dtype=np.float32)
    ys = np.linspace(ymin + dy, ymax - dy, sensor_side, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)


def sensor_marker_grid_pixel_even(cam, sensor_side=11, pixel_fill=0.75):
    """Visible marker grid whose REST projection is exactly uniform in pixels.

    Physical tracking markers are printed to look regular in the camera image.
    This helper chooses uniformly spaced rest pixel centers, then back-projects
    them to gel xy coordinates at z=0 so deformation can still be sampled from
    the dense FEM/FNO field.
    """
    half_u = float(pixel_fill) * cam.px_w / 2.0
    half_v = float(pixel_fill) * cam.px_h / 2.0
    u0, u1 = round(cam.cx - half_u), round(cam.cx + half_u)
    v0, v1 = round(cam.cy - half_v), round(cam.cy + half_v)
    us = np.linspace(u0, u1, sensor_side, dtype=np.float32)
    vs = np.linspace(v0, v1, sensor_side, dtype=np.float32)
    vv, uu = np.meshgrid(vs, us, indexing="ij")
    x = (uu - cam.cx) * cam.working_dist / cam.fx
    y = (vv - cam.cy) * cam.working_dist / cam.fy
    return np.stack([x.reshape(-1), y.reshape(-1)], axis=-1).astype(np.float32)


def sample_field_to_markers(field, field_coords, marker_coords):
    """Bilinearly sample a dense field [B,C,H,W] at marker_coords [M,2] -> [B,M,C]."""
    fc = torch.as_tensor(field_coords, device=field.device, dtype=field.dtype)
    mc = torch.as_tensor(marker_coords, device=field.device, dtype=field.dtype)
    xmin, ymin = fc.min(dim=0).values
    xmax, ymax = fc.max(dim=0).values
    x = 2.0 * (mc[:, 0] - xmin) / (xmax - xmin) - 1.0
    y = 2.0 * (mc[:, 1] - ymin) / (ymax - ymin) - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2).expand(field.shape[0], -1, -1, -1)
    sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(-1).permute(0, 2, 1)


def marker_subset_indices(coords, sensor_side=13):
    """Pick a sparser visible marker grid from a dense square FEM/FNO marker field.

    The physics field stays dense (e.g. 32x32), but real GelSight-style tracking
    markers are visually much sparser. This returns row-major indices that include
    the field boundary and are evenly spaced across the dense grid.
    """
    c = np.asarray(coords)
    total = c.shape[0]
    field_side = int(round(np.sqrt(total)))
    if sensor_side is None or sensor_side <= 0 or sensor_side >= field_side:
        return np.arange(total, dtype=np.int64)
    if field_side * field_side != total:
        raise ValueError(f"expected a square marker grid, got M={total}")
    lines = np.linspace(0, field_side - 1, sensor_side).round().astype(np.int64)
    lines = np.unique(lines)
    yy, xx = np.meshgrid(lines, lines, indexing="ij")
    return (yy.reshape(-1) * field_side + xx.reshape(-1)).astype(np.int64)
