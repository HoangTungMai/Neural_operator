from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_image(path: Path):
    import matplotlib.image as mpimg

    return mpimg.imread(path)


def _blank_image(shape=(160, 160, 3)):
    return np.ones(shape, dtype=np.float32)


def _world_frame_by_label(run_dir: Path, world_frames):
    out = {}
    for frame in world_frames or []:
        label = frame.get("label")
        path = frame.get("path")
        if not label or not path:
            continue
        p = Path(path)
        if not p.exists() and str(p).startswith("/work/"):
            repo_root = run_dir.parents[2] if len(run_dir.parents) >= 3 else Path.cwd()
            p = repo_root / str(p)[6:]
        if p.exists():
            out[label] = p
    for p in sorted(run_dir.glob("world_main_*_*.png")):
        stem = p.stem
        for label in ("pre_grasp", "close_clamp", "lift_peak"):
            if f"world_main_{label}_" in stem:
                out.setdefault(label, p)
    return out


def _phase_steps(phases, desired: str):
    if phases.size == 0:
        return []
    matches = np.where(phases == desired)[0]
    return matches.tolist()


def _tactile_index_for_phase(phases, tactile_every: int, files, label: str):
    if not files:
        return None
    if phases.size == 0:
        fallback = {"pre_grasp": 0, "close_clamp": len(files) // 2, "lift_peak": len(files) - 1}
        return int(min(max(fallback[label], 0), len(files) - 1))
    if label == "pre_grasp":
        candidates = _phase_steps(phases, "HOME")
        target_step = candidates[-1] if candidates else 0
    elif label == "close_clamp":
        candidates = _phase_steps(phases, "HOLD")
        if not candidates:
            candidates = _phase_steps(phases, "CLOSE")
        target_step = candidates[0] if candidates else int(0.55 * max(len(phases) - 1, 0))
    else:
        candidates = _phase_steps(phases, "LIFT")
        target_step = candidates[-1] if candidates else len(phases) - 1
    sample_steps = list(range(0, len(phases), max(1, int(tactile_every))))
    if not sample_steps:
        return 0
    sample_steps = sample_steps[: len(files)]
    return int(min(range(len(sample_steps)), key=lambda i: abs(sample_steps[i] - target_step)))


def make_grasp_montage(run_dir, world_frames=None, tactile_every=2, out_name="montage.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    seq_path = run_dir / "sequence.npz"
    phases = np.asarray([], dtype=str)
    if seq_path.exists():
        with np.load(seq_path, allow_pickle=False) as z:
            if "phase" in z:
                phases = z["phase"].astype(str)
    world = _world_frame_by_label(run_dir, world_frames)
    tactile_l = sorted(run_dir.glob("tactile_L_*.png"))
    tactile_r = sorted(run_dir.glob("tactile_R_*.png"))
    labels = [
        ("pre_grasp", "pre-grasp"),
        ("close_clamp", "close / clamp"),
        ("lift_peak", "lift peak"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.0), squeeze=False)
    for col, (label, title) in enumerate(labels):
        world_path = world.get(label)
        world_img = _read_image(world_path) if world_path and world_path.exists() else _blank_image((360, 480, 3))
        idx_l = _tactile_index_for_phase(phases, tactile_every, tactile_l, label)
        idx_r = _tactile_index_for_phase(phases, tactile_every, tactile_r, label)
        left_img = _read_image(tactile_l[idx_l]) if idx_l is not None else _blank_image()
        right_img = _read_image(tactile_r[idx_r]) if idx_r is not None else _blank_image()
        cells = [
            ("world camera", world_img),
            ("tactile L", left_img),
            ("tactile R", right_img),
        ]
        for row, (row_title, img) in enumerate(cells):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="none")
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle("UR3e + Robotiq grasp: world RGB and gel tactile frames", fontsize=12)
    fig.tight_layout()
    out_path = run_dir / out_name
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {
        "path": str(out_path),
        "world_frames": {k: str(v) for k, v in world.items()},
        "tactile_L_count": len(tactile_l),
        "tactile_R_count": len(tactile_r),
    }
