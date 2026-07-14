"""Progress-file logging and image dumps for the grasp demo driver.

Isaac swallows stdout inside the container, so all progress/markers are
appended to a host-visible progress file (PROGRESS). Success gating in
`infra/run_sim_grasp.sh` greps this file — keep marker strings stable.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

PROGRESS = "/work/sim_grasp_demo_progress.txt"


def flog(msg: str) -> None:
    Path(PROGRESS).parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
        f.flush()


def save_png(img: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.0, 2.0), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
    ax.set_axis_off()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_rgb_png(img: np.ndarray, path: Path) -> None:
    arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255)
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = arr.astype(np.uint8)
    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.imsave(path, arr)
