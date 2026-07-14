"""Single-sensor ``omni.ui`` control panel for the standalone VBTS demo.

The module deliberately keeps Isaac imports lazy: host-mode gates can import
``sensor_demo`` without requiring an Isaac installation.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class SensorControlPanel:
    """A fixed-size tactile image plus x/y/z controls backed by one controller."""

    _MAP_MIN = 0.34
    _MAP_MAX = 0.74

    def __init__(self, controller, *, log: Callable[[str], None], res: int = 160) -> None:
        self.controller = controller
        self.log = log
        self.res = int(res)
        self.enabled = True
        self._syncing = False
        self._window = None
        self._provider = None
        self._models: dict[str, object] = {}
        self._label = None

    @classmethod
    def _rgba_u8(cls, image: np.ndarray) -> np.ndarray:
        gray = np.clip(
            (np.asarray(image, dtype=np.float32) - cls._MAP_MIN) / (cls._MAP_MAX - cls._MAP_MIN),
            0.0,
            1.0,
        )
        if gray.ndim != 2:
            raise ValueError(f"expected a 2-D tactile image, got {gray.shape}")
        gray_u8 = (gray * 255.0 + 0.5).astype(np.uint8)
        return np.dstack((gray_u8, gray_u8, gray_u8, np.full((*gray_u8.shape, 1), 255, np.uint8)))

    @staticmethod
    def _set_provider_bytes(provider, rgba: np.ndarray) -> None:
        height, width = rgba.shape[:2]
        provider.set_bytes_data(rgba.flatten().data, [width, height])

    def setup(self, initial_image: np.ndarray) -> bool:
        """Build widgets once.  Returns false rather than crashing on UI failures."""
        try:
            import omni.ui as ui

            required = ("Window", "ByteImageProvider", "ImageWithProvider", "FloatSlider", "Label", "VStack")
            missing = [name for name in required if not hasattr(ui, name)]
            if missing:
                raise RuntimeError(f"omni.ui missing {missing}")

            provider = ui.ByteImageProvider()
            self._set_provider_bytes(provider, self._rgba_u8(initial_image))
            window = ui.Window("VBTS Sensor", width=340, height=365)
            with window.frame:
                with ui.VStack(spacing=6):
                    ui.ImageWithProvider(provider, width=200, height=200)
                    label = ui.Label("peak=0.000 mm   shear=(0.000, 0.000) mm")
                    for axis, title, lo, hi in (
                        ("x", "x (m)", self.controller.x_min, self.controller.x_max),
                        ("y", "y (m)", self.controller.y_min, self.controller.y_max),
                        ("z", "z (m)", self.controller.z_min, self.controller.z_max),
                    ):
                        with ui.HStack(height=24):
                            ui.Label(title, width=48)
                            slider = ui.FloatSlider(min=float(lo), max=float(hi), step=0.0001)
                            model = slider.model
                            model.set_value(float(getattr(self.controller.state, axis)))
                            model.add_value_changed_fn(
                                lambda model, axis=axis: self._on_slider(axis, model.as_float)
                            )
                            self._models[axis] = model
            self._provider = provider
            self._window = window
            self._label = label
            self.log('SENSOR_UI_READY window="VBTS Sensor" w=200 h=200')
            return True
        except Exception as exc:
            self.enabled = False
            self.log(f"SENSOR_UI_FAIL setup {type(exc).__name__}: {exc}")
            return False

    def _on_slider(self, axis: str, value: float) -> None:
        if self.enabled and not self._syncing:
            self.controller.set_axis(axis, float(value))

    def sync_axis(self, axis: str, value: float) -> None:
        """Reflect controller changes without feeding them back into the controller."""
        model = self._models.get(axis)
        if model is None:
            return
        self._syncing = True
        try:
            model.set_value(float(value))
        finally:
            self._syncing = False

    def update(self, image: np.ndarray, pen_max: float, shear_xy) -> None:
        if not self.enabled or self._provider is None:
            return
        try:
            self._set_provider_bytes(self._provider, self._rgba_u8(image))
            if self._label is not None:
                sx, sy = np.asarray(shear_xy, dtype=float)
                self._label.text = (
                    f"peak={float(pen_max) * 1e3:.3f} mm   "
                    f"shear=({sx * 1e3:.3f}, {sy * 1e3:.3f}) mm"
                )
        except Exception as exc:
            self.enabled = False
            self.log(f"SENSOR_UI_FAIL update {type(exc).__name__}: {exc}")

    def teardown(self) -> None:
        window = self._window
        self._window = None
        self._models = {}
        self._provider = None
        self._label = None
        if window is not None:
            try:
                destroy = getattr(window, "destroy", None)
                if callable(destroy):
                    destroy()
            except Exception as exc:
                self.log(f"SENSOR_UI_FAIL teardown {type(exc).__name__}: {exc}")
