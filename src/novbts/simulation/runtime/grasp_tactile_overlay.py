"""Screen-fixed live VBTS HUD for Isaac Sim.

The tactile images live in an ``omni.ui.Window`` rather than in the USD stage,
so camera movement cannot change their position or pixel size.  Providers and
widgets are deliberately created once in :meth:`setup`; live updates only push
new RGBA bytes into those existing providers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from novbts.simulation.runtime.grasp_output import flog


class TactileViewportOverlay:
    """Best-effort, screen-fixed left/right tactile HUD.

    ``scale_m`` and ``anchor`` remain accepted only for command-line backwards
    compatibility.  A HUD has no world-space scale or anchor.
    """

    _MAP_MIN = 0.34
    _MAP_MAX = 0.74
    _TITLE = "VBTS Tactile"
    _IMAGE_WIDTH_PX = 200
    _IMAGE_HEIGHT_PX = 200
    _IMAGE_GAP_PX = 8

    def __init__(
        self,
        enabled: bool,
        res: int = 160,
        scale_m: float = 0.12,
        anchor: str = "robot",
        update_every: int = 1,
        output_dir: str | Path | None = None,
        camera_eye=(1.20, -0.70, 0.70),
        camera_target=(0.45, 0.22, 0.36),
    ) -> None:
        # Keep the legacy constructor inputs so old callers keep working.  They
        # have no HUD effect and must never cause image files to be written.
        del output_dir, camera_eye, camera_target
        self.enabled = bool(enabled)
        self.res = max(1, int(res))
        self._legacy_scale_m = float(scale_m)
        self._legacy_anchor = str(anchor)
        self.update_every = max(1, int(update_every))
        self._providers: dict[str, object] = {}
        self._widgets: dict[str, object] = {}
        self._window = None
        self._count = 0
        self._setup = False
        self._legacy_warning_logged = False

    @classmethod
    def _gray_u8(cls, img: np.ndarray) -> np.ndarray:
        """Map the fixed observed tactile range ``[0.34, 0.74]`` to uint8."""
        gray = np.clip(
            (np.asarray(img, dtype=np.float32) - cls._MAP_MIN) / (cls._MAP_MAX - cls._MAP_MIN),
            0.0,
            1.0,
        )
        return (gray * 255.0 + 0.5).astype(np.uint8)

    @staticmethod
    def _neutral_marker_image(res: int) -> np.ndarray:
        """Return the fixed no-contact marker field used before the first frame."""
        row, col = np.indices((res, res))
        spacing = max(8, int(round(res / 14.0)))
        radius = max(1.0, 0.14 * spacing)
        marker = (
            (np.remainder(row, spacing) - 0.5 * spacing) ** 2
            + (np.remainder(col, spacing) - 0.5 * spacing) ** 2
            <= radius ** 2
        )
        return np.where(marker, 0.42, 0.70).astype(np.float32)

    @classmethod
    def _rgba_u8(cls, img: np.ndarray) -> np.ndarray:
        """Expand a fixed-map grayscale image for ByteImageProvider RGBA8."""
        gray = cls._gray_u8(img)
        if gray.ndim == 3 and gray.shape[-1] == 1:
            gray = gray[..., 0]
        if gray.ndim != 2:
            raise ValueError(f"expected a 2-D tactile image, got shape={gray.shape}")
        height, width = gray.shape
        return np.dstack((gray, gray, gray, np.full((height, width, 1), 255, dtype=np.uint8)))

    @staticmethod
    def _set_provider_bytes(provider, rgba: np.ndarray) -> None:
        """Use the ByteImageProvider contract: width first, then height."""
        height, width = rgba.shape[:2]
        provider.set_bytes_data(rgba.flatten().data, [width, height])

    def _fail(self, phase: str, reason: str) -> None:
        self.enabled = False
        self._setup = False
        flog(f"TACTILE_OVERLAY_FAIL {phase} {reason}")

    def _log_legacy_flags_ignored(self) -> None:
        if self._legacy_warning_logged:
            return
        self._legacy_warning_logged = True
        flog(
            "TACTILE_OVERLAY_WARN legacy --tactile-overlay-scale and "
            "--tactile-overlay-anchor are ignored for the screen-fixed HUD "
            f"(received scale_m={self._legacy_scale_m:.3f} anchor={self._legacy_anchor!r})"
        )

    def _confirm_no_world_prim(self) -> bool:
        """Prove this HUD did not create the legacy stage overlay path."""
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RuntimeError("no active USD stage")
            legacy_path = "/World/" + "TactileOverlay"
            legacy_prim = stage.GetPrimAtPath(legacy_path)
            if legacy_prim and legacy_prim.IsValid():
                self._fail("setup", "unexpected legacy overlay prim exists")
                return False
            flog("TACTILE_OVERLAY_NO_WORLD_PRIM ok")
            return True
        except Exception as exc:
            # Not being able to run the check says nothing about the HUD's health,
            # so warn instead of tearing down a window that is already working.
            flog(f"TACTILE_OVERLAY_WARN stage_check {type(exc).__name__}: {exc}")
            return False

    def setup(self, robot_center_w=None, robot_diag: float = 0.0) -> None:
        """Create the window, providers, and image widgets exactly once."""
        del robot_center_w, robot_diag
        if not self.enabled or self._setup:
            return
        try:
            import omni.ui
        except Exception as exc:
            self._fail("setup", f"omni.ui unavailable {type(exc).__name__}: {exc}")
            return

        if not hasattr(omni.ui, "Window"):
            self._fail("setup", "omni.ui.Window unavailable")
            return
        if not hasattr(omni.ui, "ImageWithProvider"):
            self._fail("setup", "omni.ui.ImageWithProvider unavailable")
            return
        if not hasattr(omni.ui, "ByteImageProvider"):
            self._fail("setup", "omni.ui.ByteImageProvider unavailable")
            return
        if not hasattr(omni.ui, "HStack"):
            self._fail("setup", "omni.ui.HStack unavailable")
            return

        try:
            ui = omni.ui
            providers = {
                "Left": ui.ByteImageProvider(),
                "Right": ui.ByteImageProvider(),
            }
            initial_rgba = self._rgba_u8(self._neutral_marker_image(self.res))
            for provider in providers.values():
                self._set_provider_bytes(provider, initial_rgba)

            window_width = 2 * self._IMAGE_WIDTH_PX + self._IMAGE_GAP_PX
            window = ui.Window(
                self._TITLE,
                width=window_width,
                height=self._IMAGE_HEIGHT_PX,
            )
            with window.frame:
                with ui.HStack(spacing=self._IMAGE_GAP_PX):
                    widgets = {
                        "Left": ui.ImageWithProvider(
                            providers["Left"],
                            width=self._IMAGE_WIDTH_PX,
                            height=self._IMAGE_HEIGHT_PX,
                        ),
                        "Right": ui.ImageWithProvider(
                            providers["Right"],
                            width=self._IMAGE_WIDTH_PX,
                            height=self._IMAGE_HEIGHT_PX,
                        ),
                    }
            self._providers = providers
            self._widgets = widgets
            self._window = window
            self._setup = True
            self._log_legacy_flags_ignored()
            flog(
                f'TACTILE_OVERLAY_HUD_CREATED window="{self._TITLE}" '
                f"w={self._IMAGE_WIDTH_PX} h={self._IMAGE_HEIGHT_PX}"
            )
            self._confirm_no_world_prim()
        except Exception as exc:
            self._fail("setup", f"{type(exc).__name__}: {exc}")

    def update(self, img_l: np.ndarray, img_r: np.ndarray) -> None:
        """Push a live frame to existing providers without rebuilding any UI."""
        if not self.enabled or not self._setup:
            return
        self._count += 1
        if self._count % self.update_every:
            return
        try:
            for side, image in (("Left", img_l), ("Right", img_r)):
                rgba = self._rgba_u8(image)
                self._set_provider_bytes(self._providers[side], rgba)
        except Exception as exc:
            self._fail("update", f"{type(exc).__name__}: {exc}")

    def teardown(self) -> None:
        """Best-effort cleanup for callers that keep an Isaac session alive."""
        window = self._window
        self._window = None
        self._widgets = {}
        self._providers = {}
        self._setup = False
        if window is None:
            return
        try:
            destroy = getattr(window, "destroy", None)
            if callable(destroy):
                destroy()
        except Exception as exc:
            flog(f"TACTILE_OVERLAY_FAIL teardown {type(exc).__name__}: {exc}")
