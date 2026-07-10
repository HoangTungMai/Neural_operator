"""Stage-visible tactile texture overlay for Isaac Sim.

The fixed ``[0.34, 0.74]`` grayscale-to-uint8 mapping uses the observed VBTS
range without per-frame normalization, so the marker contrast cannot flicker.
PNG rows are flipped because PNG starts top-left whereas USD ``st`` starts
bottom-left.  The quad mesh is in local ``z=0``: because Gf uses row vectors,
its explicitly constructed rotation matrix has ``[right; up; normal]`` as rows,
with the third row therefore mapping the mesh normal toward the camera.  A complete dynamic
omni.ui provider is preferred when available; otherwise unique file paths are
required because RTX caches asset paths.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from novbts.sim.grasp_output import flog


class TactileViewportOverlay:
    """Two non-emissive textured USD quads, best-effort and disabled by default."""

    _MAP_MIN = 0.34
    _MAP_MAX = 0.74

    def __init__(self, enabled: bool, res: int = 160, scale_m: float = 0.12,
                 anchor: str = "robot", update_every: int = 1,
                 output_dir: str | Path | None = None,
                 camera_eye=(1.20, -0.70, 0.70), camera_target=(0.45, 0.22, 0.36)) -> None:
        self.enabled = bool(enabled)
        self.res = int(res)
        self.scale_m = float(scale_m)
        self.anchor = anchor
        self.update_every = max(1, int(update_every))
        self.output_dir = Path(output_dir or "/work/runs/sim_grasp")
        self.camera_eye = np.asarray(camera_eye, dtype=float)
        self.camera_target = np.asarray(camera_target, dtype=float)
        self._textures = {}
        self._dynamic_providers = {}
        self._backend = "file"
        self._count = 0
        self._setup = False
        self._dynamic_probe_logged = False

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
        """Return a fixed no-contact marker field until the first live VBTS frame.

        The values deliberately stay inside the fixed ``[0.34, 0.74]`` map:
        0.70 is the gel background and 0.42 is a regular dark marker.  This
        prevents a transient black placeholder from obscuring the baseline
        pattern in the pre-grasp world-camera capture.
        """
        row, col = np.indices((res, res))
        spacing = max(8, int(round(res / 14.0)))
        radius = max(1.0, 0.14 * spacing)
        marker = (
            (np.remainder(row, spacing) - 0.5 * spacing) ** 2
            + (np.remainder(col, spacing) - 0.5 * spacing) ** 2
            <= radius ** 2
        )
        return np.where(marker, 0.42, 0.70).astype(np.float32)

    @staticmethod
    def _png(img: np.ndarray, path: Path) -> None:
        from PIL import Image
        gray = TactileViewportOverlay._gray_u8(img)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.flipud(rgb), mode="RGB").save(path)

    @staticmethod
    def _provider_details(provider) -> tuple[str | None, bool, list[str]]:
        """Return the URI and byte-update contract exposed by an actual provider."""
        uri = None
        get_uri = getattr(provider, "get_uri", None)
        if callable(get_uri):
            try:
                uri = get_uri()
            except Exception as exc:
                uri = f"<get_uri {type(exc).__name__}: {exc}>"
        names = [
            name for name in dir(provider)
            if any(token in name.lower() for token in ("texture", "image", "provider", "uri", "byte", "data"))
        ]
        return uri if isinstance(uri, str) else None, callable(getattr(provider, "set_bytes_data", None)), names

    def _probe_dynamic_texture_api(self) -> None:
        """Log the real in-container dynamic-texture surface once, never guessing it."""
        if self._dynamic_probe_logged:
            return
        self._dynamic_probe_logged = True
        terms = ("texture", "image", "provider")

        try:
            import omni.ui as ui

            ui_names = [name for name in dir(ui) if any(term in name.lower() for term in terms)]
            flog(f"TACTILE_OVERLAY_DYNAMIC_PROBE omni.ui present=True dir={ui_names}")
            for name in ("ByteImageProvider", "DynamicTextureProvider"):
                provider_type = getattr(ui, name, None)
                flog(
                    "TACTILE_OVERLAY_DYNAMIC_PROBE "
                    f"omni.ui.{name} present={provider_type is not None}"
                )
                if provider_type is None:
                    continue
                constructed = False
                for ctor_args in (("tactile_probe",), ()):
                    try:
                        provider = provider_type(*ctor_args)
                        uri, can_set_bytes, provider_names = self._provider_details(provider)
                        flog(
                            "TACTILE_OVERLAY_DYNAMIC_PROBE "
                            f"omni.ui.{name} ctor_args={ctor_args!r} uri={uri!r} "
                            f"set_bytes_data={can_set_bytes} dir={provider_names}"
                        )
                        constructed = True
                        break
                    except Exception as exc:
                        flog(
                            "TACTILE_OVERLAY_DYNAMIC_PROBE "
                            f"omni.ui.{name} ctor_args={ctor_args!r} "
                            f"{type(exc).__name__}: {exc}"
                        )
                if not constructed:
                    flog(f"TACTILE_OVERLAY_DYNAMIC_PROBE omni.ui.{name} constructible=False")
        except Exception as exc:
            flog(f"TACTILE_OVERLAY_DYNAMIC_PROBE omni.ui present=False {type(exc).__name__}: {exc}")

        try:
            import omni.kit.app as kit_app

            kit_names = [name for name in dir(kit_app) if any(term in name.lower() for term in terms)]
            app = kit_app.get_app()
            ext_names = []
            manager = app.get_extension_manager() if app is not None else None
            if manager is not None and hasattr(manager, "get_enabled_extension_ids"):
                ext_names = [
                    name for name in manager.get_enabled_extension_ids()
                    if any(term in name.lower() for term in terms)
                ]
            flog(
                "TACTILE_OVERLAY_DYNAMIC_PROBE "
                f"omni.kit.app present=True app={app is not None} dir={kit_names} "
                f"dynamic_extensions={ext_names}"
            )
        except Exception as exc:
            flog(f"TACTILE_OVERLAY_DYNAMIC_PROBE omni.kit.app present=False {type(exc).__name__}: {exc}")

        try:
            import carb

            carb_names = [name for name in dir(carb) if any(term in name.lower() for term in terms)]
            flog(f"TACTILE_OVERLAY_DYNAMIC_PROBE carb present=True dir={carb_names}")
        except Exception as exc:
            flog(f"TACTILE_OVERLAY_DYNAMIC_PROBE carb present=False {type(exc).__name__}: {exc}")

    def _try_dynamic_backend(self):
        """Use dynamic:// only if this image exposes the complete provider API."""
        self._probe_dynamic_texture_api()
        try:
            import omni.ui as ui
            byte_provider = getattr(ui, "ByteImageProvider", None)
            if byte_provider is None:
                return None
            provider_type = getattr(ui, "DynamicTextureProvider", None)
            if provider_type is None:
                return None
            providers = {}
            for side in ("Left", "Right"):
                provider = provider_type(f"tactile_{side.lower()}")
                uri = getattr(provider, "get_uri", lambda: None)()
                if (not isinstance(uri, str) or not uri.startswith("dynamic://")
                        or not hasattr(provider, "set_bytes_data")):
                    return None
                providers[side] = provider
            return providers
        except Exception:
            return None

    def setup(self, robot_center_w, robot_diag: float = 0.0) -> None:
        if not self.enabled or self._setup:
            return
        try:
            import omni.usd
            from pxr import Gf, Sdf, UsdGeom, UsdShade

            self.output_dir.mkdir(parents=True, exist_ok=True)
            stage = omni.usd.get_context().get_stage()
            dynamic = self._try_dynamic_backend()
            self._backend = "dynamic" if dynamic is not None else "file"
            self._dynamic_providers = dynamic or {}
            root = UsdGeom.Xform.Define(stage, "/World/TactileOverlay")
            root.GetPrim().CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("inherited")
            center = np.asarray(robot_center_w if robot_center_w is not None else self.camera_target, dtype=float)
            if self.anchor == "world":
                center = self.camera_target.copy()
            base = 0.55 * self.camera_target + 0.45 * center + 0.16 * (self.camera_eye - self.camera_target)
            base[2] += 0.13
            normal = self.camera_eye - base
            normal /= np.linalg.norm(normal)
            world_up = np.asarray([0.0, 0.0, 1.0])
            right = np.cross(world_up, normal)
            if np.linalg.norm(right) < 1.0e-6:
                right = np.asarray([1.0, 0.0, 0.0])
            else:
                right /= np.linalg.norm(right)
            up = np.cross(normal, right)
            up /= np.linalg.norm(up)

            # Keep the cards in the upper-right clear space.  The nominal 0.9-diagonal
            # upward move is capped by the world camera's narrow vertical field of view;
            # the lateral robot-diagonal offset keeps both cards clear of the gripper.
            span = float(robot_diag) if np.isfinite(robot_diag) and robot_diag > 1.0e-6 else 0.665
            up_offset = min(0.90 * span, max(0.055, 0.60 * self.scale_m))
            right_offset = min(0.58 * span, 0.42)
            base += up * up_offset + right * right_offset

            # Gf uses row vectors (v' = v * M).  The local mesh lies in z=0, so
            # rows [right; up; normal] map its horizontal, vertical, and +Z
            # normal axes into world space respectively.
            frame = Gf.Matrix3d(
                float(right[0]), float(right[1]), float(right[2]),
                float(up[0]), float(up[1]), float(up[2]),
                float(normal[0]), float(normal[1]), float(normal[2]),
            )
            frame_rotation = frame.ExtractRotation()
            transformed_normal = frame_rotation.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
            normal_error = (transformed_normal - Gf.Vec3d(*normal)).GetLength()
            assert normal_error < 1.0e-6, (
                f"tactile overlay frame normal mismatch: {normal_error:.3e}"
            )
            rotation = frame_rotation.GetQuat()
            rot_qf = Gf.Quatf(float(rotation.GetReal()), Gf.Vec3f(*rotation.GetImaginary()))
            for side, sign in (("Left", -1.0), ("Right", 1.0)):
                path = f"/World/TactileOverlay/{side}"
                mesh = UsdGeom.Mesh.Define(stage, path)
                half = self.scale_m / 2.0
                mesh.CreatePointsAttr([(-half, -half, 0), (half, -half, 0), (half, half, 0), (-half, half, 0)])
                mesh.CreateFaceVertexCountsAttr([4])
                mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
                mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying).Set(
                    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                )
                mesh.CreateDoubleSidedAttr(True)
                xf = UsdGeom.Xformable(mesh)
                xf.AddTranslateOp().Set(Gf.Vec3d(*(base + sign * right * (self.scale_m * 0.58))))
                xf.AddOrientOp().Set(rot_qf)
                material = UsdShade.Material.Define(stage, path + "/Material")
                surface = UsdShade.Shader.Define(stage, path + "/Material/Surface")
                surface.CreateIdAttr("UsdPreviewSurface")
                surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
                tex = UsdShade.Shader.Define(stage, path + "/Material/TactileTexture")
                tex.CreateIdAttr("UsdUVTexture")
                tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
                tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
                tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                reader = UsdShade.Shader.Define(stage, path + "/Material/StReader")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
                surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
                material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI(mesh).Bind(material)
                self._textures[side] = tex
                if self._backend == "dynamic":
                    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(self._dynamic_providers[side].get_uri()))
                else:
                    initial = self.output_dir / f"tactile_overlay_{side.lower()}_00000.png"
                    self._png(self._neutral_marker_image(self.res), initial)
                    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(initial)))
            self._setup = True
            flog(
                f"TACTILE_OVERLAY_SETUP anchor={self.anchor} scale_m={self.scale_m:.3f} "
                f"path=/World/TactileOverlay center=({base[0]:.3f},{base[1]:.3f},{base[2]:.3f}) "
                f"offset_up_m={up_offset:.3f} offset_right_m={right_offset:.3f}"
            )
            flog(f"TACTILE_OVERLAY_BACKEND {self._backend}")
        except Exception as exc:
            self.enabled = False
            flog(f"TACTILE_OVERLAY_FAIL setup {type(exc).__name__}: {exc}")

    def update(self, img_l: np.ndarray, img_r: np.ndarray) -> None:
        if not self.enabled or not self._setup:
            return
        self._count += 1
        if self._count % self.update_every:
            return
        try:
            from pxr import Sdf
            for side, image in (("Left", img_l), ("Right", img_r)):
                if self._backend == "dynamic":
                    gray = self._gray_u8(image)
                    rgba = np.empty((*gray.shape, 4), dtype=np.uint8)
                    rgba[..., :3] = gray[..., None]
                    rgba[..., 3] = 255
                    self._dynamic_providers[side].set_bytes_data(rgba.tobytes(), [self.res, self.res])
                else:
                    path = self.output_dir / f"tactile_overlay_{side.lower()}_{self._count:05d}.png"
                    self._png(image, path)
                    self._textures[side].GetInput("file").Set(Sdf.AssetPath(str(path)))
                    stale = self.output_dir / f"tactile_overlay_{side.lower()}_{self._count - 32:05d}.png"
                    if stale.exists():
                        stale.unlink()
        except Exception as exc:
            self.enabled = False
            flog(f"TACTILE_OVERLAY_FAIL update {type(exc).__name__}: {exc}")

    def teardown(self) -> None:
        return
