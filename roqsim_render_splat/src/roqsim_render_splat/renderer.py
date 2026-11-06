# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""The Gaussian-splat renderer: a captured room as every camera sees it, the robots drawn into it.

Registered as ``splat`` (``roqsim.renderers``); a world chooses it with ``sim.renderer: splat`` and
places its capture in ``sim.renderers.splat.splat`` (``file``, ``pose``, ``scale``).

- **Static world**: the capture, rasterised by gsplat (Apache-2.0) on a CUDA GPU, the cameras of a
  batch that share size and clip planes in one call, split into passes under ``max_pass_mib``. The
  Gaussians stay in the capture's frame; each camera is mapped into it by the inverse of the splat's
  pose and scale, so no spherical harmonic is ever rotated.
- **Moving bodies**: drawn by MuJoCo's own renderer, opened in this renderer with the same scene
  and asked for the moving bodies alone (``CameraSpec.bodies = "moving"``), from the same snapshot
  on the same thread, and composited per pixel by depth: the nearer wins.
- **Depth**: the splat's expected depth (gsplat ``ED``) along the view axis, in metres; a pixel
  whose Gaussians cover less than ``min_alpha`` reads ``inf``, no return.

What it cannot give it refuses: no segmentation (a capture has no objects), no orthographic camera,
no GPU. It never falls back to MuJoCo's renderer for the room.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from roqsim.pose import PoseError, parse_pose
from roqsim.render_api import (
    CameraRequest,
    Frame,
    Renderer,
    RendererCapabilities,
    RenderError,
    SceneDescription,
    Setting,
    StateSnapshot,
    unknown_keys,
)
from roqsim.world import resolve_file_ref

from .ply import SplatFileError, read_header, read_splat

#: The keys of the ``splat`` block: the capture and where it sits.
SPLAT_KEYS = ("file", "pose", "scale")

#: GPU bytes one camera of a pass takes per Gaussian, and per pixel, as measured on a captured room
#: (gsplat 1.5, packed): what splits a batch into passes under ``max_pass_mib``.
_BYTES_PER_GAUSSIAN_CAMERA = 72
_BYTES_PER_PIXEL_CAMERA = 64


def _matrix(quat) -> np.ndarray:
    """The rotation matrix of a ``w, x, y, z`` quaternion."""
    w, x, y, z = (float(c) for c in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


#: MuJoCo's camera frame (looking down -z, +y up) to the OpenCV frame gsplat projects in
#: (looking down +z, +y down).
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


def splat_placement(block: Mapping) -> tuple[np.ndarray, np.ndarray, float]:
    """``(R, t, s)`` of a ``splat`` block: a capture point ``p`` sits at ``t + s * R @ p``."""
    pos, quat = parse_pose(block.get("pose") or {}, relative=True)
    return _matrix(quat), np.asarray(pos, dtype=float), float(block.get("scale", 1.0))


def view_matrix(cam_pos, cam_quat, placement) -> np.ndarray:
    """The 4x4 world-to-camera matrix gsplat takes, in the capture's frame and units.

    *cam_pos*/*cam_quat* are a :class:`CameraRequest`'s world pose (MuJoCo's camera frame).
    """
    rot, trans, scale = placement
    r_cam = rot.T @ _matrix(cam_quat) @ _MJ_TO_CV  # camera axes in the capture's frame
    p_cam = rot.T @ (np.asarray(cam_pos, dtype=float) - trans) / scale
    view = np.eye(4)
    view[:3, :3] = r_cam.T
    view[:3, 3] = -r_cam.T @ p_cam
    return view


def intrinsics_matrix(request: CameraRequest) -> np.ndarray:
    """The 3x3 projection gsplat takes for *request*.

    roqsim counts ``cx, cy`` from the top-left pixel's centre; gsplat from its corner, hence the
    half pixel.
    """
    spec = request.camera
    if spec.intrinsics is not None:
        fx, fy, cx, cy = spec.intrinsics
    else:
        fy = 0.5 * spec.height / math.tan(math.radians(request.fovy) / 2.0)
        fx, cx, cy = fy, (spec.width - 1) / 2.0, (spec.height - 1) / 2.0
    return np.array([[fx, 0.0, cx + 0.5], [0.0, fy, cy + 0.5], [0.0, 0.0, 1.0]])


class SplatRenderer(Renderer):
    """A captured room as a 3D Gaussian splat (gsplat, CUDA), the moving bodies drawn into it by depth.

    The splat is the static world; the world's own static geometry -- what the robot collides
    with -- is not drawn. Needs a CUDA GPU.
    """

    capabilities = RendererCapabilities(
        images=frozenset({"rgb", "depth"}),
        camera_models=frozenset({"pinhole", "free"}),
        composes_moving_bodies=True,
        draws_static=False,
        batched=True,
        deterministic=True,
        in_process=True,
        decorations=False,
        geom_groups=False,
        principal_point=True,
        non_square_pixels=True,
        moving_only=False,
    )
    SETTINGS = {
        "splat": Setting(
            None,
            "the capture (required): {file: a 3DGS .ply beside the world or <package>:<path>, "
            "pose: where it sits in the world as any pose is written (position in m, orientation "
            "as roll/pitch/yaw in rad or a quaternion), scale: metres per unit of the capture "
            "(default 1)}; `roqsim splat frame <ply>` proposes the pose",
        ),
        "sh_degree": Setting(
            None,
            "the spherical-harmonics degree drawn, 0-3 (view-dependent colour costs time); null: "
            "the file's own",
        ),
        "background": Setting(
            [0, 0, 0], "the colour behind the capture, where it has nothing, as RGB 0-255"
        ),
        "min_opacity": Setting(
            0.0,
            "Gaussians less opaque than this (0-1) are dropped at load: fewer to draw, a "
            "changed picture; 0 keeps every one",
        ),
        "min_alpha": Setting(
            0.5,
            "a pixel the capture covers less than this (0-1) reads depth inf (no return)",
        ),
        "device": Setting("cuda:0", "the CUDA device the capture is drawn on"),
        "max_pass_mib": Setting(
            4096,
            "the GPU memory one rasterisation pass may take, in MiB; a batch of more cameras is "
            "drawn in several passes, the same pictures",
        ),
        "open_timeout_s": Setting(
            600.0,
            "how long opening may take, in s: reading the capture and, on first use in an "
            "environment, compiling gsplat's CUDA kernels",
        ),
    }

    # -- configuration -------------------------------------------------------------------------

    @classmethod
    def validate_config(cls, settings, base_dir=None) -> list[str]:
        errors = super().validate_config(settings, base_dir)
        if not isinstance(settings, Mapping):
            return errors
        block = settings.get("splat")
        if block is None:
            errors.append(
                "'splat' is required: {file: room.ply, pose: {...}, scale: ...}; "
                "`roqsim splat frame room.ply` proposes it"
            )
        elif not isinstance(block, Mapping):
            errors.append(f"'splat' must be a mapping of {', '.join(SPLAT_KEYS)}, got {block!r}")
        else:
            errors += unknown_keys(block, SPLAT_KEYS, "splat")
            errors += cls._check_placement(block)
            errors += cls._check_file(block, base_dir)
        sh = settings.get("sh_degree")
        if sh is not None and (isinstance(sh, bool) or sh not in (0, 1, 2, 3)):
            errors.append(f"'sh_degree' is 0, 1, 2, 3 or null, got {sh!r}")
        bg = settings.get("background", [0, 0, 0])
        if not (
            isinstance(bg, list | tuple)
            and len(bg) == 3
            and all(
                isinstance(c, int | float) and not isinstance(c, bool) and 0 <= c <= 255 for c in bg
            )
        ):
            errors.append(f"'background' is three numbers 0-255 (R, G, B), got {bg!r}")
        for key in ("min_opacity", "min_alpha"):
            value = settings.get(key, cls.SETTINGS[key].default)
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
                errors.append(f"'{key}' is a number from 0 to 1, got {value!r}")
        for key in ("max_pass_mib", "open_timeout_s"):
            value = settings.get(key, cls.SETTINGS[key].default)
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                errors.append(f"'{key}' is a number > 0, got {value!r}")
        device = settings.get("device", "cuda:0")
        if not (isinstance(device, str) and (device == "cuda" or device.startswith("cuda:"))):
            errors.append(f"'device' is a CUDA device ('cuda' or 'cuda:N'), got {device!r}")
        return errors

    @staticmethod
    def _check_placement(block: Mapping) -> list[str]:
        errors = []
        pose = block.get("pose")
        if pose is not None:
            if not isinstance(pose, Mapping):
                return [f"'splat.pose' must be a mapping of position and orientation, got {pose!r}"]
            errors += unknown_keys(pose, ("position", "orientation"), "splat.pose")
            if not errors:
                try:
                    parse_pose(pose, relative=True)
                except PoseError as err:
                    errors.append(f"splat.{err}".replace("splat.'pose", "'splat.pose"))
        scale = block.get("scale", 1.0)
        if isinstance(scale, bool) or not isinstance(scale, int | float) or not scale > 0:
            errors.append(
                f"'splat.scale' is metres per unit of the capture, a number > 0, got {scale!r}"
            )
        return errors

    @staticmethod
    def _check_file(block: Mapping, base_dir) -> list[str]:
        ref = block.get("file")
        if not ref or not isinstance(ref, str):
            return [f"'splat.file' is required: the 3DGS .ply of the capture, got {ref!r}"]
        if base_dir is None and ":" not in ref and not Path(ref).is_absolute():
            return []  # a tool's own settings: resolved against the world when it opens
        try:
            read_header(resolve_file_ref(ref, base_dir, "splat file"))
        except FileNotFoundError as err:
            return [f"'splat.file': {err}"]
        except SplatFileError as err:
            return [f"'splat.file': {err}"]
        return []

    @classmethod
    def open_timeout_s(cls, settings: Mapping) -> float | None:
        return float((settings or {}).get("open_timeout_s", cls.SETTINGS["open_timeout_s"].default))

    # -- lifecycle -----------------------------------------------------------------------------

    def __init__(self, settings=None) -> None:
        super().__init__(settings)
        self._torch = None
        self._gsplat = None
        self._gaussians: dict | None = None
        self._placement = None
        self._mujoco = None
        self._moving = False
        #: What a test or a benchmark reads: Gaussians drawn, passes per batch.
        self.stats = {"gaussians": 0, "batches": 0, "passes": 0}

    def open(self, scene: SceneDescription) -> None:
        torch = self._import_torch()
        device = str(self.settings["device"])
        if not torch.cuda.is_available():
            raise RenderError(
                f"the splat renderer draws on a CUDA GPU ({device!r}), and torch sees none here; "
                "run on a machine with an NVIDIA GPU and a CUDA build of torch, or choose another "
                "renderer (sim.renderer)"
            )
        index = torch.device(device).index or 0
        if index >= torch.cuda.device_count():
            raise RenderError(
                f"the splat renderer was asked for {device!r}; this machine has "
                f"{torch.cuda.device_count()} CUDA device(s)"
            )
        self._device = torch.device(device)
        block = dict(self.settings["splat"] or {})
        try:
            path = resolve_file_ref(block.get("file", ""), scene.base_dir, "splat file")
            splat = read_splat(path)
        except (FileNotFoundError, SplatFileError) as err:
            raise RenderError(f"sim.renderers.splat.splat.file: {err}") from err
        self._placement = splat_placement(block)
        self._upload(splat)
        if scene.moving.any():
            from roqsim.rendering import MujocoRenderer  # noqa: PLC0415 -- GL only where needed

            self._mujoco = MujocoRenderer()
            cameras = [
                replace(c, bodies="moving", images=frozenset({"rgb", "depth"}))
                for c in scene.cameras
            ]
            self._mujoco.open(replace(scene, cameras=cameras))
        self._moving = self._mujoco is not None

    def _import_torch(self):
        if self._torch is None:
            try:
                import gsplat  # noqa: PLC0415 -- heavy; only a world that draws through it pays
                import torch  # noqa: PLC0415
            except ImportError as err:
                raise RenderError(
                    f"the splat renderer needs torch and gsplat: {err}. Install roqsim_render_splat "
                    "with its dependencies (pip install roqsim_render_splat)"
                ) from err
            self._torch, self._gsplat = torch, gsplat
        return self._torch

    def _upload(self, splat) -> None:
        torch = self._torch
        opacity = 1.0 / (1.0 + np.exp(-splat.opacity_logits.astype(np.float64)))
        keep = opacity >= float(self.settings["min_opacity"])
        degree = (
            splat.sh_degree
            if self.settings["sh_degree"] is None
            else int(self.settings["sh_degree"])
        )
        if degree > splat.sh_degree:
            raise RenderError(
                f"sim.renderers.splat.sh_degree is {degree}, and the capture holds spherical "
                f"harmonics of degree {splat.sh_degree} only"
            )
        if not keep.any():
            raise RenderError(
                f"sim.renderers.splat.min_opacity {self.settings['min_opacity']} drops every one of "
                f"the capture's {len(keep):,} Gaussians (the most opaque is {opacity.max():.4f})"
            )
        k = (degree + 1) ** 2
        dev = self._device
        try:
            quats = torch.from_numpy(splat.quats[keep]).to(dev)
            self._gaussians = {
                "means": torch.from_numpy(splat.means[keep]).to(dev),
                "quats": quats / quats.norm(dim=1, keepdim=True).clamp_min(1e-12),
                "scales": torch.from_numpy(splat.log_scales[keep]).to(dev).exp(),
                "opacities": torch.from_numpy(opacity[keep].astype(np.float32)).to(dev),
                "colors": torch.from_numpy(np.ascontiguousarray(splat.sh[keep, :k, :])).to(dev),
            }
        except torch.cuda.OutOfMemoryError as err:
            free, total = torch.cuda.mem_get_info(dev)
            raise RenderError(
                f"the capture's {int(keep.sum()):,} Gaussians do not fit on {dev} "
                f"({free / 2**20:,.0f} of {total / 2**20:,.0f} MiB free); drop faint ones with "
                "min_opacity, or draw on a larger GPU"
            ) from err
        self._sh_degree = degree
        self.stats["gaussians"] = int(keep.sum())

    def close(self) -> None:
        if self._mujoco is not None:
            self._mujoco.close()
            self._mujoco = None
        self._gaussians = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # -- drawing -------------------------------------------------------------------------------

    def render(self, snapshot: StateSnapshot, requests: Sequence[CameraRequest]) -> list[Frame]:
        if self._gaussians is None:
            raise RenderError("the splat renderer was used before open() or after close()")
        self.stats["batches"] += 1
        frames = [Frame(r.id, snapshot.sim_time, snapshot.step) for r in requests]
        depths: list[np.ndarray | None] = [None] * len(requests)
        # One rasterisation takes one size and one pair of clip planes, so cameras share a pass only
        # where both agree: a camera is drawn as it would be alone.
        by_pass: dict[tuple, list[int]] = {}
        for i, r in enumerate(requests):
            key = (r.camera.width, r.camera.height, r.camera.near, r.camera.far)
            by_pass.setdefault(key, []).append(i)
        bodies = self._draw_bodies(snapshot, requests) if self._moving else None
        for (width, height, _near, _far), indices in by_pass.items():
            for chunk in self._passes(indices, width, height):
                rgb, depth = self._rasterise(
                    [requests[i] for i in chunk],
                    width,
                    height,
                    None if bodies is None else [bodies[i] for i in chunk],
                )
                for j, i in enumerate(chunk):
                    frames[i].rgb, depths[i] = rgb[j], depth[j]
        for frame, request, depth in zip(frames, requests, depths, strict=True):
            frame.depth = depth if "depth" in request.images else None
            if "rgb" not in request.images:
                frame.rgb = None
        return frames

    def _passes(self, indices: list[int], width: int, height: int) -> list[list[int]]:
        """*indices* in passes whose estimated GPU memory stays under ``max_pass_mib``."""
        per_camera = (
            self.stats["gaussians"] * _BYTES_PER_GAUSSIAN_CAMERA
            + width * height * _BYTES_PER_PIXEL_CAMERA
        )
        budget = float(self.settings["max_pass_mib"]) * 2**20
        size = max(1, int(budget // max(per_camera, 1)))
        return [indices[i : i + size] for i in range(0, len(indices), size)]

    def _rasterise(self, requests: Sequence[CameraRequest], width: int, height: int, bodies=None):
        """The pass's colour and depth, as arrays: the capture rasterised on the GPU and, where
        *bodies* (MuJoCo's frames of the moving bodies) are given, composited with them there by
        depth, so each frame crosses to the CPU once."""
        torch, g = self._torch, self._gaussians
        dev = self._device
        scale = self._placement[2]
        views = np.stack([view_matrix(r.pos, r.quat, self._placement) for r in requests])
        ks = np.stack([intrinsics_matrix(r) for r in requests])
        near = min((r.camera.near or 0.01) for r in requests) / scale
        far_m = [r.camera.far for r in requests if r.camera.far]
        far = (max(far_m) / scale) if far_m else 1e10
        background = torch.tensor(
            [float(c) / 255.0 for c in self.settings["background"]], device=dev
        )
        self.stats["passes"] += 1
        try:
            with torch.no_grad():
                image, alpha, _meta = self._gsplat.rasterization(
                    g["means"],
                    g["quats"],
                    g["scales"],
                    g["opacities"],
                    g["colors"],
                    torch.from_numpy(views.astype(np.float32)).to(dev),
                    torch.from_numpy(ks.astype(np.float32)).to(dev),
                    width,
                    height,
                    near_plane=near,
                    far_plane=far,
                    sh_degree=self._sh_degree,
                    render_mode="RGB+ED",
                )
                # The background behind what the capture covers (gsplat's own `backgrounds` is
                # refused in its packed mode, so it is blended here, the same way).
                colour = image[..., :3] + (1.0 - alpha) * background
                rgb = (colour.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
                depth = image[..., 3] * scale
                depth = torch.where(
                    alpha[..., 0] >= float(self.settings["min_alpha"]), depth, torch.inf
                )
                if bodies is not None:
                    rgb, depth = self._compose(rgb, depth, bodies)
                rgb_np = rgb.cpu().numpy()
                depth_np = depth.to(torch.float32).cpu().numpy()
        except torch.cuda.OutOfMemoryError as err:
            free, total = torch.cuda.mem_get_info(dev)
            raise RenderError(
                f"{len(requests)} camera(s) of {width}x{height} over {self.stats['gaussians']:,} "
                f"Gaussians ran out of GPU memory on {dev} ({free / 2**20:,.0f} of "
                f"{total / 2**20:,.0f} MiB free); lower max_pass_mib"
            ) from err
        return rgb_np, depth_np

    def _draw_bodies(self, snapshot, requests) -> list[Frame]:
        """The moving bodies alone, drawn by MuJoCo's renderer from the same snapshot: colour, and
        depth that is ``inf`` wherever no moving body is."""
        moving = [
            replace(
                r, camera=replace(r.camera, bodies="moving"), images=frozenset({"rgb", "depth"})
            )
            for r in requests
        ]
        return self._mujoco.render(snapshot, moving)

    def _compose(self, rgb, depth, bodies):
        """*rgb*/*depth* (the capture, on the GPU) with *bodies* put in by depth, the nearer winning
        per pixel."""
        torch, dev = self._torch, self._device
        body_rgb = torch.from_numpy(np.ascontiguousarray(np.stack([b.rgb for b in bodies]))).to(dev)
        body_depth = torch.from_numpy(np.ascontiguousarray(np.stack([b.depth for b in bodies]))).to(
            dev
        )
        nearer = body_depth < depth
        return (
            torch.where(nearer[..., None], body_rgb, rgb),
            torch.where(nearer, body_depth, depth),
        )
