# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Render each model's preview thumbnail once, beside the model itself (committed with it).

Offscreen MuJoCo rendering is too expensive (and GL/GPU-dependent) to run on every ``make doc``, so
previews are generated here deliberately -- ``make thumbnails`` -- and written as
``<model-dir>/<name>.thumb.png`` next to each MJCF (so the thumbnail travels with the model, not in a
separate docs folder). The ``roqsim-models`` / ``roqsim-worlds`` doc directives reference that
co-located file by relative path and fall back to text when it is absent, so the docs build needs no
GL.

Covers every ``roqsim.models`` entry (flat ``<model>.xml``, nested ``<name>/<name>.xml`` props,
walker blueprints) and every baked ``roqsim.worlds`` scene. A robot whose manifest mounts devices (a
``spawn_sensor`` lidar or camera) is rendered as ``spawn_robot`` builds it, since its own MJCF does not
carry them (:func:`model_scene`). Textures already ship their own colour map, and built-in code-built
worlds have no on-disk home, so neither is rendered here.

Usage::

    roqsim assets render-thumbnails

Rendering is best-effort per item: a model that will not compile standalone is skipped with a note,
never aborting the run.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
from importlib import import_module
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image

from roqsim.mesh_preview import build_mesh_scene as _build_mesh_scene
from roqsim.render import (
    PREVIEW_HEADLIGHT_AMBIENT,
    PREVIEW_HEADLIGHT_DIFFUSE,
    PREVIEW_LIGHT_AMBIENT,
    PREVIEW_LIGHT_DIFFUSE,
    reset_to_home,
)
from roqsim.rendering import FrameRenderer
from roqsim.world import FLOOR_RGB1, FLOOR_RGB2, SKY_RGB1, SKY_RGB2

_SIZE = 480  # square source PNG; the doc pages display it at ~150px.


def _render(
    model: mujoco.MjModel, cam: mujoco.MjvCamera, out: Path, data: mujoco.MjData | None = None
) -> None:
    """Render ``model`` to ``out``; ``data`` is an already-posed state, else one reset to ``home``."""
    if data is None:
        data = mujoco.MjData(model)
        reset_to_home(model, data)
    fr = FrameRenderer(model, _SIZE, _SIZE, camera=cam)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fr.render(data)).save(out)
    fr.close()


def _framed_cam(model: mujoco.MjModel) -> mujoco.MjvCamera:
    """A 3/4 free camera auto-framed on the whole model."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.azimuth, cam.elevation = 45, -20
    cam.distance *= 1.25
    return cam


def _render_mjcf(xml_path: Path, out: Path) -> None:
    model = mujoco.MjSpec.from_file(str(xml_path)).compile()
    _render(model, _framed_cam(model), out)


def _ground_and_light(spec: mujoco.MjSpec) -> None:
    """Give a bare model MJCF the room's checker ground (named ``floor``) + a light so it renders lit
    and grounded. Some robot models reference a world-provided ``floor`` in a contact pair and won't
    compile standalone without it. No-ops when the model already defines a ``floor``/its own light."""
    if not any(g.name == "floor" for g in spec.geoms):
        sky = spec.add_texture()
        sky.name = "ss_sky"
        sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
        sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        sky.width = sky.height = 512
        sky.rgb1 = SKY_RGB1
        sky.rgb2 = SKY_RGB2
        tex = spec.add_texture()
        tex.name = "ss_grid"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        tex.width = tex.height = 512
        tex.rgb1 = FLOOR_RGB1
        tex.rgb2 = FLOOR_RGB2
        mat = spec.add_material()
        mat.name = "ss_ground"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "ss_grid"
        mat.texrepeat = [20, 20]
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [5, 5, 0.05]
        floor.material = "ss_ground"
    if not list(spec.lights):
        light = spec.worldbody.add_light()
        light.pos = [1.5, -1.5, 3]
        light.dir = [-1, 1, -2]
        # The same split `roqsim render` uses for a model preview: this light only casts the contact
        # shadow, and the shadow-free headlight does the modelling. Carrying both here would let the
        # shadow map comb the terminator into a ragged white/grey border on a matte white robot --
        # see `roqsim.render.fill_preview_self_shadows` for why no MuJoCo shadow knob fixes it.
        light.ambient = [PREVIEW_LIGHT_AMBIENT] * 3
        light.diffuse = [PREVIEW_LIGHT_DIFFUSE] * 3
        spec.visual.headlight.diffuse = [PREVIEW_HEADLIGHT_DIFFUSE] * 3
        spec.visual.headlight.ambient = [PREVIEW_HEADLIGHT_AMBIENT] * 3


def mounts_devices(model_file: Path) -> bool:
    """Whether the model's manifest mounts a device: an entry whose plugin registers an entity."""
    from roqsim.config import parse_plugin_entry
    from roqsim.manifest import load_manifest
    from roqsim.registry import resolve_plugin

    return any(
        resolve_plugin(parse_plugin_entry(entry, "manifest plugin").ref).provides_entity
        for entry in load_manifest(model_file)
    )


@contextlib.contextmanager
def model_scene(ref: str):
    """The scene a model's thumbnail shows, compiled and posed: yields ``(model, data)``.

    A robot whose manifest mounts devices is built as ``spawn_robot`` builds it, through the same path
    as ``roqsim render`` (:func:`roqsim.render.build_target`: transport plugins dropped, then reset),
    so its devices are attached and it stands as the spawn and its controllers leave it. Any other
    model is its own MJCF at ``home``. Both stand on the ground and under the light
    :func:`_ground_and_light` gives.
    """
    from roqsim import models as M
    from roqsim.render import build_target

    asset = M.resolve_model(ref)
    if not mounts_devices(asset.path):
        spec = mujoco.MjSpec.from_file(str(asset.path))
        M.apply_assets(spec, asset)
        _ground_and_light(spec)
        model = spec.compile()
        data = mujoco.MjData(model)
        # `home` over qpos0, via the same helper `roqsim render` uses -- so a model's thumbnail and its
        # `roqsim render` output are the same picture. (Why it matters: for an articulated robot the
        # two poses are very different, and the TIAGo Pro's arms stick straight out in front of it at
        # qpos0.)
        reset_to_home(model, data)
        yield model, data
        return

    ground = mujoco.MjSpec()
    _ground_and_light(ground)
    with tempfile.TemporaryDirectory(prefix="roqsim-thumb-") as tmp:
        world = Path(tmp) / "ground.xml"
        world.write_text(ground.to_xml())
        doc = Path(tmp) / "robot.yaml"
        doc.write_text(
            yaml.safe_dump(
                {"sim": {"world": str(world)}, "components": [{"spawn_robot": {"model": ref}}]}
            )
        )
        model, data, ctx, _view, _cam = build_target(str(doc), None)
    try:
        yield model, data
    finally:
        ctx.engine.shutdown()


def _render_model(ref: str, out: Path) -> None:
    with model_scene(ref) as (model, data):
        _render(model, _framed_cam(model), out, data)


def _render_mesh(obj_path: Path, out: Path) -> None:
    """Render a bare mesh (walker blueprint / prop OBJ) framed on it, via the shared preview scene."""
    model = _build_mesh_scene(str(obj_path))
    mid = model.mesh("prop").id
    vadr, vnum = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = model.mesh_vert[vadr : vadr + vnum]
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = ((lo + hi) / 2).tolist()
    cam.distance = float(np.linalg.norm(hi - lo)) * 1.6 + 0.5
    cam.azimuth, cam.elevation = 45, -20
    _render(model, cam, out)


def thumb_path(model_file: Path) -> Path:
    """The co-located thumbnail beside a model/world MJCF: ``<dir>/<stem>.thumb.png``."""
    return model_file.parent / f"{model_file.stem}.thumb.png"


def _iter_models():
    from roqsim import models as M

    def _render_for(name, stem):
        ref = f"{name}:{stem}"
        dest = thumb_path(M.resolve_model(ref).path)
        return dest, (lambda: _render_model(ref, dest))

    for name, models_dir, _mesh, _tex in M.providers():
        models_dir = Path(models_dir)
        for xml in sorted(models_dir.glob("*.xml")):  # flat <name>.xml
            yield _render_for(name, xml.stem)
        for sub in sorted(p for p in models_dir.iterdir() if p.is_dir()):  # nested props
            if (sub / f"{sub.name}.xml").is_file():
                yield _render_for(name, sub.name)
        people = models_dir / "people"
        if people.is_dir():
            for bp in sorted(p for p in people.iterdir() if p.is_dir()):
                objs = sorted(bp.glob("*.obj"))
                if objs:
                    dest = bp / f"{bp.name}.thumb.png"
                    yield dest, (lambda o=objs[0], d=dest: _render_mesh(o, d))


def _iter_worlds():
    from roqsim import world as W

    # Built-in world definitions (e.g. empty_room) are code-built with no on-disk home, so they have
    # no co-located thumbnail (the catalog falls back to text for them).
    for ep in W._world_entry_points():
        module = import_module(ep.value.split(":")[0] if isinstance(ep.value, str) else ep.value)
        worlds_dir = Path(module.WORLDS_DIR)
        names: set[str] = {
            p.name for p in worlds_dir.iterdir() if p.is_dir() and (p / f"{p.name}.xml").is_file()
        }
        names |= {p.stem for p in worlds_dir.glob("*.xml")}
        for wname in sorted(names):
            path = W.world_file(f"{ep.name}:{wname}", base_dir=worlds_dir)
            if path:
                dest = thumb_path(Path(path))
                yield dest, (lambda p=Path(path), d=dest: _render_mjcf(p, d))


def main(argv: list | None = None) -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args(argv)

    targets = list(_iter_models()) + list(_iter_worlds())
    ok = 0
    for dest, render in targets:
        try:
            render()
            print(f"  rendered {dest}")
            ok += 1
        except Exception as exc:  # noqa: BLE001 - best-effort per item
            print(f"  skipped  {dest} ({type(exc).__name__}: {exc})", file=sys.stderr)
    print(f"{ok}/{len(targets)} thumbnails written (beside each model)")


if __name__ == "__main__":
    main()
