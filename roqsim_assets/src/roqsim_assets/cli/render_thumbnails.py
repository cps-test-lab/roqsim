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
walker blueprints) and every baked ``roqsim.worlds`` scene. Textures already ship their own colour
map, and built-in code-built worlds have no on-disk home, so neither is rendered here.

Usage::

    roqsim assets render-thumbnails

Rendering is best-effort per item: a model that will not compile standalone is skipped with a note,
never aborting the run.
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from roqsim.mesh_preview import build_mesh_scene as _build_mesh_scene
from roqsim.render import reset_to_home
from roqsim.rendering import FrameRenderer

_SIZE = 480  # square source PNG; the doc pages display it at ~150px.


def _render(model: mujoco.MjModel, cam: mujoco.MjvCamera, out: Path) -> None:
    data = mujoco.MjData(model)
    # `home` over qpos0, via the same helper `roqsim render` uses -- so a model's thumbnail and its
    # `roqsim render` output are the same picture. (Why it matters: for an articulated robot the two poses
    # are very different, and the TIAGo Pro's arms stick straight out in front of it at qpos0.)
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


def _render_mjcf(xml_path: Path, out: Path, apply=None) -> None:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    if apply is not None:
        apply(spec)
    model = spec.compile()
    _render(model, _framed_cam(model), out)


def _ground_and_light(spec: mujoco.MjSpec) -> None:
    """Give a bare model MJCF a checker ground (named ``floor``) + a light so it renders lit and
    grounded. Some robot models reference a world-provided ``floor`` in a contact pair and won't
    compile standalone without it. No-ops when the model already defines a ``floor``/its own light."""
    if not any(g.name == "floor" for g in spec.geoms):
        tex = spec.add_texture()
        tex.name = "ss_grid"
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        tex.width = tex.height = 512
        tex.rgb1 = [0.2, 0.3, 0.4]
        tex.rgb2 = [0.1, 0.15, 0.2]
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
        light.ambient = [0.4, 0.4, 0.4]
        light.diffuse = [0.9, 0.9, 0.9]


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
        asset = M.resolve_model(f"{name}:{stem}")

        def _apply(spec, a=asset):
            M.apply_assets(spec, a)
            _ground_and_light(spec)

        dest = thumb_path(asset.path)
        return dest, (lambda a=asset, ap=_apply: _render_mjcf(a.path, dest, apply=ap))

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
