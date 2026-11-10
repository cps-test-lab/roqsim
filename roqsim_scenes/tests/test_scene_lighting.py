"""How scene-to-mjcf lights a room, placed and measured on a small one baked in the test.

The room is 6 x 5 m, 2.8 m high, away from the origin, with a ceiling slab and a table. Each picture
is rendered twice, as lit and with every light's ``castshadow`` off: their ratio is what the shadow
maps alone do to a pixel, and depth places the pixel in the room.

- No band: on the wall the room's centre faces, the ratio below 1 m is the ratio higher up, whichever
  way the walls' faces are wound. MuJoCo culls front faces when it draws a shadow map, so a wall
  wound inward (as an importer or a hand-written tool may write one) draws its own room-facing side
  into the map of a light that reaches it, and shadows itself in a band. A control bakes such a room
  under a single casting hemispherical lamp at its centre and finds the band there, so the measure
  is shown to see one; the bake's lights find none, since the one that casts never lights a wall.
- A contact shadow: the floor under the table is darker, relative to the open floor, with the shadows
  than without them.
"""

from __future__ import annotations

import json
import math

import roqsim  # noqa: F401, I001  -- selects the offscreen GL backend before mujoco binds one

import mujoco
import numpy as np
import pytest

from roqsim_scenes import scene_manifest
from roqsim_scenes import scene_mesh_io as mio
from roqsim_scenes.cli import scene_to_mjcf

X0, Y0 = 20.0, 10.0  # the room's corner: away from the origin, so a light at (0, 0) would show
W, D, H, T = 6.0, 5.0, 2.8, 0.15
TABLE = (X0 + 2.0, Y0 + 2.5)  # its centre
TOP = (1.0, 0.6, 0.72)  # its top: width, depth, height


def _box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    verts, faces = mio.box(" ".join(str(v) for v in hi - lo))
    return verts + (lo + hi) / 2, faces


def _room(tmp_path, ceiling=True, light=None, inward=False):
    """Write the room's scene.json and OBJs, and its scene.yaml when ``light`` is given; with
    ``inward`` its walls' faces are wound inward."""
    parts = {}
    for name, (lo, hi) in {
        "wall_S": ((X0 - T, Y0 - T), (X0 + W + T, Y0)),
        "wall_N": ((X0 - T, Y0 + D), (X0 + W + T, Y0 + D + T)),
        "wall_W": ((X0 - T, Y0), (X0, Y0 + D)),
        "wall_E": ((X0 + W, Y0), (X0 + W + T, Y0 + D)),
    }.items():
        verts, faces = _box((*lo, 0.0), (*hi, H))
        parts[name] = (verts, faces[:, ::-1] if inward else faces)
    if ceiling:
        parts["ceiling"] = _box((X0 - T, Y0 - T, H), (X0 + W + T, Y0 + D + T, H + 0.05))
    tx, ty = TABLE
    w, d, h = TOP
    parts["table_top"] = _box((tx - w / 2, ty - d / 2, h - 0.04), (tx + w / 2, ty + d / 2, h))
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        cx, cy = tx + sx * (w / 2 - 0.04), ty + sy * (d / 2 - 0.04)
        parts[f"table_leg{i}"] = _box((cx - 0.02, cy - 0.02, 0.0), (cx + 0.02, cy + 0.02, h - 0.04))
    (tmp_path / "meshes").mkdir()
    objects = []
    for name, (verts, faces) in parts.items():
        mio.write_obj(tmp_path / "meshes" / f"{name}.obj", verts, faces)
        objects.append({"name": name, "mesh": f"meshes/{name}.obj", "rgba": [0.85, 0.85, 0.82, 1]})
    allv = np.vstack([v for v, _ in parts.values()])
    manifest = scene_manifest.stamp(
        {
            "name": "lit_room",
            "bounds_min": allv.min(0).tolist(),
            "bounds_max": allv.max(0).tolist(),
            "ground_z": 0.0,
            "objects": objects,
        }
    )
    (tmp_path / "scene.json").write_text(json.dumps(manifest))
    if light is not None:
        (tmp_path / "scene.yaml").write_text(json.dumps({"light": light}))
    return tmp_path / "scene.json"


def _bake(tmp_path, **room):
    scene = _room(tmp_path, **room)
    out = tmp_path / "out" / "lit_room.xml"
    scene_to_mjcf.main(["--scene", str(scene), "--out", str(out)])
    return out


def _lights(model):
    return {model.light(i).name: i for i in range(model.nlight)}


# --- where the lights go ----------------------------------------------------------------------


def test_a_room_gets_a_casting_down_light_under_its_ceiling_and_a_shadowless_uplight(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(_bake(tmp_path)))
    lights = _lights(model)
    assert set(lights) == {"scene_down", "scene_up"}
    down, up = lights["scene_down"], lights["scene_up"]
    assert model.light_type[down] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    assert np.allclose(model.light_dir[down], [0, 0, -1])
    assert model.light_castshadow[down]
    assert H - 0.1 < model.light_pos[down][2] < H  # under the soffit, not above it
    assert np.allclose(model.light_pos[down][:2], [X0 + W / 2, Y0 + D / 2], atol=0.2)
    assert np.allclose(model.light_dir[up], [0, 0, 1])
    assert not model.light_castshadow[up]
    assert model.light_pos[up][2] < 1.0


def test_fixtures_true_leaves_the_room_to_its_fixtures_and_keeps_the_shadows(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(_bake(tmp_path, light={"fixtures": True})))
    assert set(_lights(model)) == {"scene_down"}
    assert model.light_castshadow[_lights(model)["scene_down"]]


def test_an_open_scene_is_lit_from_over_its_top(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(_bake(tmp_path, ceiling=False)))
    assert model.light_pos[_lights(model)["scene_down"]][2] > H


def test_the_fill_is_the_headlights_ambient(tmp_path):
    model = mujoco.MjModel.from_xml_path(str(_bake(tmp_path, light={"fill": [0.2, 0.2, 0.2]})))
    assert np.allclose(model.vis.headlight.ambient, [0.2, 0.2, 0.2])


# --- what the pictures show -------------------------------------------------------------------


class _Shots:
    """A view rendered as lit and without shadows, with every pixel's point in the room."""

    def __init__(self, model, pos, target, w=320, h=240, fovy=70.0):
        self.model, self.data = model, mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)
        model.vis.global_.offwidth, model.vis.global_.offheight = w, h
        self.r = mujoco.Renderer(model, height=h, width=w)
        self.w, self.h, self.fovy = w, h, fovy
        self.pos = np.asarray(pos, float)
        f = np.asarray(target, float) - self.pos
        self.fwd = f / np.linalg.norm(f)
        right = np.cross(self.fwd, [0, 0, 1.0])
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.fwd)

    def _render(self, depth=False):
        if depth:
            self.r.enable_depth_rendering()
        self.r.update_scene(self.data)
        near = self.model.vis.map.znear * self.model.stat.extent
        half_h = near * math.tan(math.radians(self.fovy) / 2)
        half_w = half_h * self.w / self.h
        for cam in self.r.scene.camera:
            cam.pos[:] = self.pos
            cam.forward[:] = self.fwd
            cam.up[:] = self.up
            cam.frustum_center, cam.frustum_width = 0.0, half_w
            cam.frustum_bottom, cam.frustum_top = -half_h, half_h
            cam.frustum_near = near
            cam.frustum_far = self.model.vis.map.zfar * self.model.stat.extent
        out = self.r.render().copy()
        self.r.disable_depth_rendering()
        return out

    def take(self):
        """(shadow ratio per pixel, luminance as lit, as unshadowed, points)."""
        lit = self._render()
        cast = self.model.light_castshadow.copy()
        self.model.light_castshadow[:] = 0
        flat = self._render()
        self.model.light_castshadow[:] = cast
        depth = self._render(depth=True)
        self.r.close()
        f = self.h / 2 / math.tan(math.radians(self.fovy) / 2)
        v, u = np.mgrid[0 : self.h, 0 : self.w] + 0.5
        x, y = (u - self.w / 2) / f * depth, (self.h / 2 - v) / f * depth
        pts = (
            self.pos
            + depth[..., None] * self.fwd
            + x[..., None] * self.right
            + y[..., None] * self.up
        )
        lum = lambda a: a.astype(float).mean(-1) / 255  # noqa: E731
        lit_l, flat_l = lum(lit), lum(flat)
        ratio = np.where(flat_l > 0.02, lit_l / np.maximum(flat_l, 1e-6), 1.0)
        return ratio, lit_l, flat_l, pts


def _band(model):
    """1 - the east wall's shadow ratio below 1 m over its ratio at 1.4-2.4 m; 0 is no band."""
    centre = (X0 + W / 2, Y0 + D / 2 + 0.6, 1.0)  # beside the table, which stays out of view
    ratio, _, _, pts = _Shots(model, centre, (X0 + W, centre[1], 1.0)).take()
    # Either face of the wall: an inward-wound wall shows its far face, its near one being culled.
    wall = (pts[..., 0] > X0 + W - 0.01) & (pts[..., 0] < X0 + W + T + 0.01)
    low = wall & (pts[..., 2] > 0.05) & (pts[..., 2] < 1.0)
    high = wall & (pts[..., 2] > 1.4) & (pts[..., 2] < 2.4)
    assert low.sum() > 500 and high.sum() > 500, "the view must see the wall low and high"
    return 1 - ratio[low].mean() / ratio[high].mean()


def _contact(model):
    """1 - (floor under the table over the open floor), lit, over the same without shadows."""
    tx, ty = TABLE
    _, lit, flat, pts = _Shots(model, (tx - 1.8, ty - 1.8, 1.6), (tx, ty, 0.1)).take()
    floor = np.abs(pts[..., 2]) < 0.01
    w, d, _ = TOP
    dx = np.maximum(np.abs(pts[..., 0] - tx) - w / 2, 0)
    dy = np.maximum(np.abs(pts[..., 1] - ty) - d / 2, 0)
    dist = np.hypot(dx, dy)
    under, open_ = floor & (dist == 0), floor & (dist > 0.8)
    assert under.sum() > 200 and open_.sum() > 200, "the view must see the floor under the table"
    return 1 - (lit[under].mean() / lit[open_].mean()) / (flat[under].mean() / flat[open_].mean())


def _with_the_single_lamp(xml):
    """The baked room under a single lamp: one 90-degree casting spot at the centre."""
    spec = mujoco.MjSpec.from_file(str(xml))
    for light in list(spec.lights):
        spec.delete(light)
    lamp = spec.worldbody.add_light()
    lamp.pos = [X0 + W / 2, Y0 + D / 2, H - 0.15]
    lamp.dir = [0, 0, -1]
    lamp.cutoff, lamp.exponent = 90.0, 0.0
    lamp.diffuse = [0.4] * 3
    return spec.compile()


@pytest.mark.parametrize("inward", [False, True], ids=["outward", "inward"])
def test_the_walls_carry_no_band(tmp_path, inward):
    xml = _bake(tmp_path, inward=inward)
    assert abs(_band(mujoco.MjModel.from_xml_path(str(xml)))) < 0.03


def test_the_measure_finds_the_band_the_single_lamp_drew(tmp_path):
    """The control: inward walls under a single casting lamp band, so a zero above is a finding."""
    assert _band(_with_the_single_lamp(_bake(tmp_path, inward=True))) > 0.1


@pytest.mark.parametrize("light", [None, {"fixtures": True}], ids=["default", "fixtures"])
def test_the_floor_under_a_table_carries_its_shadow(tmp_path, light):
    model = mujoco.MjModel.from_xml_path(str(_bake(tmp_path, light=light)))
    assert _contact(model) > 0.1
