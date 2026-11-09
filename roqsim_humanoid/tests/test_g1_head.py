"""The G1's head as it is drawn and as a ray meets it: whole on its torso in every picture, open where
its Mid-360 scans out.

The head mesh carries the opening its lidar needs (``external/convert/g1_head_window.py``), a band all
round the head, which drawn would show the head's top apart from the torso. The cut head is the
collision geometry a ray meets; the drawn head is the whole one, in the window geom group
(:data:`roqsim.presence.WINDOW_GEOM_GROUP`), which every render and viewer window draws and no ray
sees.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.presence import ABSENT_GEOM_GROUP, visible_geomgroup_mask

MODELS = Path(__file__).resolve().parents[1] / "src" / "roqsim_humanoid" / "models"
#: The groups a roqsim render draws: MuJoCo's default 0-2, and the window group; never the absent one.
DRAWN = {0, 1, 2, 5} - {ABSENT_GEOM_GROUP}


@pytest.fixture(scope="module", params=["unitree_g1_dex1", "unitree_g1"])
def g1(request):
    m = mujoco.MjModel.from_xml_path(str(MODELS / f"{request.param}.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return request.param, m, d


def _drawn_heights(m, d, geoms, frame_pos, frame_rot):
    """The heights (m, in a frame) the drawn vertices of ``geoms`` reach, as a set of 5 mm bins."""
    bins = set()
    for g in geoms:
        if int(m.geom_group[g]) not in DRAWN or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        z = (world - frame_pos) @ frame_rot[:, 2]
        bins |= set(np.floor(z / 0.005).astype(int).tolist())
    return bins


def test_the_drawn_head_sits_on_its_torso_with_no_gap(g1):
    name, m, d = g1
    head = "head_link" if name == "unitree_g1_dex1" else "base_link"
    body = m.body(head).id
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
    # the head's drawn geometry, from its lowest to its highest point, in 5 mm steps
    hm = [g for g in geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
          and "head" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or "")]
    assert hm, f"{name}: no head mesh on {head}"
    bins = _drawn_heights(m, d, hm, d.xpos[body], d.xmat[body].reshape(3, 3))
    missing = [b for b in range(min(bins), max(bins) + 1) if b not in bins]
    assert not missing, f"{name}: the drawn head has nothing between {missing[0] * 5} and {missing[-1] * 5 + 5} mm"


def test_a_ray_from_inside_the_head_leaves_it_through_the_window(g1):
    name, m, d = g1
    head = "head_link" if name == "unitree_g1_dex1" else "base_link"
    body = m.body(head).id
    window = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_group[g] == 5]
    assert window, f"{name}: the head has no window geom"
    # the opening's height: where the cut head (what rays meet) has no surface
    cut = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_group[g] == 3
           and "head" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) or "")]
    rot = d.xmat[body].reshape(3, 3)
    mid = m.geom_dataid[cut[0]]
    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    world = v @ d.geom_xmat[cut[0]].reshape(3, 3).T + d.geom_xpos[cut[0]]
    z = (world - d.xpos[body]) @ rot[:, 2]
    occupied = set(np.floor(z / 0.005).astype(int).tolist())
    gap = [b for b in range(min(occupied), max(occupied)) if b not in occupied]
    assert gap, f"{name}: the head's collision mesh has no opening"
    # on the head's axis (its vertices' middle), at the opening's height
    centre = world.mean(axis=0)
    height = float(np.median(gap)) * 0.005 + 0.0025
    origin = centre + rot[:, 2] * (height - float((centre - d.xpos[body]) @ rot[:, 2]))
    geomid = np.zeros(1, np.int32)
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        ray = rot @ np.array([np.cos(a), np.sin(a), 0.0])
        dist = mujoco.mj_ray(m, d, origin, ray, visible_geomgroup_mask(), 1, -1, geomid)
        assert dist < 0 or int(m.geom_bodyid[geomid[0]]) != body or dist > 0.2, (
            f"{name}: a ray at {np.degrees(a):.0f} deg meets the head {dist:.3f} m out"
        )
