"""Fit a cheap collision primitive to a link's mesh, for vendor trees that ship none usable.

Some sources hand you collision geometry that is already right -- Husarion ships a body box and four
wheel cylinders, Menagerie's Crazyflie a 32-piece convex decomposition -- and then the porting
playbook's rule is simply to use it. Others do not: MuJoCo Menagerie's xArm collides against nine
full-detail convex hulls, and Doosan's ``*_collision`` meshes are byte-for-byte copies of its visual
CAD. Colliding against those is the playbook's first anti-pattern ("detailed collision meshes because
they were already there"), so the primitive gets fitted here instead.

Bounding-box based on purpose. The point is a cheap, conservative volume, not a faithful one -- a
tight fit to concave CAD is exactly what the visual mesh is for.

**The radius scale is a calibration, not a constant.** A capsule's circular cross-section is a poor
envelope for a prismatic link, so a radius that merely bounds the widest point over-reports
self-collision badly, and a motion planner then refuses configurations that are physically fine.
Measured on the xArm 7 over 3000 uniform joint samples: upstream's exact hulls 8.0%, scale 1.00 →
35.3%, 0.93 → 21.1%, 0.86 → 16.8%. 0.86 was chosen because it puts that arm level with the package's
own reference arms (ur10e 17.4%, ur5e 17.5%) measured identically. **Re-measure per robot** rather
than inheriting the number -- ``roqsim_manipulation_assets/tests/test_xarm7.py`` shows the shape of
that measurement.
"""

from __future__ import annotations

import numpy as np

#: A capsule suits a limb and not a hub: fit one only when the link is meaningfully longer than it is
#: wide, otherwise a box bounds a stubby link far more tightly than a capsule swollen to contain it.
CAPSULE_ASPECT = 1.3

#: MuJoCo capsules and cylinders run along local z; these rotate z onto x or y.
AXIS_QUAT = {0: "0.7071068 0 0.7071068 0", 1: "0.7071068 -0.7071068 0 0", 2: None}


def fit_collision_primitive(verts: np.ndarray, radius_scale: float) -> dict[str, str]:
    """Fit one primitive to *verts* (a link's mesh vertices, in the body frame).

    Returns the MJCF geom attributes -- ``type``, ``size``, ``pos`` and, for a capsule off the z
    axis, ``quat``.
    """
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    centre, extent = (lo + hi) / 2, hi - lo
    axis = int(np.argmax(extent))
    perpendicular = [i for i in range(3) if i != axis]
    radius = float(max(extent[perpendicular]) / 2 * radius_scale)
    half = float(extent[axis] / 2)

    pos = " ".join(f"{v:.5g}" for v in centre)
    if radius > 0 and half / radius >= CAPSULE_ASPECT:
        # size is (radius, cylinder half-length): subtract the radius so the caps stay in the bbox.
        attrs = {"type": "capsule", "size": f"{radius:.5g} {max(half - radius, 1e-4):.5g}",
                 "pos": pos}
        if (quat := AXIS_QUAT[axis]) is not None:
            attrs["quat"] = quat
        return attrs
    return {"type": "box",
            "size": " ".join(f"{v / 2 * radius_scale:.5g}" for v in extent),
            "pos": pos}


def mesh_vertices(model, body_name: str) -> np.ndarray:
    """Vertices of every mesh geom on *body_name*, expressed in that body's frame."""
    import mujoco

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    chunks = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] != body or model.geom_dataid[geom] < 0:
            continue
        mesh = model.geom_dataid[geom]
        adr, num = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[adr:adr + num].reshape(-1, 3)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[geom])
        chunks.append(verts @ rot.reshape(3, 3).T + model.geom_pos[geom])
    if not chunks:
        raise RuntimeError(f"{body_name} carries no mesh geom to fit a collision primitive to")
    return np.concatenate(chunks)


def obj_vertices(path) -> np.ndarray:
    """Vertices of a Wavefront OBJ, for fitting straight off a converted file."""
    return np.array([
        [float(v) for v in line.split()[1:4]]
        for line in open(path).read().splitlines()
        if line.startswith("v ")
    ])
