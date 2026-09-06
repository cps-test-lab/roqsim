"""The office desk is a trestle: the span between its two A-frames is meant to be driven through.

MuJoCo collides a mesh geom as its CONVEX HULL, and this desk's hull is a solid block from the floor
to the top -- so a prop that is only its mesh fills the very space it is shaped to leave open, and a
small ground robot crossing under it crashes into thin air. The model therefore carries the mesh as
visual-only and a primitive skeleton for physics; what follows pins both halves of that, because
either one alone is wrong: an open span is not enough if a leg has stopped stopping anything.

The probe is a TurtleBot 4's collision envelope -- the 0.164 m body cylinder from z=0.019 to 0.079
(``roqsim_mobile``'s ``turtlebot4.xml``) -- written out as a plain cylinder so this package keeps its
own tests to its own dependencies.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

TABLE = "office_table"
PROBE_RADIUS = 0.164
PROBE_HEIGHT = 0.060
PROBE_Z = 0.049  # centre: the TB4 body cylinder spans z = 0.019 .. 0.079


def _world(tmp_path, probe_y, probe_x=-1.5):
    plugins = [
        {"spawn_model": {"model": TABLE, "pos": [0.0, 0.0, 0.0]}, "name": "table"},
        {
            "cylinder": {
                "prefix": "probe_",
                "pos": [probe_x, probe_y, PROBE_Z],
                "radius": PROBE_RADIUS,
                "height": PROBE_HEIGHT,
                "mass": 3.0,
                "free": True,
            },
            "name": "probe",
        },
    ]
    engine = Engine(load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path))
    engine.setup()
    engine.reset()
    return engine


def _bodies_prefixed(m, prefix):
    return [
        b
        for b in range(m.nbody)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix)
    ]


def _probe_body(engine):
    m = engine.ctx.model
    bid = _bodies_prefixed(m, "probe_")
    assert bid, "the probe body was not spawned"
    return bid[0]


def _drive(engine, seconds=8.0, speed=0.3):
    """Push the probe along +x until it touches the desk, and return where it got to.

    The push stops at first contact rather than being held on through it: a velocity written every
    step overrides the solver, so a probe that is still being pushed grinds into whatever stopped it
    and the position it reports is a penetration depth rather than a face.
    """
    m, d = engine.ctx.model, engine.ctx.data
    bid = _probe_body(engine)
    adr = m.jnt_dofadr[m.body_jntadr[bid]]
    table = set(_geoms_of_the_table(engine))
    probe = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid}
    for _ in range(int(seconds / m.opt.timestep)):
        touched = any(
            {int(d.contact[i].geom1), int(d.contact[i].geom2)} & table
            and {int(d.contact[i].geom1), int(d.contact[i].geom2)} & probe
            for i in range(d.ncon)
        )
        if touched:
            break
        d.qvel[adr] = speed
        engine.step()
    return d.xpos[bid].copy()


def _geoms_of_the_table(engine):
    m = engine.ctx.model
    return [
        g
        for g in range(m.ngeom)
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(TABLE)
    ]


def test_the_span_between_the_frames_is_open(tmp_path):
    # Down the centre line the desk's footprint is empty: the probe crosses the whole 1.2 m top and
    # comes out the far side, rather than stopping at the near edge as it does against the hull.
    engine = _world(tmp_path, probe_y=0.0)
    end = _drive(engine)
    assert end[0] > 0.6, f"probe stopped under the table at x={end[0]:.3f}"


def test_a_leg_still_stops_the_probe(tmp_path):
    # 0.305 m off the centre line is the line of the two front floor pads. Same push, and the probe
    # is held at the pad's near face (x = -0.58) plus its own radius.
    engine = _world(tmp_path, probe_y=0.305)
    end = _drive(engine)
    assert end[0] == pytest.approx(-0.58 - PROBE_RADIUS, abs=0.02)
    assert end[0] < -0.5, "the probe went through the leg"


def test_the_mesh_is_visual_and_the_skeleton_is_solid(tmp_path):
    # The invariant behind both tests above, stated directly so a regression is named rather than
    # inferred from a probe that suddenly stops.
    engine = _world(tmp_path, probe_y=0.0)
    m = engine.ctx.model
    meshes = [
        g for g in _geoms_of_the_table(engine) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]
    solid = [g for g in _geoms_of_the_table(engine) if m.geom_contype[g] or m.geom_conaffinity[g]]
    assert meshes, "the desk lost its visual mesh"
    assert not any(m.geom_contype[g] or m.geom_conaffinity[g] for g in meshes), (
        "the desk's mesh collides again -- its convex hull fills the span under the top"
    )
    assert len(solid) == 11, "expected the top plus two A-frames of five geoms each"


def test_the_top_is_still_a_surface(tmp_path):
    # Taking contacts off the mesh must not take the tabletop with them: something set down on the
    # desk has to rest on it at 0.717 m, not fall through to the floor.
    plugins = [
        {"spawn_model": {"model": TABLE, "pos": [0.0, 0.0, 0.0]}, "name": "table"},
        {
            "box": {
                "prefix": "parcel_",
                "pos": [0.0, 0.0, 0.85],
                "size": [0.1, 0.1, 0.1],
                "free": True,
            },
            "name": "parcel",
        },
    ]
    engine = Engine(load_config_from_dict({"sim": {}, "plugins": plugins}, base_dir=tmp_path))
    engine.setup()
    engine.reset()
    m, d = engine.ctx.model, engine.ctx.data
    bid = _bodies_prefixed(m, "parcel_")[0]
    for _ in range(int(2.0 / m.opt.timestep)):
        engine.step()
    assert d.xpos[bid][2] == pytest.approx(0.717 + 0.05, abs=0.01)
