"""spawn_model `mocap: true`: a prop a plugin drives, rather than physics or nothing.

The third state of the same axis. ``free: false`` is welded scenery and ``free: true`` is a body
physics moves; this is the one in between -- collision geometry that goes where something *tells* it
to go. It is what a controlled obstacle needs: the robot under test can see it and bump into it, but
cannot shove it off the course the experiment set, and it costs the solver nothing because it has no
degrees of freedom.

The guards are the point as much as the feature. Both ways of getting it wrong -- asking for a free
joint too, or driving a prop that has its own articulation -- compile without complaint in MuJoCo and
then silently do the wrong thing.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError

CRATE = """<mujoco model="crate">
  <worldbody><body name="crate">
    <geom name="crate_geom" type="box" size=".25 .25 .25"/>
  </body></worldbody>
</mujoco>"""

HINGED = """<mujoco model="hinged">
  <worldbody><body name="hinged">
    <joint name="lid" type="hinge" axis="0 1 0"/>
    <geom name="hinged_geom" type="box" size=".25 .25 .25"/>
  </body></worldbody>
</mujoco>"""


def _model(tmp_path, name, xml):
    path = tmp_path / f"{name}.xml"
    path.write_text(xml)
    return str(path)


def _world(tmp_path, xml=CRATE, name="crate", **prop):
    return load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_model": {
                        "model": _model(tmp_path, name, xml),
                        "pose": {"position": {"x": 1.0, "y": 2.0, "z": 0.25}},
                        **prop,
                    },
                    "name": "cart",
                }
            ],
        },
        base_dir=tmp_path,
    )


def _mocapid(engine):
    model = engine.ctx.model
    body = engine.ctx.entities.get("cart").body
    return int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])


def test_a_mocap_prop_has_no_degrees_of_freedom(tmp_path):
    """The whole reason to prefer it over a driven free body: it is free of the solver.

    ``nq == 0`` is the strong form -- the prop contributes no generalised coordinate at all, so a
    world can carry many of them without the physics step growing.
    """
    engine = Engine(_world(tmp_path, mocap=True))
    engine.setup()
    entity = engine.ctx.entities.get("cart")
    assert entity.kind == "object"  # it moves, so it is not scenery
    assert entity.meta.get("mocap") is True  # how a nested driver discovers it may write the pose
    assert _mocapid(engine) >= 0
    assert engine.ctx.model.nq == 0


def test_a_driver_writes_the_pose_and_physics_leaves_it_alone(tmp_path):
    engine = Engine(_world(tmp_path, mocap=True))
    engine.setup()
    engine.reset()
    data, mid = engine.ctx.data, _mocapid(engine)
    assert data.mocap_pos[mid] == pytest.approx([1.0, 2.0, 0.25])

    data.mocap_pos[mid] = [5.0, 5.0, 0.25]
    for _ in range(50):
        engine.step()
    # Gravity would have taken a free body to the floor over 50 steps; this one stays put.
    assert data.mocap_pos[mid] == pytest.approx([5.0, 5.0, 0.25])


def test_reset_reseats_a_driven_prop(tmp_path):
    """Repetition N must not start where repetition N-1 was abandoned.

    ``spawn_model`` writes no code for this: ``mj_resetData`` re-initialises ``mocap_pos`` from the
    body's ``body_pos``, which is the spawn pose. The test pins that behaviour rather than the
    absence of code, so the day it stops holding this fails instead of the campaign.
    """
    engine = Engine(_world(tmp_path, mocap=True))
    engine.setup()
    engine.reset()
    data, mid = engine.ctx.data, _mocapid(engine)
    data.mocap_pos[mid] = [5.0, 5.0, 0.25]
    engine.reset()
    assert data.mocap_pos[mid] == pytest.approx([1.0, 2.0, 0.25])


def test_default_is_still_welded_scenery(tmp_path):
    engine = Engine(_world(tmp_path))
    engine.setup()
    assert engine.ctx.entities.get("cart").kind == "prop"
    assert _mocapid(engine) < 0


def test_free_and_mocap_together_are_refused(tmp_path):
    """They are not two strengths of one thing -- they disagree about who owns the pose."""
    with pytest.raises(PluginError, match="mutually exclusive"):
        Engine(_world(tmp_path, mocap=True, free=True))


def test_static_publish_tf_is_refused_for_a_driven_prop(tmp_path):
    with pytest.raises(PluginError, match="publish_tf: static"):
        Engine(_world(tmp_path, mocap=True, publish_tf="static"))


def test_mocap_on_an_articulated_prop_is_refused(tmp_path):
    """MuJoCo compiles a jointed mocap body and then never moves the joint.

    Without this the prop arrives looking right and is frozen, which is a slow thing to diagnose.
    """
    engine = Engine(_world(tmp_path, xml=HINGED, name="hinged", mocap=True))
    with pytest.raises(Exception, match="no degrees of freedom|inert"):
        engine.setup()
