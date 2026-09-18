"""``ground_truth_pose`` checks: the pose it publishes, and the frame it publishes it under.

The frame NAME carries the weight here. The transform's numbers are ``data.xpos``/``xquat`` read
straight out of MuJoCo, so there is little to get wrong about them; the name is what an offline
evaluator looks the ground truth up by, and a wrong one produces a bag that is well-formed, complete,
and useless. Both halves of that are asserted: that the Gazebo-compatible ``<model>_base_link_gt``
falls out of a correctly nested entry, and that an entry with no owner to ask for a model name is
refused up front rather than publishing whatever it can assemble without one.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.ground_truth_pose import GroundTruthPosePlugin

from roqsim.config import PluginError, load_config_from_dict
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin

SPAWN = (1.25, -0.5)
#: The mount site's offset from the base, in the base frame.
MOUNT = (0.1, 0.05, 0.02)


class _RobotScene(Plugin):
    """A minimal robot entity: a box on a free joint at :data:`SPAWN`, with an unprefixed base.

    Unprefixed on purpose -- ``spawn_robot``'s ``prefix`` defaults to empty, so a real world leaves a
    bare ``base_link`` that an unowned plugin instance can resolve. Prefixing it here would hide the
    very condition :func:`test_an_unowned_entry_is_refused` exists for.
    """

    provides_entity = True
    #: The ``model:`` reference the entity reports -- the frame name is derived from it.
    model = "turtlebot4"

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        base = spec.worldbody.add_body(name="base_link", pos=[*SPAWN, 0.1])
        base.add_freejoint()
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.15, 0.15, 0.1], mass=5.0)
        # A sensor mount off the base's centre, yawed a quarter turn: what a site pose is about.
        base.add_site(name="mouse", pos=list(MOUNT), quat=[0.7071068, 0, 0, 0.7071068])

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.name,
                kind="robot",
                body="base_link",
                meta={"prefix": "", "namespace": "", "model": self.model},
            )
        )


class _PathModelScene(_RobotScene):
    """A robot referenced by PATH rather than by bundled name -- ``spawn_robot`` accepts both."""

    model = "/opt/models/turtlebot4.xml"


def _engine(scene: str = f"{__name__}:_RobotScene", **config):
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {scene: {}, "name": "robot", "components": [{"ground_truth_pose": dict(config)}]}
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    return engine


def _endpoint(scene: str = f"{__name__}:_RobotScene", **config):
    """The ``pose`` endpoint of one ground_truth_pose nested under the robot."""
    engine = _engine(scene, **config)
    return next(e for e in engine.ctx.interface._endpoints if e.name == "pose")


# -- the frame it publishes under ------------------------------------------------------------


def test_the_default_child_frame_is_the_gazebo_one():
    """``<model>_base_link_gt``, so a bag recorded against either simulator analyses the same."""
    assert _endpoint().read()[0][0] == "turtlebot4_base_link_gt"


def test_a_model_given_as_a_path_still_yields_a_frame_name():
    """A frame name cannot carry a path's separators, so only the stem of the reference is used."""
    assert _endpoint(f"{__name__}:_PathModelScene").read()[0][0] == "turtlebot4_base_link_gt"


def test_an_explicit_child_frame_wins():
    """What a multi-robot world sets, since the converter does not namespace child frames."""
    assert _endpoint(child_frame="left_gt").read()[0][0] == "left_gt"


def test_an_unowned_entry_is_refused():
    """No owner means no model name, and an unprefixed 'base_link' resolves anyway -- so the frame
    would come out named after nothing and the run would look healthy to the end."""
    with pytest.raises(PluginError, match="attaches to an entity"):
        load_config_from_dict(
            {
                "sim": {},
                # Declared BESIDE the robot rather than nested under it -- the unowned form.
                "components": [
                    {f"{__name__}:_RobotScene": {}, "name": "robot"},
                    {"ground_truth_pose": {}},
                ],
            }
        )


# -- the pose itself -------------------------------------------------------------------------


def test_the_published_pose_is_the_true_world_pose():
    """Read from ``data.xpos``, so it is the substrate's pose and not an estimate of it."""
    _frame, pos, quat = _endpoint().read()[0]
    assert np.allclose(pos[:2], SPAWN, atol=1e-3)
    assert quat[0] == pytest.approx(1.0, abs=1e-3)  # MuJoCo (w, x, y, z), level


def test_it_declares_itself_owned():
    """The class attribute the refusal above is enforced from."""
    assert GroundTruthPosePlugin.requires_owner is True


# -- a site, and a pose relative to the base ---------------------------------------------------


def _yawed(engine, yaw: float):
    """Put the base at a yaw, so a relative pose and a world pose visibly differ."""
    import math

    d = engine.ctx.data
    d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_forward(engine.ctx.model, d)


def test_a_site_is_published_under_its_own_name_at_its_world_pose():
    """What a stack reading 'the mouse' off a simulator's pose stream matches on is the site's
    name; the numbers are the site's, not the base's."""
    engine = _engine(site="mouse")
    ep = next(e for e in engine.ctx.interface._endpoints if e.name == "pose")
    frame, pos, quat = ep.read()[0]
    assert frame == "mouse"
    d = engine.ctx.data
    sid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_SITE, "mouse")
    assert np.allclose(pos, d.site_xpos[sid])
    # Level base, so the site's world yaw is the mount's quarter turn.
    assert quat[0] == pytest.approx(0.7071068, abs=1e-3)
    assert quat[3] == pytest.approx(0.7071068, abs=1e-3)


def test_a_relative_site_pose_is_the_mount_offset_wherever_the_base_stands():
    """`relative_to: base` is the link-pose a simulator publishes for a model's links: constant
    for a rigid mount however the base is placed, and hanging from the base body."""
    engine = _engine(site="mouse", relative_to="base")
    ep = next(e for e in engine.ctx.interface._endpoints if e.name == "pose")
    assert ep.backend["ros2"]["frame_id"] == "base_link"
    # Copied: a read hands back the plugin's scratch buffers, as the body path hands back views
    # into MjData, and the bridge converts each read before the next.
    level = tuple(np.array(v) if i else v for i, v in enumerate(ep.read()[0]))
    _yawed(engine, 1.0)
    turned = tuple(np.array(v) if i else v for i, v in enumerate(ep.read()[0]))
    for frame, pos, quat in (level, turned):
        assert frame == "mouse"
        assert np.allclose(pos, MOUNT, atol=1e-6)
        assert quat[0] == pytest.approx(0.7071068, abs=1e-6)
        assert quat[3] == pytest.approx(0.7071068, abs=1e-6)
    # ...whereas the world pose of the same site moved with the base.
    world = _engine(site="mouse")
    wep = next(e for e in world.ctx.interface._endpoints if e.name == "pose")
    before = np.array(wep.read()[0][1])
    _yawed(world, 1.0)
    after = np.array(wep.read()[0][1])
    assert not np.allclose(before, after, atol=1e-3)


def test_a_relative_body_pose_is_the_identity():
    """The base relative to itself: the degenerate case a stack composing link poses can hit."""
    frame, pos, quat = _endpoint(relative_to="base").read()[0]
    assert np.allclose(pos, 0.0, atol=1e-9)
    assert np.allclose(quat, [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_a_missing_site_fails_loudly():
    with pytest.raises(RuntimeError, match="site 'nope' not found"):
        _engine(site="nope")


@pytest.mark.parametrize(
    "config", [{"relative_to": "robot"}, {"site": "mouse", "body": "base_link"}, {"rate_hz": 0}]
)
def test_bad_config_is_reported(config):
    assert GroundTruthPosePlugin(config).validate_config(config) != []
