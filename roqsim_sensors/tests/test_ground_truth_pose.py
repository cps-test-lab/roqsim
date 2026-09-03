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


def _endpoint(scene: str = f"{__name__}:_RobotScene", **config):
    """The ``pose`` endpoint of one ground_truth_pose nested under the robot."""
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
                # Declared BESIDE the robot rather than nested under it -- the pre-ownership dialect.
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
