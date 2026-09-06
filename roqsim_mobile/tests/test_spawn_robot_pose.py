# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``pose:`` -- the spawn pose written the way ``SpawnEntity.srv`` states one.

One shape for a pose whichever door it comes in at: a world declaring where the robot starts, and a
spawn call placing it mid-trial. What these check is that the declared pose really reaches the base
free joint, and that the two ways of writing it cannot both be given.
"""

from __future__ import annotations

import math

import mujoco
import pytest

from roqsim.config import instantiate_plugins, load_config_from_dict
from roqsim.engine import Engine

pytest.importorskip("roqsim_mobile", reason="spawn_robot lives in roqsim_mobile")

SPAWN_ROBOT = "roqsim_mobile.plugins.spawn_robot:SpawnRobotPlugin"


def _quat(yaw):
    return {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2.0), "w": math.cos(yaw / 2.0)}


def _cfg(config):
    return load_config_from_dict(
        {"sim": {"world": "empty_room"}, "plugins": [{SPAWN_ROBOT: config, "name": "robot"}]},
        overrides={"components": {"robot.oakd_camera": {"enabled": False}}},
    )


def _engine(config):
    engine = Engine(_cfg({"model": "turtlebot4", **config}))
    engine.setup()
    return engine


def _base_qpos(engine):
    jid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_JOINT, "base_free")
    adr = int(engine.ctx.model.jnt_qposadr[jid])
    return [float(v) for v in engine.ctx.data.qpos[adr : adr + 7]]


def test_a_pose_places_the_base():
    engine = _engine({"pose": {"position": {"x": 1.5, "y": -2.0}, "orientation": _quat(0.75)}})
    x, y, _z, w, qx, qy, qz = _base_qpos(engine)
    assert (x, y) == pytest.approx((1.5, -2.0))
    assert (w, qx, qy, qz) == pytest.approx((math.cos(0.375), 0.0, 0.0, math.sin(0.375)), abs=1e-9)


def test_a_pose_and_pos_yaw_agree_on_the_same_placement():
    """The two spellings are one pose, so a world converted from one to the other does not move."""
    by_pose = _base_qpos(
        _engine({"pose": {"position": {"x": 1.0, "y": 2.0}, "orientation": _quat(-0.4)}})
    )
    by_keys = _base_qpos(_engine({"pos": [1.0, 2.0], "yaw": -0.4}))
    assert by_pose == pytest.approx(by_keys, abs=1e-9)


def test_an_omitted_z_still_uses_the_model_s_resting_height():
    """A TurtleBot's base_link is authored at the origin with its wheels below it, so a pose that
    left z unstated must not bury it -- the same rule ``pos: [x, y]`` already follows."""
    z = _base_qpos(_engine({"pose": {"position": {"x": 0.0, "y": 0.0}}}))[2]
    assert z == pytest.approx(_base_qpos(_engine({"pos": [0.0, 0.0]}))[2])
    assert z > 0.0


def test_a_stated_z_overrides_it():
    z = _base_qpos(_engine({"pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.4}}}))[2]
    assert z == pytest.approx(0.4)


def test_a_pose_carries_a_rotation_that_is_not_a_heading():
    """The service accepts any unit quaternion and the base has a free joint, so a tilted spawn is
    taken at its word rather than flattened to its yaw."""
    tilt = {"x": math.sin(0.2), "y": 0.0, "z": 0.0, "w": math.cos(0.2)}  # roll 0.4 rad
    _x, _y, _z, w, qx, qy, qz = _base_qpos(
        _engine({"pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.5}, "orientation": tilt}})
    )
    assert (w, qx, qy, qz) == pytest.approx((math.cos(0.2), math.sin(0.2), 0.0, 0.0), abs=1e-9)


def test_the_entity_meta_still_advertises_a_heading():
    """``initial_pose`` is what a consumer that can only act on a heading reads, so it stays
    (x, y, yaw) whatever rotation the pose carried."""
    engine = _engine({"pose": {"position": {"x": 3.0, "y": 4.0}, "orientation": _quat(1.2)}})
    x, y, yaw = engine.ctx.entities.get("robot").meta["initial_pose"]
    assert (x, y) == pytest.approx((3.0, 4.0))
    assert yaw == pytest.approx(1.2)


@pytest.mark.parametrize("other", [{"pos": [1.0, 2.0]}, {"yaw": 0.5}])
def test_pose_cannot_be_combined_with_pos_or_yaw(other):
    """Refused rather than resolved: the wrong guess puts the robot somewhere plausible."""
    cfg = _cfg({"model": "turtlebot4", "pose": {"position": {"x": 0.0, "y": 0.0}}, **other})
    with pytest.raises(Exception, match="cannot be combined with"):
        instantiate_plugins(cfg)


def test_a_yaw_inside_the_orientation_is_refused():
    """The one mistake the shared shape exists to catch, checked through the plugin's validator so
    it is reported before compute is spent rather than at compile."""
    cfg = _cfg(
        {
            "model": "turtlebot4",
            "pose": {"position": {"x": 0.0, "y": 0.0}, "orientation": {"yaw": 1.57}},
        }
    )
    with pytest.raises(Exception, match="not a yaw"):
        instantiate_plugins(cfg)


def test_a_disabled_entry_never_reads_its_pose():
    """``enabled: false`` is the removal, so there is no pose to state.

    It matters for a campaign that sweeps an entry's ``enabled`` and writes its pose on the same
    channel: the cells that switch the entry off must not have to supply a pose for a body nothing
    builds. Nothing here special-cases that -- a disabled entry is never constructed, so its
    validator never runs -- and this is what keeps it true.
    """
    cfg = load_config_from_dict(
        {
            "sim": {"world": "empty_room"},
            "plugins": [
                {
                    SPAWN_ROBOT: {
                        "model": "turtlebot4",
                        # Refused outright on an enabled entry (the test above); read by nobody here.
                        "pose": {"position": {"x": 0.0, "y": 0.0}, "orientation": {"yaw": 1.57}},
                    },
                    "name": "robot",
                    "enabled": False,
                }
            ],
        }
    )
    # Nothing at all, not merely no spawn_robot: disabling an entry that registers an entity
    # disables the components it owns, which is resolved when the document loads rather than
    # when a plugin is built.
    assert [type(p).__name__ for p in instantiate_plugins(cfg)] == []
