# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``attachment``: the load follows while held, is released where it was, and is never teleported.

A cart on a rail with a parcel floating beside it, in a world with gravity switched off. The zero
gravity is what makes the assertions clean rather than a study of friction: a parcel dragged across a
floor by a soft weld constraint slips, and a test that tolerated the slip would tolerate a weld that
was not holding at all. Here the only thing that can move the parcel is the constraint under test.

The assertion that matters most is the negative one -- attaching must not MOVE the parcel, because a
weld activated without rewriting its relative pose snaps the load back to wherever the MJCF declared
it, which is the bug this plugin exists to not have.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.config import PluginError, load_config_from_dict
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.plugins.attachment import AttachmentPlugin

#: Enough force on a light cart that it travels metres in the steps a test runs for -- the
#: distances then dwarf the solver's own tolerance instead of sitting next to it.
PUSH = 40.0


class _TurntableScene(Plugin):
    """A carrier that ROTATES, with the parcel held off its axis (see the rotation test)."""

    provides_entity = True

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.option.gravity = [0.0, 0.0, 0.0]
        table = spec.worldbody.add_body(name="base_link", pos=[0, 0, 0.3])
        table.add_joint(name="spin", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=0.2)
        table.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.2, 0.05], mass=5.0)
        motor = spec.add_actuator()
        motor.name = "spin_motor"
        motor.target = "spin"
        motor.trntype = mujoco.mjtTrn.mjTRN_JOINT

        parcel = spec.worldbody.add_body(name="parcel", pos=[0.0, 0.5, 0.3])
        parcel.add_freejoint()
        parcel.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], mass=0.5)

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.name, kind="robot", body="base_link", meta={"prefix": "", "namespace": ""}
            )
        )


class _CartScene(Plugin):
    """A cart driven along x, a parcel floating beside it, and a bolted-down post.

    Gravity is off (see the module docstring): the parcel is then moved by the weld or by nothing.
    """

    provides_entity = True

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.option.gravity = [0.0, 0.0, 0.0]
        cart = spec.worldbody.add_body(name="base_link", pos=[0, 0, 0.3])
        cart.add_joint(name="rail", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], damping=0.5)
        cart.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.05], mass=5.0)
        motor = spec.add_actuator()
        motor.name = "rail_motor"
        motor.target = "rail"
        motor.trntype = mujoco.mjtTrn.mjTRN_JOINT

        parcel = spec.worldbody.add_body(name="parcel", pos=[0.0, 0.5, 0.3])
        parcel.add_freejoint()
        parcel.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], mass=0.5)

        # Something that cannot move, to check the refusal.
        post = spec.worldbody.add_body(name="post", pos=[1.0, 1.0, 0.25])
        post.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.25], mass=1.0)

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.name, kind="robot", body="base_link", meta={"prefix": "", "namespace": ""}
            )
        )


def _engine(*, scene: str = f"{__name__}:_CartScene", **config):
    from roqsim.engine import Engine

    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    scene: {},
                    "name": "robot",
                    "components": [{"attachment": {"body": "parcel", **config}}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    for _ in range(50):  # settle
        engine.step()
    return engine


def _plugin(engine) -> AttachmentPlugin:
    return next(p for p in engine.plugins if isinstance(p, AttachmentPlugin))


def _drive(engine, steps: int = 400):
    for _ in range(steps):
        engine.ctx.data.ctrl[:] = PUSH
        engine.step()


def _pos(engine, body: str) -> np.ndarray:
    bid = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, body)
    return np.array(engine.ctx.data.xpos[bid])


# -- carrying --------------------------------------------------------------------------------


def test_an_unattached_load_is_left_behind():
    """The control: without the weld, driving away leaves the parcel where it was."""
    engine = _engine()
    before = _pos(engine, "parcel")
    _drive(engine)
    assert _pos(engine, "base_link")[0] > 0.2, "the cart should have moved"
    assert np.allclose(_pos(engine, "parcel"), before, atol=1e-3)


def test_attaching_does_not_move_the_load():
    """The bug this exists to not have: a weld activated without rewriting its relative pose snaps
    the load to wherever the MJCF declared it."""
    engine = _engine()
    _drive(engine, 200)  # the cart is somewhere else now; the parcel is not
    before = _pos(engine, "parcel")
    _plugin(engine).set_attached(True, engine.ctx.sim_time)
    engine.step()
    assert np.allclose(_pos(engine, "parcel"), before, atol=2e-3)


def test_an_attached_load_travels_with_the_carrier():
    engine = _engine()
    parcel_before, cart_before = _pos(engine, "parcel"), _pos(engine, "base_link")
    _plugin(engine).set_attached(True, engine.ctx.sim_time)
    _drive(engine)
    moved_cart = _pos(engine, "base_link")[0] - cart_before[0]
    moved_parcel = _pos(engine, "parcel")[0] - parcel_before[0]
    assert moved_cart > 0.2
    # A weld is a solver constraint, not a rigid link, so the load lags by a little under
    # acceleration. What must not happen is it staying behind or being flung somewhere else.
    assert moved_parcel == pytest.approx(moved_cart, abs=0.15)
    # And it keeps its OFFSET: the direction of the stored relative pose is what decides this, and
    # written the other way round the solver drives the parcel to twice its offset.
    assert _pos(engine, "parcel")[1] == pytest.approx(0.5, abs=0.05)


def test_a_rotating_carrier_swings_its_load_around_with_it():
    """A translation-only check cannot tell the relative pose's direction from its negation.

    The turntable rotates a quarter turn with the parcel held off-axis: held correctly the parcel
    swings to where that offset now points, and its own orientation follows the carrier's.
    """
    engine = _engine(scene=f"{__name__}:_TurntableScene")
    plugin = _plugin(engine)
    plugin.set_attached(True, engine.ctx.sim_time)
    data, model = engine.ctx.data, engine.ctx.model
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "spin")
    target = np.pi / 2
    for _ in range(2000):  # position-servo the turntable to a quarter turn and hold it
        data.ctrl[:] = 20.0 * (target - data.qpos[model.jnt_qposadr[hinge]]) - 4.0 * data.qvel[0]
        engine.step()
    assert data.qpos[model.jnt_qposadr[hinge]] == pytest.approx(target, abs=0.05)
    # The parcel started 0.5 m along +y of an unrotated carrier; a quarter turn about z takes that
    # offset to -x. Getting the stored pose backwards would swing it to +x instead.
    parcel = _pos(engine, "parcel")
    assert parcel[0] == pytest.approx(-0.5, abs=0.08)
    assert abs(parcel[1]) < 0.08


def test_a_released_load_stops_following_the_carrier():
    """Release keeps the velocity it had -- a parcel let go from a moving deck coasts on, which is
    the behaviour that makes a release worth simulating. What must end is the COUPLING."""
    engine = _engine(attached=True)
    _drive(engine, 300)
    gap_before = _pos(engine, "base_link")[0] - _pos(engine, "parcel")[0]
    _plugin(engine).set_attached(False, engine.ctx.sim_time)
    # 200 steps, not more: the default `empty_room` is walled at 5 m, and a cart that reaches the
    # wall stops while the released parcel coasts into it -- which looks exactly like a release that
    # never happened.
    _drive(engine, 200)
    gap_after = _pos(engine, "base_link")[0] - _pos(engine, "parcel")[0]
    assert gap_after > gap_before + 0.3, "the cart accelerated away; the parcel did not follow"


# -- state and its wiring ----------------------------------------------------------------------


def test_the_initial_state_is_config_so_it_can_be_swept():
    """ "Does the robot start loaded" is a campaign factor, not a second world file."""
    assert _plugin(_engine())._report.attached is False
    loaded = _engine(attached=True)
    assert _plugin(loaded)._report.attached is True
    assert bool(loaded.ctx.data.eq_active[_plugin(loaded)._eq_id]) is True


def test_a_reset_restores_the_configured_state():
    engine = _engine()
    plugin = _plugin(engine)
    plugin.set_attached(True, engine.ctx.sim_time)
    engine.reset()
    assert bool(engine.ctx.data.eq_active[plugin._eq_id]) is False
    assert plugin._report.changes == 0, "the reset state is the start, not a change the trial made"


def test_the_switch_and_its_report_are_on_the_blackboard_and_the_interface():
    engine = _engine()
    handle = engine.ctx.blackboard.get("attachment:robot.attachment")
    assert handle is not None and handle.is_active() is False
    handle.set_active(True)
    assert handle.is_active() is True
    assert handle.read_state().attached is True

    names = {e.name: e for e in engine.ctx.interface.all()}
    assert names["attach"].backend["ros2"]["service"] == "std_srvs.srv.SetBool"
    assert names["attach"].backend["ros2"]["name"] == "robot/attachment/attach"
    assert names["attached"].backend["ros2"]["field"] == "attached"
    assert names["attached"].read().attached is True


def test_switching_to_the_state_it_is_already_in_is_not_a_change():
    engine = _engine()
    plugin = _plugin(engine)
    plugin.set_attached(False, 1.0)
    assert plugin._report.changes == 0


# -- refusals ----------------------------------------------------------------------------------


def test_a_load_welded_to_the_world_is_refused():
    """A weld to a body with no degrees of freedom solves nothing, and the failure is invisible:
    the service replies 'attached' and the object never follows."""
    with pytest.raises(RuntimeError, match="welded to the world"):
        _engine(body="post")


def test_it_belongs_to_the_thing_that_carries():
    with pytest.raises(PluginError):
        load_config_from_dict({"sim": {}, "components": [{"attachment": {"body": "parcel"}}]})


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "'body' is required"),
        ({"body": "parcel", "attached": "yes"}, "'attached' must be true or false"),
    ],
)
def test_config_errors_are_reported_by_name(config, expected):
    errors = AttachmentPlugin(config, entity="robot", label="attachment").validate_config(config)
    assert any(expected in e for e in errors), errors
