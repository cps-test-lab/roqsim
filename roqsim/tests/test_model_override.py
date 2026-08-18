"""model_override: changing named model values mid-run, and noticing when the change did nothing.

Written for a mobile-manipulation trial that has to LOSE a grasped object at a chosen instant, so the
load-bearing behaviour is a pair: a crate resting on a ramp must not move while the override is
inert, and must slide once it is applied. Everything else here defends one of the two ways such an
override fails *silently* -- a write the dynamics ignore (``body_mass`` without ``mj_setConst``), and a
write that lands in the model while MuJoCo keeps using the other geom's value.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
import pytest

from roqsim.context import Entity, SimContext
from roqsim.plugins.model_override import MJMINMU, ModelOverridePlugin, field_catalog, refusal_reasons

# A crate resting on a ramp, plus a slider driven by a position servo.
#
# The crate carries priority="1", which is what makes this scene able to test the trap: MuJoCo takes a
# contact's friction from the higher-priority geom, so overriding the CRATE governs the contact and
# overriding the RAMP cannot. The oversized contype=0 geom is the visual-only case that `bodies:` and
# `entity:` sweep up alongside the collider.
SCENE = """
<mujoco model="model_override_test">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ramp" type="box" size="1 1 0.02" euler="0 20 0" friction="1.0 0.005 0.0001"/>
    <body name="crate" pos="0 0 0.4">
      <freejoint/>
      <geom name="crate" type="box" size="0.05 0.05 0.05" mass="1"
            priority="1" friction="0.7 0.02 0.001" euler="0 20 0"/>
      <geom name="crate_visual" type="box" size="0.06 0.06 0.06" contype="0" conaffinity="0"/>
    </body>
    <body name="slider" pos="1 0 0.5">
      <joint name="slide_joint" type="slide" axis="0 0 1" range="-1 1"/>
      <geom name="slider_geom" type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="lift" joint="slide_joint" kp="100" forcerange="-50 50"/>
  </actuator>
</mujoco>
"""

# For `body_mass` only, and deliberately: with nothing touching the body and no gravity, qacc IS
# force/mass, so the assertion is arithmetic rather than the outcome of a constraint solve. Put the
# body on the ground instead and the number stops meaning anything (measured: 26.12 m/s^2, which is
# neither mass's answer).
FREE = """
<mujoco model="model_override_free">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="crate" pos="0 0 1">
      <freejoint/>
      <geom name="crate" type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

FRICTION_OFF = {"field": "geom_friction", "select": ["crate"], "to": 0.0}
MASKS_OFF = [
    {"field": "geom_contype", "select": ["crate"], "to": 0},
    {"field": "geom_conaffinity", "select": ["crate"], "to": 0},
]


def _plugin(*overrides, scene=SCENE, **cfg):
    """A configured plugin on a fresh scene. Config errors are raised, not returned."""
    model = mujoco.MjModel.from_xml_string(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    # An entity whose NAME differs from its BODY, which is the case `entity:` exists for.
    ctx.entities.add(Entity(name="parcel", kind="object", body="crate"))
    config = {"overrides": list(overrides), **cfg}
    plugin = ModelOverridePlugin(config, name="grip_fault")
    errors = plugin.validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    plugin.configure(ctx)
    plugin.on_reset(ctx)
    return ctx, plugin


def _run(ctx, plugin, seconds):
    """Step for *seconds*, draining posted commands as the engine does. Returns the crate's travel."""
    start = float(ctx.data.qpos[0])
    for _ in range(int(seconds / ctx.model.opt.timestep)):
        ctx.drain_commands()
        mujoco.mj_step(ctx.model, ctx.data)
        plugin.post_step(ctx)
    return abs(float(ctx.data.qpos[0]) - start)


def _at_rest(ctx, plugin):
    """Land the crate on the ramp and stop it, so a later measurement starts from rest."""
    _run(ctx, plugin, 2.0)
    ctx.data.qvel[:] = 0.0
    return ctx, plugin


# -- the load-bearing pair ----------------------------------------------------------------------
def test_the_crate_holds_until_friction_is_overridden():
    """Inert, the crate stays put; applied, it slides. Both halves, or the test proves nothing."""
    ctx, plugin = _at_rest(*_plugin(FRICTION_OFF))

    held = _run(ctx, plugin, 2.0)
    assert ctx.data.ncon > 0, "the test is vacuous unless the crate is actually resting on the ramp"
    # ~2.9 mm of creep at this timestep on a soft contact -- not zero, and 50x smaller than the slide.
    assert held < 0.005, f"the crate should hold while the override is inert, moved {held:.4f} m"

    plugin.set_active(True)
    assert _run(ctx, plugin, 0.3) > 0.05, "the crate should slide once friction is overridden"


def test_the_applied_contact_friction_is_clamped_not_zero():
    """MuJoCo clamps a contact's friction to mjMINMU, so `to: 0` never reads back as 0.0."""
    ctx, plugin = _at_rest(*_plugin(FRICTION_OFF))
    plugin.set_active(True)
    _run(ctx, plugin, 0.05)
    applied = float(ctx.data.contact[0].friction[0])
    assert applied <= 1e-4, f"expected the pair to be frictionless, got {applied}"
    assert applied == pytest.approx(MJMINMU, abs=1e-6)


def test_restoring_is_exact_and_the_crate_holds_again():
    """Restoring writes back the values read at configure, not a guess at what the world declared."""
    ctx, plugin = _at_rest(*_plugin(FRICTION_OFF))
    before = np.array(ctx.model.geom_friction, copy=True)

    plugin.set_active(True)
    _run(ctx, plugin, 0.3)
    plugin.set_active(False)
    ctx.data.qvel[:] = 0.0

    np.testing.assert_array_equal(ctx.model.geom_friction, before)
    assert _run(ctx, plugin, 1.0) < 0.005, "the crate should hold again once friction is restored"


# -- the contact masks --------------------------------------------------------------------------
def test_zeroing_both_masks_removes_the_contact():
    """Both fields, and the ncon assertion is the point: contype alone leaves the contact in place."""
    ctx, plugin = _at_rest(*_plugin(*MASKS_OFF))
    assert ctx.data.ncon > 0

    plugin.set_active(True)
    _run(ctx, plugin, 0.02)

    crate = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_GEOM, "crate")
    assert int(ctx.model.geom_contype[crate]) == 0
    assert int(ctx.model.geom_conaffinity[crate]) == 0
    assert ctx.data.ncon == 0, "the contact should be gone, not merely masked on one side"


def test_contype_without_conaffinity_is_refused():
    """Measured to be a no-op, so it is a startup error rather than a fault that never fires."""
    with pytest.raises(RuntimeError, match="but not geom_conaffinity"):
        _plugin(MASKS_OFF[0])


# -- the other write classes --------------------------------------------------------------------
def test_actuator_forcerange_limits_a_saturating_servo():
    ctx, plugin = _plugin({"field": "actuator_forcerange", "select": ["lift"], "to": [-0.5, 0.5]})
    ctx.data.ctrl[0] = 1.0
    _run(ctx, plugin, 0.2)
    assert abs(float(ctx.data.actuator_force[0])) > 1.0

    plugin.set_active(True)
    _run(ctx, plugin, 0.2)
    assert abs(float(ctx.data.actuator_force[0])) == pytest.approx(0.5, abs=0.01)


def test_body_mass_reaches_the_dynamics():
    """Fails without mj_setConst: the write lands in the model and the mass matrix ignores it.

    Free body, no gravity, so qacc is force/mass and nothing else -- see FREE's comment.
    """
    ctx, plugin = _plugin({"field": "body_mass", "select": ["crate"], "to": 20.0}, scene=FREE)

    def qacc_under_100n():
        ctx.data.qfrc_applied[:] = 0.0
        ctx.data.qfrc_applied[2] = 100.0
        mujoco.mj_forward(ctx.model, ctx.data)
        return float(ctx.data.qacc[2])

    assert qacc_under_100n() == pytest.approx(100.0, rel=1e-3)
    plugin.set_active(True)
    assert qacc_under_100n() == pytest.approx(5.0, rel=1e-3), "100 N on 20 kg is 5 m/s^2"
    plugin.set_active(False)
    assert qacc_under_100n() == pytest.approx(100.0, rel=1e-3)
    assert float(ctx.model.body_subtreemass[1]) == pytest.approx(1.0)


# -- did it land? -------------------------------------------------------------------------------
def test_a_dominating_partner_is_reported_rather_than_predicted(caplog):
    """Override the LOW-priority side and MuJoCo keeps using the crate's friction.

    This is the trap the plugin deliberately does not pre-compute: it is caught by reading the
    applied contact, which cannot be wrong, instead of re-deriving MuJoCo's mixing rule at configure.
    """
    ctx, plugin = _at_rest(*_plugin({"field": "geom_friction", "select": ["ramp"], "to": 0.0}))

    with caplog.at_level(logging.WARNING):
        plugin.set_active(True)
        _run(ctx, plugin, 0.05)

    assert plugin.read_state().verified == "no_effect"
    assert float(ctx.data.contact[0].friction[0]) == pytest.approx(0.7, abs=0.01)
    assert "crate governs this pair" in caplog.text


def test_a_landed_override_is_reported_as_landed():
    ctx, plugin = _at_rest(*_plugin(FRICTION_OFF))
    plugin.set_active(True)
    _run(ctx, plugin, 0.05)
    assert plugin.read_state().verified == "landed"


def test_verified_is_untested_when_nothing_was_touching(caplog):
    """An override fired mid-air is not a failure, and must not warn like one."""
    ctx, plugin = _plugin(FRICTION_OFF)  # not settled: the crate is still falling
    with caplog.at_level(logging.WARNING):
        plugin.set_active(True)
        _run(ctx, plugin, 0.02)
    assert plugin.read_state().verified == "untested"
    assert "did not land" not in caplog.text


# -- refusals and loud failures -----------------------------------------------------------------
@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("geom_size", 0.4, "geom_rbound is cached"),
        ("geom_priority", 3, "swaps the contact's stiffness model"),
        ("geom_rgba", 0.0, "not on the allowlist"),
    ],
)
def test_refused_fields_say_why(field, value, expected):
    with pytest.raises(ValueError, match=expected):
        _plugin({"field": field, "select": ["crate"], "to": value})


def test_a_global_option_is_redirected_to_the_sim_block():
    """The overlap that would otherwise be silent: two ways to set the same global parameters."""
    with pytest.raises(ValueError, match="sim.contact_override"):
        _plugin({"field": "opt.o_friction", "to": 0.0})


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"field": "geom_friction", "select": ["nope"], "to": 0.0}, "no geom named 'nope'"),
        ({"field": "geom_friction", "select": ["crate_visual"], "to": 0.0}, "visual-only"),
        ({"field": "geom_friction", "bodies": ["nope"], "to": 0.0}, "unknown or carries no geoms"),
        ({"field": "geom_friction", "entity": "nope", "to": 0.0}, "not registered"),
        (
            {"field": "actuator_forcerange", "entity": "parcel", "to": [-1.0, 1.0]},
            "cannot select actuators",
        ),
    ],
)
def test_selection_failures_are_loud(override, expected):
    """A typo'd override never fires, and the trial then reports the UNFAULTED outcome as if it had."""
    with pytest.raises((RuntimeError, ValueError), match=expected):
        _plugin(override)


def test_an_empty_selection_is_refused():
    with pytest.raises(ValueError, match="selects nothing"):
        _plugin({"field": "geom_friction", "to": 0.0})


def test_an_explicit_pair_is_refused():
    """A <pair>'s own friction wins, so the write would land in the model and change no contact."""
    paired = SCENE.replace(
        "</worldbody>",
        '</worldbody><contact><pair geom1="crate" geom2="ramp" friction="0.3 0.3 0.005"/></contact>',
    )
    with pytest.raises(RuntimeError, match="explicit <pair>"):
        _plugin(FRICTION_OFF, scene=paired)


def test_a_geom_made_absent_by_presence_is_refused():
    """presence writes the same two mask fields with its own save/restore; interleaving corrupts both."""
    from roqsim.presence import set_present

    model = mujoco.MjModel.from_xml_string(SCENE)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, mujoco.MjData(model)
    entity = Entity(name="parcel", kind="object", body="crate")
    ctx.entities.add(entity)
    set_present(ctx, entity, False)

    plugin = ModelOverridePlugin({"overrides": [FRICTION_OFF]}, name="grip_fault")
    with pytest.raises(RuntimeError, match="roqsim.presence has made absent"):
        plugin.configure(ctx)


# -- selection, reset, and the surfaces ---------------------------------------------------------
def test_entity_and_bodies_expand_to_geoms():
    """`parcel` is the ENTITY; `crate` is the body. Both expand, and the wrong one resolves to nothing."""
    _, by_entity = _plugin({"field": "geom_friction", "entity": "parcel", "to": 0.0})
    _, by_body = _plugin({"field": "geom_friction", "bodies": ["crate"], "to": 0.0})
    assert by_entity._targets[0].ids == by_body._targets[0].ids
    assert len(by_entity._targets[0].ids) == 2  # the collider and its visual sibling

    with pytest.raises(RuntimeError, match="unknown or carries no geoms"):
        _plugin({"field": "geom_friction", "bodies": ["parcel"], "to": 0.0})


def test_on_reset_returns_to_the_configured_state():
    """Without this, repetition 2 of a campaign starts already faulted and reports a plausible wrong
    number: Engine.reset resets MjData and never touches MjModel."""
    ctx, plugin = _at_rest(*_plugin(FRICTION_OFF))
    nominal = np.array(ctx.model.geom_friction, copy=True)

    plugin.set_active(True)
    plugin.on_reset(ctx)

    np.testing.assert_array_equal(ctx.model.geom_friction, nominal)
    assert plugin.read_state().active is False


def test_a_configured_active_override_applies_and_survives_reset():
    """`active: true` is how a whole campaign cell runs degraded, so it is a static factor."""
    ctx, plugin = _plugin(FRICTION_OFF, active=True)
    crate = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_GEOM, "crate")
    assert float(ctx.model.geom_friction[crate][0]) == 0.0

    plugin.on_reset(ctx)
    assert float(ctx.model.geom_friction[crate][0]) == 0.0
    assert plugin.read_state().active is True


def test_the_endpoints_and_the_handle():
    """The ROS path, exercised without ROS: the bridge only ever calls write() and read()."""
    ctx, plugin = _at_rest(*_plugin(FRICTION_OFF))
    endpoints = {e.name: e for e in ctx.interface.all()}

    inbound = endpoints["override"]
    assert inbound.direction == "in"
    assert inbound.backend["ros2"]["service"] == "std_srvs.srv.SetBool"
    assert inbound.backend["ros2"]["state_key"] == "model_override:grip_fault"

    assert endpoints["override_state"].backend["ros2"] == {
        "type": "std_msgs.msg.Bool",
        "field": "active",
        "topic": "override_state",
    }
    assert endpoints["override_verified"].backend["ros2"]["field"] == "verified"

    inbound.write(True)  # what the bridge's service handler does, via ctx.post
    _run(ctx, plugin, 0.05)
    assert endpoints["override_state"].read().active is True
    assert endpoints["override_verified"].read().verified == "landed"

    # Scoped by the instance name, so two faults in one world do not both serve `/override`.
    assert all(e.namespace == "grip_fault" for e in endpoints.values())

    handle = ctx.blackboard.require("model_override:grip_fault")
    assert handle.is_active() is True
    handle.set_active(False)
    ctx.drain_commands()
    assert handle.read_state().active is False


def test_two_faults_in_one_world_do_not_collide():
    """Both would serve `/override`, and two services on one name is a collision, not redundancy."""
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data

    for name in ("grip_fault", "traction_fault"):
        ModelOverridePlugin({"overrides": [FRICTION_OFF]}, name=name).configure(ctx)

    served = {(e.namespace, e.name) for e in ctx.interface.all()}
    assert ("grip_fault", "override") in served
    assert ("traction_fault", "override") in served
    assert len(served) == 6, f"each instance needs its own three endpoints, got {sorted(served)}"


def test_an_unnamed_instance_stays_unscoped(capsys):
    """`self.name` is the class name when the world never named it -- not a namespace anyone wants."""
    ctx, _ = _plugin(FRICTION_OFF)
    unnamed = ModelOverridePlugin({"overrides": [FRICTION_OFF]})
    ctx2 = SimContext(config={})
    ctx2.model, ctx2.data = ctx.model, ctx.data
    unnamed.configure(ctx2)
    assert {e.namespace for e in ctx2.interface.all()} == {""}


def test_the_catalog_documents_every_allowlisted_field():
    """`roqsim scenes describe` hands this to a caller that is not a roqsim process, so it is data."""
    catalog = {row["field"]: row for row in field_catalog()}
    assert "geom_friction" in catalog and "body_mass" in catalog
    for field, row in catalog.items():
        assert row["namespace"] in ("geom", "body", "actuator", "joint"), field
        assert row["write"] in ("live", "needs_setconst"), field
        for key in ("does", "caveats", "measured"):
            assert row[key].strip(), f"{field} has no {key}"
    assert "element-wise MAXIMUM" in catalog["geom_friction"]["caveats"]
    assert "geom_size" in refusal_reasons()
