# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""``actuators:`` -- the merge, the refusals, and that declaring nothing changes nothing.

The physics these gains produce is tested against a real arm in
``roqsim_manipulation_assets/tests/test_ur5e_actuator_gains.py``; this file is about the table and
the messages. Both matter: a refusal that does not name the replacement is a refusal a world author
has to bisect a file to act on.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.actuators import (
    apply_gravity_compensation,
    resolve,
    uses_impedance,
    validate_override,
)
from roqsim.plugin import PluginError

#: A two-joint arm with a position servo, in the shape the real models use: gains on a `<default>`
#: class rather than the actuator, which is where every shipped arm actually keeps them.
_MODEL = """
<mujoco model="probe">
  <default>
    <default class="servo">
      <general gaintype="fixed" biastype="affine" gainprm="2000" biasprm="0 -2000 -400"
               forcerange="-120 120" ctrlrange="-3.14 3.14"/>
    </default>
  </default>
  <worldbody>
    <body name="a">
      <joint name="a_joint"/><geom size="0.1"/>
      <body name="b" pos="0 0 .3">
        <joint name="b_joint"/><geom size="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general class="servo" name="a_act" joint="a_joint"/>
    <general class="servo" name="b_act" joint="b_joint"/>
  </actuator>
</mujoco>
"""

#: A tendon transmission, which has no joint position and therefore no joint stiffness.
_TENDON = """
<mujoco model="tendon_probe">
  <worldbody>
    <body name="a"><joint name="s" type="slide" axis="1 0 0"/><geom size="0.1"/>
      <site name="p1"/></body>
  </worldbody>
  <tendon><spatial name="t"><site site="p1"/><site site="p1"/></spatial></tendon>
  <actuator><general name="grip" tendon="t"/></actuator>
</mujoco>
"""


def _spec(xml: str = _MODEL):
    return mujoco.MjSpec.from_string(xml)


def _by_name(rows):
    return {row.name: row for row in rows}


# -- the merge ---------------------------------------------------------------------------------


def test_shared_keys_apply_to_every_actuator():
    rows = _by_name(resolve(_spec(), {"control": "impedance", "stiffness": 5.0}, model_name="probe"))
    assert [r.control for r in rows.values()] == ["impedance", "impedance"]
    assert [r.stiffness for r in rows.values()] == [5.0, 5.0]
    assert {r.source for r in rows.values()} == {"shared"}


def test_each_entry_sits_on_top_of_the_shared_keys():
    rows = _by_name(
        resolve(
            _spec(),
            {"control": "impedance", "stiffness": 5.0, "damping": 1.0, "each": {"b_act": {"stiffness": 9.0}}},
            model_name="probe",
        )
    )
    assert rows["a_act"].stiffness == 5.0 and rows["a_act"].source == "shared"
    # Only the key the entry named changes: the damping is still the block's.
    assert rows["b_act"].stiffness == 9.0 and rows["b_act"].damping == 1.0
    assert rows["b_act"].source == "each"


def test_an_each_entry_may_choose_a_different_law_without_inheriting_the_others_gains():
    """The shared gains belong to the shared law. An entry that picks another law is not stating a
    gain that law cannot read -- it is declining the block's, which is the ordinary case."""
    rows = _by_name(
        resolve(
            _spec(),
            {"control": "impedance", "stiffness": 2.0, "damping": 0.02,
             "each": {"b_act": {"control": "position", "p": 2000, "d": 500}}},
            model_name="probe",
        )
    )
    assert rows["b_act"].control == "position"
    assert (rows["b_act"].p, rows["b_act"].d) == (2000.0, 500.0)
    assert rows["b_act"].stiffness is None


def test_absent_key_keeps_the_model_value():
    rows = _by_name(resolve(_spec(), {"control": "position", "p": 10.0}, model_name="probe"))
    # `d`, the effort limit and the ctrlrange were not stated, so they are still the model's.
    assert rows["a_act"].d == 400.0
    assert rows["a_act"].effort_limit == 120.0
    assert rows["a_act"].ctrlrange == (-3.14, 3.14)


def test_the_table_describes_a_model_nobody_overrode():
    rows = _by_name(resolve(_spec(), None, model_name="probe"))
    assert rows["a_act"].control == "position"
    assert (rows["a_act"].p, rows["a_act"].d) == (2000.0, 400.0)
    assert rows["a_act"].source == "model"
    assert rows["a_act"].joint == "a_joint"


def test_no_override_leaves_the_compiled_model_identical():
    """The invariant is the COMPILED model, not the MJCF text.

    ``MjSpec.to_xml()`` reorders the asset list once any actuator attribute has been read, so a text
    comparison reports a difference that does not exist in the model MuJoCo actually runs. Comparing
    what is compiled is both the honest check and the one a world cares about.
    """
    untouched = _spec().compile()
    probed = _spec()
    resolve(probed, None, model_name="probe")
    got = probed.compile()
    for field in (
        "actuator_gaintype", "actuator_biastype", "actuator_gainprm", "actuator_biasprm",
        "actuator_forcerange", "actuator_ctrlrange", "actuator_forcelimited",
        "actuator_ctrllimited", "actuator_trntype", "body_gravcomp",
    ):
        assert np.array_equal(getattr(untouched, field), getattr(got, field)), field


@pytest.mark.parametrize(
    "control, gainprm, biasprm",
    [
        ("position", 7.0, [0.0, -7.0, -3.0]),
        ("impedance", 7.0, [0.0, -7.0, -3.0]),
        ("velocity", 3.0, [0.0, 0.0, -3.0]),
        ("effort", 1.0, [0.0, 0.0, 0.0]),
    ],
)
def test_each_law_compiles_to_its_mujoco_form(control, gainprm, biasprm):
    gains = {"position": {"p": 7.0, "d": 3.0}, "impedance": {"stiffness": 7.0, "damping": 3.0},
             "velocity": {"d": 3.0}, "effort": {}}[control]
    spec = _spec()
    resolve(spec, {"control": control, "ctrlrange": [-5.0, 5.0], **gains}, model_name="probe")
    m = spec.compile()
    assert m.actuator_gainprm[0][0] == pytest.approx(gainprm)
    assert list(m.actuator_biasprm[0][:3]) == pytest.approx(biasprm)


def test_stale_parameters_are_cleared_on_a_control_change():
    """A law's parameters are written in full, never patched over the previous law's.

    MuJoCo's own ``set_to_*`` helpers leave the old vector in place -- ``set_to_motor`` after
    ``set_to_velocity`` keeps the previous ``biasprm`` -- so a joint asked for torque would still
    carry the velocity servo's damping term.
    """
    spec = _spec()
    resolve(spec, {"control": "effort", "ctrlrange": [-120.0, 120.0]}, model_name="probe")
    m = spec.compile()
    assert list(m.actuator_biasprm[0][:3]) == [0.0, 0.0, 0.0]
    assert m.actuator_biastype[0] == mujoco.mjtBias.mjBIAS_NONE


# -- the refusals ------------------------------------------------------------------------------


def test_a_joint_name_is_refused_naming_the_actuator_that_drives_it():
    with pytest.raises(PluginError) as exc:
        resolve(_spec(), {"each": {"a_joint": {"p": 1.0}}}, model_name="probe")
    assert "is a joint, not an actuator" in str(exc.value)
    assert "'a_act'" in str(exc.value)


def test_an_unknown_name_lists_the_models_actuators_and_the_catalog_command():
    with pytest.raises(PluginError) as exc:
        resolve(_spec(), {"each": {"c_act": {"p": 1.0}}}, model_name="probe")
    message = str(exc.value)
    assert "a_act, b_act" in message
    assert "roqsim catalog model probe" in message


def test_every_bad_entry_is_named_in_one_error():
    """One run should find every mistake in the block, the way `instantiate_plugins` aggregates."""
    with pytest.raises(PluginError) as exc:
        resolve(_spec(), {"each": {"nope": {"p": 1.0}, "also_nope": {"p": 1.0}}}, model_name="probe")
    assert "nope" in str(exc.value) and "also_nope" in str(exc.value)


def test_a_command_unit_change_without_ctrlrange_is_refused():
    with pytest.raises(PluginError) as exc:
        resolve(_spec(), {"control": "effort"}, model_name="probe")
    message = str(exc.value)
    assert "N*m" in message and "ctrlrange" in message


def test_position_to_impedance_is_not_refused():
    """Both command a joint position, so the model's ctrlrange still means what it says. This is the
    common case -- a paper's compliance on an arm the substrate ships as a position servo."""
    rows = resolve(_spec(), {"control": "impedance", "stiffness": 2.0}, model_name="probe")
    assert {row.control for row in rows} == {"impedance"}


def test_a_shared_gain_on_a_models_own_tendon_actuator_is_refused():
    with pytest.raises(PluginError) as exc:
        resolve(_spec(_TENDON), {"control": "impedance", "stiffness": 2.0}, model_name="grip_probe")
    assert "tendon" in str(exc.value)


def test_a_gain_the_models_own_law_does_not_read_is_refused():
    """Nothing names a control, so the law is the model's -- which only the model can say."""
    with pytest.raises(PluginError) as exc:
        resolve(_spec(), {"stiffness": 5.0}, model_name="probe")
    assert "keeps the model's control: position" in str(exc.value)


# -- the shape, checked before anything is compiled ----------------------------------------------


def test_nothing_declared_is_no_error():
    assert validate_override(None) == []


@pytest.mark.parametrize(
    "key, replacement",
    [("kp", "'p'"), ("kv", "'d'"), ("kd", "'d'"), ("forcerange", "'effort_limit'")],
)
def test_a_mujoco_spelling_is_refused_naming_its_replacement(key, replacement):
    errors = validate_override({key: 1.0})
    assert errors and replacement in errors[0]


@pytest.mark.parametrize("control, replacement", [("motor", "effort"), ("pd", "impedance")])
def test_a_mujoco_actuator_type_is_refused_naming_the_command_interface(control, replacement):
    assert any(replacement in e for e in validate_override({"control": control}))


def test_a_gain_meaningless_for_the_control_is_refused():
    assert any("not a gain of control: effort" in e for e in validate_override(
        {"control": "effort", "stiffness": 5.0}))


def test_an_each_entry_is_judged_against_the_law_it_inherits():
    errors = validate_override({"control": "effort", "each": {"a_act": {"stiffness": 5.0}}})
    assert any("not a gain of control: effort" in e for e in errors)


def test_an_unknown_key_is_refused():
    assert any("not a setting" in e for e in validate_override({"stifness": 5.0}))


def test_a_gain_of_the_wrong_type_is_refused():
    assert any("must be float" in e for e in validate_override({"control": "position", "p": "soft"}))


def test_every_gain_declares_its_unit():
    """A paper states "kp = 2.0" with no units, so a value wrong by a factor of a thousand looks
    exactly like a right one. The unit is published through `roqsim plugins describe`."""
    from roqsim.actuators import GAIN_SCHEMA

    for name in ("p", "d", "stiffness", "damping", "effort_limit"):
        assert GAIN_SCHEMA[name].unit, name


# -- the body-level half of impedance ------------------------------------------------------------


def test_impedance_is_the_only_law_that_needs_a_body_term():
    assert uses_impedance(resolve(_spec(), {"control": "impedance", "stiffness": 1.0},
                                  model_name="probe"))
    assert not uses_impedance(resolve(_spec(), {"control": "position", "p": 1.0},
                                      model_name="probe"))


def test_gravity_compensation_covers_every_body_of_the_model():
    spec = _spec()
    assert apply_gravity_compensation(spec) == 2
    m = spec.compile()
    assert m.ngravcomp == 2
