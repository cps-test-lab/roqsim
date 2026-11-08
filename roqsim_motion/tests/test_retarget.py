"""Retargeting tables: parsed, checked against the model, and mapping rotations onto its joints."""

from __future__ import annotations

import math

import mujoco
import pytest
from minimal_humanoid import TABLE, mjcf

from roqsim_motion import quat as Q
from roqsim_motion.retarget import TableError, parse_table


def test_a_table_maps_canonical_rotations_onto_the_models_joints_and_back():
    table = parse_table(TABLE)
    angles = {"spine": [0.3, -0.1, 0.4], "shoulder_l": [-0.8, 0.4, 0.2], "elbow_l": [-1.1], "wrist_r": [0.5, -0.3, 0.2]}
    local = {j: Q.from_euler("".join(d.axis for d in table.joints[j]), a) for j, a in angles.items()}
    out = table.angles(local)
    for joint, values in angles.items():
        for dof, value in zip(table.joints[joint], values, strict=True):
            assert math.isclose(out[dof.joint], value, abs_tol=1e-9)


def test_a_table_is_checked_against_the_model():
    model = mujoco.MjModel.from_xml_string(mjcf())
    parse_table(TABLE).check(model)
    bad = {**TABLE, "joints": {**TABLE["joints"], "elbow_l": [{"y": "left_elbow_pitch_joint"}]}}
    with pytest.raises(TableError, match="joint 'left_elbow_pitch_joint'.*Its joints: base_free"):
        parse_table(bad).check(model)


def test_a_table_refuses_what_is_no_canonical_joint_or_key():
    with pytest.raises(TableError, match="'neck' is no canonical joint"):
        parse_table({**TABLE, "joints": {"neck": [{"z": "a"}]}})
    with pytest.raises(TableError, match=r"unknown key\(s\) \['feet'\]"):
        parse_table({**TABLE, "feet": {}})
