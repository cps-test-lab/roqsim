# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Frankie carries the panda model's Franka Hand.

``frankie.xml`` is generated from ``panda.xml`` by ``external/convert/build_frankie_mjcf.py``, which
needs the base's source meshes and so cannot run here. What the generator carries over is checked
instead: every actuator the panda declares, the hand's tendon and its finger joints are the same in
both models, so a change to the panda's hand that frankie.xml was not regenerated for fails here.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.models import apply_assets, resolve_model

#: MuJoCo's XML writer keeps six significant digits, and frankie.xml is written by it while
#: panda.xml states the finger servo's command gain, kp * 0.04 / 255, to eight.
RTOL = 1e-5

ACTUATOR_FIELDS = ("gainprm", "biasprm", "ctrlrange", "forcerange", "gear", "trntype", "biastype")
JOINT_FIELDS = ("range", "armature", "damping", "frictionloss", "type")
FINGER_JOINTS = ("finger_joint1", "finger_joint2")


def _compiled(name: str) -> mujoco.MjModel:
    asset = resolve_model(name)
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    # Frankie's wheel contact pairs name the world's floor.
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    return spec.compile()


@pytest.fixture(scope="module")
def models():
    return _compiled("panda"), _compiled("frankie")


def _close(a, b) -> bool:
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=RTOL, atol=0.0)


def test_every_panda_actuator_is_frankies(models):
    panda, frankie = models
    for i in range(panda.nu):
        name = panda.actuator(i).name
        theirs, ours = panda.actuator(name), frankie.actuator(name)
        for field in ACTUATOR_FIELDS:
            assert _close(getattr(theirs, field), getattr(ours, field)), (name, field)


def test_the_hands_tendon_and_finger_joints_are_frankies(models):
    panda, frankie = models

    def split(model):
        t = model.tendon("split").id
        adr, num = int(model.tendon_adr[t]), int(model.tendon_num[t])
        joints = [model.joint(int(j)).name for j in model.wrap_objid[adr : adr + num]]
        return joints, model.wrap_prm[adr : adr + num]

    (joints_p, coef_p), (joints_f, coef_f) = split(panda), split(frankie)
    assert joints_p == joints_f
    assert _close(coef_p, coef_f)
    for joint in FINGER_JOINTS:
        for field in JOINT_FIELDS:
            mine = getattr(panda.joint(joint), field)
            theirs = getattr(frankie.joint(joint), field)
            assert _close(mine, theirs), (joint, field)
