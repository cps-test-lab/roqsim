"""spawn_robot strips a model's keyframes before attach, so composing a robot into a world does not
emit MuJoCo's "nkey: parent has 0, child has 1" attach-conflict warning (see _strip_keyframes) --
and takes the base's resting height out of that keyframe first (see _keyframe_base_z), because it is
the only place the model states one.
"""

from __future__ import annotations

import warnings

import mujoco

from roqsim_mobile.plugins.spawn_robot import _keyframe_base_z, _strip_keyframes

# A minimal robot model that, like the LimX Oli, reserves a keyframe (``<size nkey>`` + ``<keyframe>``).
# The reservation -- not the key element alone -- is what makes MuJoCo warn on attach.
_MODEL = """<mujoco>
  <size nkey="1"/>
  <worldbody><body name="base_link"><freejoint/><geom type="box" size=".1 .1 .1"/></body></worldbody>
  <keyframe><key name="home" qpos="0 0 0.5 1 0 0 0"/></keyframe>
</mujoco>"""


def _child() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(_MODEL)


def _attach_warnings(child: mujoco.MjSpec) -> list[str]:
    parent = mujoco.MjSpec()
    frame = parent.worldbody.add_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parent.attach(child, prefix="r_", frame=frame)
        parent.compile()
    return [str(w.message) for w in caught]


def test_strip_clears_keys_and_reservation():
    spec = _child()
    assert len(spec.keys) == 1 and spec.nkey == 1
    _strip_keyframes(spec)
    assert len(spec.keys) == 0 and spec.nkey == 0
    spec.compile()  # still a valid model


def test_unstripped_attach_warns():
    # Guardrail: without the strip the warning really does fire, so the test below is meaningful.
    assert any("nkey" in m for m in _attach_warnings(_child()))


def test_stripped_attach_is_warning_free():
    child = _child()
    _strip_keyframes(child)
    msgs = _attach_warnings(child)
    assert not any("nkey" in m or "Attach conflict" in m for m in msgs), msgs


def test_base_height_is_read_from_the_keyframe():
    assert _keyframe_base_z(_child(), "base_free") is None  # this model's free joint is unnamed
    named = mujoco.MjSpec.from_string(_MODEL.replace("<freejoint/>", '<freejoint name="base_free"/>'))
    assert _keyframe_base_z(named, "base_free") == 0.5


def test_a_model_without_a_keyframe_states_no_height():
    """None, not 0.0: "the model says nothing" and "the model says zero" are different answers, and
    only the first may fall back to the compiled qpos0."""
    plain = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="base_link">'
        '<freejoint name="base_free"/><geom type="box" size=".1 .1 .1"/>'
        "</body></worldbody></mujoco>"
    )
    assert _keyframe_base_z(plain, "base_free") is None


def test_the_address_is_resolved_not_assumed():
    """A keyframe's qpos is the whole robot's. With a hinge declared before the base free joint, the
    base z is qpos[3], and reading qpos[2] would spawn the robot at a wheel angle."""
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody>'
        '<body name="arm"><joint name="j" type="hinge" axis="0 0 1"/>'
        '<geom type="box" size=".1 .1 .1"/></body>'
        '<body name="base_link"><freejoint name="base_free"/>'
        '<geom type="box" size=".1 .1 .1"/></body>'
        "</worldbody>"
        '<keyframe><key name="home" qpos="0.7  0 0 0.5 1 0 0 0"/></keyframe></mujoco>'
    )
    assert _keyframe_base_z(spec, "base_free") == 0.5
