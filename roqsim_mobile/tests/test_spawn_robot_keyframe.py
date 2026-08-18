"""spawn_robot strips a model's keyframes before attach, so composing a robot into a world does not
emit MuJoCo's "nkey: parent has 0, child has 1" attach-conflict warning (see _strip_keyframes)."""

from __future__ import annotations

import warnings

import mujoco

from roqsim_mobile.plugins.spawn_robot import _strip_keyframes

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
