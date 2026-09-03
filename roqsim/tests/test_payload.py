"""payload: carried mass on a robot's body.

Stated in the world and applied after compile, so what is worth pinning is what a "does it load"
check cannot see: that the mass write reaches the dynamics, that the zero level is a true no-op, and
that an offset payload is refused rather than approximated.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.context import Entity, SimContext
from roqsim.plugins.payload import PayloadPlugin

# A single free body. Gravity is off so that qacc IS force/mass: the mass assertion is then
# arithmetic rather than the outcome of a constraint solve.
SCENE = """
<mujoco model="payload_test">
  <option timestep="0.002" gravity="0 0 0" density="1.225" viscosity="1.8e-5"/>
  <worldbody>
    <body name="cart" pos="0 0 1">
      <freejoint name="cart_free"/>
      <geom name="cart" type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _ctx(scene=SCENE, *, seed=7, body="cart"):
    model = mujoco.MjModel.from_xml_string(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctx = SimContext(config={})
    ctx.model, ctx.data = model, data
    ctx.seed = seed
    # The entity a spawn plugin would register: name, root body, and the base joint that is the
    # general fallback when a model does not call its root `base_link`.
    ctx.entities.add(
        Entity(name="robot", kind="robot", body=body, meta={"base_joint": "cart_free"})
    )
    return ctx


def _plugin(cls, config, *, name="p", entity="robot"):
    """A configured plugin. Config errors are raised rather than returned."""
    plugin = cls(config, name=name, entity=entity)
    errors = plugin.validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    return plugin


def test_payload_adds_mass_to_the_body():
    ctx = _ctx()
    _plugin(PayloadPlugin, {"mass": 0.5}).configure(ctx)
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "cart")
    assert float(ctx.model.body_mass[bid]) == pytest.approx(1.5, abs=1e-9)


def test_the_added_mass_reaches_the_dynamics():
    """mj_setConst is what makes a body_mass write take effect rather than sit in an array."""
    accelerations = {}
    for mass in (0.0, 1.0):
        ctx = _ctx()
        _plugin(PayloadPlugin, {"mass": mass}).configure(ctx)
        ctx.data.xfrc_applied[
            mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "cart"), 0
        ] = 1.0
        mujoco.mj_forward(ctx.model, ctx.data)
        accelerations[mass] = float(ctx.data.qacc[0])
    # 1 N on 1 kg, then on 2 kg.
    assert accelerations[0.0] == pytest.approx(1.0, rel=1e-6)
    assert accelerations[1.0] == pytest.approx(0.5, rel=1e-6)


def test_zero_payload_leaves_the_model_untouched():
    # The unloaded cell of a sweep must be identical to a world that never declared a payload,
    # otherwise the sweep's baseline is its own separate configuration.
    loaded, bare = _ctx(), _ctx()
    _plugin(PayloadPlugin, {"mass": 0.0}).configure(loaded)
    assert np.array_equal(loaded.model.body_mass, bare.model.body_mass)


def test_payload_resolves_the_body_owning_the_base_joint():
    """The entity's registered body may be absent from the compiled model; the base joint is not."""
    ctx = _ctx(body="base_link")
    _plugin(PayloadPlugin, {"mass": 0.5}).configure(ctx)
    bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, "cart")
    assert float(ctx.model.body_mass[bid]) == pytest.approx(1.5, abs=1e-9)


def test_refuses_an_offset_payload():
    # An offset payload shifts the centre of mass and adds a parallel-axis term. Approximating it as
    # a centred point mass would report a result for a body nobody configured.
    with pytest.raises(ValueError, match="offset"):
        _plugin(PayloadPlugin, {"mass": 0.5, "offset": [0.1, 0.0, 0.0]})


def test_requires_a_mass():
    with pytest.raises(ValueError, match="mass"):
        _plugin(PayloadPlugin, {})


def test_names_the_entity_it_could_not_find():
    # The entity comes from the entry this one is nested under, so an absent one is an owner that
    # registered nothing -- not a config key pointing somewhere else.
    with pytest.raises(RuntimeError, match="no entity named"):
        _plugin(PayloadPlugin, {"mass": 0.5}, entity="absent").configure(_ctx())
