# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The blackboard keys this package reaches for are the keys the plugins publish.

This package depends on ``roqsim`` and nothing else, deliberately: importing a plugin package
would pull MuJoCo into the behaviour-tree build, which happens before any world is compiled, and
it would not stop at one -- every package whose plugins a scenario can address would follow.

So the coupling is conventional. ``OVERRIDE_KINDS`` names a key prefix as a literal, and the
plugin that publishes under it lives in another package with nothing between them. Rename one side
and nothing complains: no import to fail, no collection error, just an action that raises at run
time in a campaign cell.

A test may import freely. So the convention is checked here, where it costs nothing at run time --
the same shape as the bridge's guard that every service type a plugin declares has a handler, and
for the same reason: a declaration and its consumer in different packages is how they come apart.
"""

from __future__ import annotations

import mujoco
import pytest

from roqsim.context import Entity, SimContext
from scenario_execution_roqsim.access import WorldAccess
from scenario_execution_roqsim.access.in_process import InProcessAccess

SCENE = """
<mujoco model="conventions">
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="crate" pos="0 0 0.4"><freejoint/><geom name="crate_geom" size="0.1" mass="1"/></body>
    <body name="cart" pos="1 0 0.1" mocap="true"><geom name="cart_geom" size="0.1"/></body>
  </worldbody>
</mujoco>
"""


class _Sim:
    def __init__(self, ctx):
        self.context = ctx


def _world():
    ctx = SimContext(config={})
    model = mujoco.MjModel.from_xml_string(SCENE)
    ctx.model, ctx.data = model, mujoco.MjData(model)
    mujoco.mj_forward(model, ctx.data)
    ctx.entities.add(Entity(name="parcel", kind="object", body="crate"))
    ctx.entities.add(Entity(name="cart", kind="object", body="cart", meta={"mocap": True}))
    return ctx


def test_the_sensor_fault_prefix_is_the_one_the_publisher_spells():
    """``roqsim_sensors`` owns the spelling and exports the function that makes it.

    This package cannot call that function -- it would be the dependency the contract forbids --
    so it repeats the prefix as a literal. Repeating a string is fine; repeating it with nothing
    checking that the two agree is what leaves a rename silent.
    """
    live_config = pytest.importorskip("roqsim_sensors.live_config")

    address = "robot.lidar"
    expected = live_config.blackboard_key(address)

    assert f"{WorldAccess.OVERRIDE_KINDS['sensor']}:{address}" == expected


def test_a_model_override_is_reachable_under_the_key_this_package_computes():
    """Behavioural, not textual: the plugin is built and the access layer finds it.

    Stronger than comparing two strings, because it also pins the shape of the address -- an
    instance named by its ``name:``, with no prefix and no namespace folded in.
    """
    from roqsim.plugins.model_override import ModelOverridePlugin

    ctx = _world()
    plugin = ModelOverridePlugin(
        {"overrides": [{"field": "geom_friction", "select": ["crate_geom"], "to": 0.0}]},
        name="grip_fault",
    )
    plugin.configure(ctx)

    access = InProcessAccess(_Sim(ctx))
    access.apply_override("grip_fault", True, kind="model")  # raises if the key is wrong


def test_a_navigator_handle_is_reachable_under_the_key_this_package_computes():
    """The third convention, and the one with a suffix as well as a prefix.

    ``nav:<entity>:handle`` -- the navigator publishes itself under one key and its handle under
    another, and this package wants the second. A rename of either is invisible without this.
    """
    navigator = pytest.importorskip("roqsim_nav.plugins.navigator")

    ctx = _world()
    # Published by the plugin itself, so the key under test is the plugin's and not this test's.
    plugin = navigator.NavigatorPlugin(
        {"goals": [[1.0, 0.0]], "autostart": False, "output": "mocap"}, entity="cart"
    )
    plugin.configure(ctx)

    access = InProcessAccess(_Sim(ctx))
    # Raises when the key is wrong; the route itself is another test's business.
    access.navigate("cart", [(1.0, 0.0)], wait=False)


def test_the_refusal_names_the_key_it_looked_for():
    """When the convention does break, the message has to carry the key.

    A refusal that says only "not found" leaves the reader comparing two packages by eye; the key
    is what turns it into a diff.
    """
    from scenario_execution_roqsim.access import AccessError

    access = InProcessAccess(_Sim(_world()))
    with pytest.raises(AccessError) as caught:
        access.apply_override("nobody", True, kind="sensor")

    assert "sensor_fault:nobody" in str(caught.value)


#: The prefixes the cases above actually exercise. Compared against what the source reaches for,
#: so this list cannot quietly fall behind the code.
COVERED = {"model_override", "sensor_fault", "nav"}


def _consumed_prefixes() -> set:
    """Every blackboard prefix this package reads, taken from its own source.

    Two shapes, and both are here because both are how a key gets written: a literal in the
    ``get`` call, and a value from :attr:`WorldAccess.OVERRIDE_KINDS` for the channels that share
    one method. A third shape would need adding -- which the test below is what makes noticeable.
    """
    import importlib
    import re
    from pathlib import Path

    # From the imported module, so the scan follows the code wherever it is installed from
    # rather than guessing a layout.
    root = Path(importlib.import_module(WorldAccess.__module__).__file__).parent
    found = set(WorldAccess.OVERRIDE_KINDS.values())
    for path in sorted(root.glob("*.py")):
        for literal in re.findall(r'blackboard\.get\(\s*f?"([a-z_]+):', path.read_text()):
            found.add(literal)
    return found


def test_every_key_this_package_reads_is_pinned_by_a_case_here():
    """The reason this file can be trusted: it cannot fall behind the code it guards.

    An action that reaches a new blackboard key is a new conventional coupling to a plugin package
    this one does not import, and nothing else in the build will notice a rename of either side.
    Discovering that requires knowing this file exists, so instead the file discovers the action:
    a prefix with no case fails here, naming itself.
    """
    consumed = _consumed_prefixes()

    missing = consumed - COVERED
    assert not missing, (
        f"these blackboard prefixes are read by this package and pinned by nothing: "
        f"{sorted(missing)}. Each is a key a plugin in ANOTHER package publishes, with no import "
        "between them -- so a rename on either side is silent until a campaign cell fails. Add a "
        "case above that builds the publishing plugin and reaches it through the access layer, "
        "then list the prefix in COVERED."
    )
    stale = COVERED - consumed
    assert not stale, (
        f"COVERED lists prefixes nothing reads any more: {sorted(stale)}. Drop the case, or the "
        "guard is pinning a convention that no longer has two sides."
    )
