"""A sensor's ``fault:`` block: what may be written at run time, and what must not be.

The physics channel has been switchable mid-run since ``set_model_override``; the report channel had
no trigger at all, so ``dropout_percent`` was fixed for a whole run and a lidar that fails *halfway
down a corridor* could not be expressed. These tests pin the three properties that make the new
switch trustworthy rather than merely present:

* a key that is not read per frame is **refused by name**, because writing one takes effect nowhere
  while reading back as though it had (the ``geom_size`` lesson, applied to sensors);
* a fault that changes nothing reports ``no_effect``, so a trial cannot record an unfaulted outcome
  under a faulted label;
* a fault does not survive ``reset``, so trial 1's fault cannot silently degrade trials 2..N of the
  same process.
"""

from __future__ import annotations

import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugins.model_override import LANDED, NO_EFFECT, UNTESTED
from roqsim_sensors.live_config import blackboard_key
from roqsim_sensors.plugins.lidar import LidarPlugin

_SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="6 6 .1"/>
    <body pos="2 0 .5"><geom type="box" size=".3 .3 .5"/></body>
    <body name="base" pos="0 0 .2">
      <geom type="cylinder" size=".2 .2"/>
      <site name="scan" pos="0 0 .3"/>
    </body>
  </worldbody>
</mujoco>
"""


def _world(tmp_path, lidar_cfg):
    scene = tmp_path / "s.xml"
    scene.write_text(_SCENE)
    cfg = {"site": "scan", "rays": 64}
    cfg.update(lidar_cfg)
    # `name:` is a SIBLING of the plugin ref, not a key inside its config -- it labels the entry, and
    # the label is what an address is built from. Written inside the config it is ignored, and the
    # entry falls back to its ref.
    return {
        "sim": {"world": str(scene), "seed": 3},
        "components": [{"lidar": cfg, "name": "front"}],
    }


def _engine(world):
    engine = Engine(load_config_from_dict(world))
    engine.setup()
    engine.reset()
    return engine


def _handle(engine, address="front"):
    return engine.ctx.blackboard.get(blackboard_key(address))


# -- the allowlist -----------------------------------------------------------------------------


def test_a_key_read_per_frame_is_writable():
    assert LidarPlugin.refusal_reason("dropout_percent") == ""
    assert set(LidarPlugin.live_writable()) >= {"range_stddev", "dropout_percent", "max_range"}


def test_rays_is_refused_by_name_with_the_reason():
    """The one someone reaches for first, and the one that corrupts a fixed-length scan."""
    reason = LidarPlugin.refusal_reason("rays")
    assert reason
    assert "length" in reason, reason


def test_geometry_keys_are_refused():
    for key in ("angle_min", "angle_max", "site", "frame_id"):
        assert LidarPlugin.refusal_reason(key), f"{key} should be refused"


def test_an_undeclared_key_is_refused_and_the_message_lists_what_is_writable():
    reason = LidarPlugin.refusal_reason("nonesuch")
    assert "not declared live-writable" in reason
    assert "dropout_percent" in reason


def test_a_refused_key_in_a_fault_block_fails_validation(tmp_path):
    with pytest.raises(Exception) as err:
        _engine(_world(tmp_path, {"fault": {"rays": 8}}))
    assert "rays" in str(err.value)


def test_a_faulted_value_is_checked_by_the_devices_own_rules(tmp_path):
    """A fault may not set what the nominal config could not: 400% dropout is still nonsense."""
    with pytest.raises(Exception) as err:
        _engine(_world(tmp_path, {"fault": {"dropout_percent": 400.0}}))
    assert "dropout_percent" in str(err.value)


def test_an_empty_fault_block_is_refused(tmp_path):
    with pytest.raises(Exception) as err:
        _engine(_world(tmp_path, {"fault": {}}))
    assert "empty" in str(err.value).lower()


# -- applying and restoring --------------------------------------------------------------------


def test_apply_then_restore_returns_the_nominal_value_exactly(tmp_path):
    engine = _engine(_world(tmp_path, {"dropout_percent": 2.0, "fault": {"dropout_percent": 60.0}}))
    try:
        plugin = engine.ctx.blackboard  # handle is the public surface; read through it
        handle = _handle(engine)
        assert handle is not None, "a sensor with a fault: block must publish a handle"
        assert handle.is_active() is False

        handle.set_active(True)
        assert handle.is_active() is True
        assert handle.read_state().verified == LANDED

        handle.set_active(False)
        assert handle.is_active() is False
        # Exactly, not approximately: a restore that drifted would leave every later trial slightly
        # degraded in a way nothing reports.
        assert _dropout(engine) == 2.0
        del plugin
    finally:
        engine.shutdown()


def _dropout(engine):
    for p in engine.plugins:
        if isinstance(p, LidarPlugin):
            return p.dropout_percent
    raise AssertionError("no lidar in the world")


def test_a_fault_that_changes_nothing_reports_no_effect(tmp_path):
    """The failure this exists to catch: a row claiming a fault that never happened."""
    engine = _engine(
        _world(tmp_path, {"dropout_percent": 60.0, "fault": {"dropout_percent": 60.0}})
    )
    try:
        handle = _handle(engine)
        handle.set_active(True)
        assert handle.read_state().verified == NO_EFFECT
    finally:
        engine.shutdown()


def test_a_restore_is_untested_rather_than_verified(tmp_path):
    engine = _engine(_world(tmp_path, {"fault": {"dropout_percent": 60.0}}))
    try:
        handle = _handle(engine)
        handle.set_active(True)
        handle.set_active(False)
        assert handle.read_state().verified == UNTESTED
    finally:
        engine.shutdown()


def test_switching_to_the_state_it_is_already_in_does_nothing(tmp_path):
    engine = _engine(_world(tmp_path, {"fault": {"dropout_percent": 60.0}}))
    try:
        handle = _handle(engine)
        handle.set_active(True)
        changes = handle.read_state().changes
        handle.set_active(True)
        assert handle.read_state().changes == changes
    finally:
        engine.shutdown()


# -- it actually degrades the scan ---------------------------------------------------------------


def _valid_count(engine, endpoint, steps=120):
    for _ in range(steps):
        engine.step()
    ranges = np.asarray(endpoint.read().ranges, dtype=float)
    return int(np.isfinite(ranges).sum())


def test_the_fault_reaches_the_published_scan(tmp_path):
    """Guards against every test above passing while the write reached nothing that is read."""
    engine = _engine(_world(tmp_path, {"dropout_percent": 0.0, "fault": {"dropout_percent": 90.0}}))
    try:
        endpoint = next(
            e for e in engine.ctx.interface.all() if e.direction == "out" and e.name == "scan"
        )
        before = _valid_count(engine, endpoint)
        _handle(engine).set_active(True)
        after = _valid_count(engine, endpoint)
        assert after < before, f"dropout 90% left {after} valid returns against {before} nominal"
    finally:
        engine.shutdown()


# -- it does not leak across trials ---------------------------------------------------------------


def test_reset_puts_the_sensor_back_to_nominal(tmp_path):
    """One process serves several trials; a fault surviving reset makes the control cell faulted."""
    engine = _engine(_world(tmp_path, {"dropout_percent": 2.0, "fault": {"dropout_percent": 60.0}}))
    try:
        _handle(engine).set_active(True)
        assert _dropout(engine) == 60.0
        engine.reset()
        assert _dropout(engine) == 2.0
        assert _handle(engine).is_active() is False
    finally:
        engine.shutdown()


# -- wiring ---------------------------------------------------------------------------------------


def test_a_sensor_with_no_fault_block_publishes_nothing(tmp_path):
    """A service that always replied 'nothing configured' would make a typo look like a real call."""
    engine = _engine(_world(tmp_path, {}))
    try:
        assert _handle(engine) is None
        assert not [e for e in engine.ctx.interface.all() if e.name == "override"]
    finally:
        engine.shutdown()


def test_the_endpoints_are_named_by_address_with_slashes(tmp_path):
    """A dot is legal in an address and not in a ROS name, so the translation happens once."""
    engine = _engine(_world(tmp_path, {"fault": {"dropout_percent": 60.0}}))
    try:
        names = {
            e.name: e.backend["ros2"].get("name") or e.backend["ros2"].get("topic")
            for e in engine.ctx.interface.all()
            if e.name in ("override", "override_state", "override_verified")
        }
        assert names, "a sensor with a fault: block must offer the switch"
        assert all("." not in n for n in names.values()), names
        assert names["override"].endswith("front/override")
    finally:
        engine.shutdown()


def test_a_sensor_owned_by_a_robot_is_addressed_through_it(tmp_path):
    """The case the address exists for: two robots may each carry a lidar, and a bare 'lidar' names
    neither. The handle key and the service name both follow the full path."""
    scene = tmp_path / "s.xml"
    scene.write_text(_SCENE)
    rover = tmp_path / "rover.xml"
    rover.write_text(
        """
<mujoco>
  <worldbody>
    <body name="chassis" pos="0 0 .2">
      <geom type="cylinder" size=".2 .2"/>
      <site name="rscan" pos="0 0 .3"/>
    </body>
  </worldbody>
</mujoco>
"""
    )
    world = {
        "sim": {"world": str(scene), "seed": 3},
        "components": [
            {
                "spawn_model": {"model": str(rover), "free": False},
                "name": "rover",
                "components": [
                    {
                        "lidar": {
                            "site": "rscan",
                            "rays": 64,
                            "exclude_body": "",
                            "fault": {"dropout_percent": 60.0},
                        },
                        "name": "lidar",
                    }
                ],
            }
        ],
    }
    engine = _engine(world)
    try:
        assert _handle(engine, "rover.lidar") is not None
        override = next(e for e in engine.ctx.interface.all() if e.name == "override")
        assert override.backend["ros2"]["name"] == "rover/lidar/override"
    finally:
        engine.shutdown()
