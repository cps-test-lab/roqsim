"""A ray-cast sensor casts through the backend it or its world chose, and the CPU when neither does."""

from __future__ import annotations

import numpy as np
import pytest
from roqsim_sensors.plugins.lidar_common import RayCastSensorPlugin

from roqsim import raycast
from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import PluginError


def _world(sim=None, lidar=None) -> dict:
    return {
        "sim": {"pacing": "asap", **(sim or {})},
        "components": [
            {
                "spawn_sensor": {"model": "sick_tim571", "frame_id": "scan_link"},
                "name": "scanner",
                "components": [{"lidar": dict(lidar or {})}],
            }
        ],
    }


def _lidar(engine):
    return next(p for p in engine.plugins if isinstance(p, RayCastSensorPlugin))


def test_a_2d_lidar_under_the_default_auto_casts_on_the_cpu_and_its_choice_is_recorded():
    """A 2D scanner's few hundred rays are cheaper on the CPU, whatever is installed."""
    engine = Engine(load_config_from_dict(_world()))
    engine.ctx.seed = 1
    with engine:
        lidar = _lidar(engine)
        assert isinstance(lidar._raycaster, raycast.CpuRaycaster)
        (condition,) = [c for c in engine.ctx.conditions if c["code"] == "raycaster"]
        assert condition["source"] == lidar.address
        assert condition["level"] == "info"
        assert condition["details"]["requested"] == "auto"
        assert condition["details"]["backend"] == "cpu"
        assert condition["details"]["rays"] == lidar.num_rays
        assert condition["details"]["reason"]
        engine.reset()
        engine.step()
        assert lidar.latest is not None
    assert engine.ctx.raycast is None  # closed with the world


def test_cpu_named_on_the_lidar_is_recorded_as_named():
    engine = Engine(load_config_from_dict(_world(lidar={"raycaster": "cpu"})))
    engine.ctx.seed = 1
    with engine:
        (condition,) = [c for c in engine.ctx.conditions if c["code"] == "raycaster"]
        assert condition["details"] == {
            "requested": "cpu",
            "backend": "cpu",
            "rays": _lidar(engine).num_rays,
            "reason": "named",
        }


def test_a_lidar_choice_overrides_the_world_and_an_unknown_one_fails_naming_the_lidar():
    engine = Engine(load_config_from_dict(_world({"raycaster": "cpu"}, {"raycaster": "nope"})))
    engine.ctx.seed = 1
    with pytest.raises(PluginError, match=r"lidar scanner\.lidar: unknown raycaster 'nope'"):
        engine.setup()


def test_the_raycaster_key_is_refused_as_a_live_write():
    assert "raycaster" in RayCastSensorPlugin.REFUSED_WRITES


class _Fixed(raycast.Raycaster):
    """A backend that returns the distances it is given, to set a frame's hits by hand."""

    def __init__(self, dist):
        super().__init__()
        self.dist = dist

    def cast(self, model, data, origin, dirs, *, cutoff, out=None, **kw):
        out.dist[:] = self.dist
        out.geomid[:] = np.where(self.dist >= 0, 0, -1)
        return out


def test_a_ray_that_hits_otherwise_changes_its_own_noise_and_no_other():
    engine = Engine(load_config_from_dict(_world(lidar={"range_stddev": 0.05})))
    engine.ctx.seed = 1
    with engine:
        lidar = _lidar(engine)
        dist = np.full(lidar.num_rays, 3.0)
        lidar._raycaster = _Fixed(dist.copy())
        lidar.capture(engine.ctx)
        first = np.asarray(lidar.latest.ranges, dtype=float).copy()
        other = dist.copy()
        other[10] = -1.0  # one ray returns nothing this time
        lidar._raycaster = _Fixed(other)
        lidar.capture(engine.ctx)
        second = np.asarray(lidar.latest.ranges, dtype=float)
    changed = np.nonzero(~np.isclose(first, second, equal_nan=True))[0]
    assert list(changed) == [10]
    assert np.std(first - 3.0) > 0.01  # the noise is on
