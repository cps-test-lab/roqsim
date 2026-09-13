"""Sensor plugin: 2D lidar via batched ray-casting (:func:`roqsim.raycast.cast`, no GL).

Ported from our earlier in-house nav prototype's ``Lidar``. Casts a horizontal fan from the robot's
``lidar`` site at ``rate_hz``, optionally applies sensor noise, and exposes the latest
:class:`~.payloads.LaserScan` via a ``scan`` output endpoint for a transport plugin.

The shared machinery -- the rate gate, the detection limits, the noise model, the static mount TF and
the endpoint -- lives in :class:`~.lidar_common.RayCastSensorPlugin`; this file is the fan pattern,
the ``LaserScan`` payload, and the device defaults.

Config (in addition to ``lidar_common``'s ``namespace``/``site``/``frame_id``/
``rate_hz``/``exclude_body``/``dropout_percent``/``emit_static_tf``/``tf_parent``)::

    lidar:
      rays: 360                  # samples; the first at angle_min, the last exactly at angle_max
      angle_min: 0.0
      angle_max: 6.265732015     # default: angle_min + 2*pi*(rays-1)/rays, a full turn
                                 #   with no bearing published twice
      range_min: 0.164           # published as the header's range_min
      max_range: 20.0            # published as the header's range_max
      detection_min: 0.164       # nearest distance measured; a nearer hit is too close
                                 #   (default: range_min)
      detection_max: 20.0        # farthest distance measured; a farther hit is no return
                                 #   (default: max_range)
      too_close: -inf            # published for a too-close hit: -inf, +inf, nan, raw
                                 #   (the true distance) or a number
      no_return: +inf            # published where nothing is measured: +inf, nan or a number
      range_stddev: 0.0          # Gaussian range sigma (m)
      range_stddev_relative: 0.0 # sigma as a fraction of the distance, at and beyond
                                 #   range_stddev_relative_from (0 = constant sigma)
      range_stddev_relative_from: 0.0   # m; nearer than this the sigma is range_stddev
      range_resolution: 0.0      # quantisation step of a published distance (m); 0 = continuous

**What a scan publishes follows ROS REP 117 unless the device says otherwise.** A hit nearer than
``detection_min`` is published as ``too_close`` (``-inf`` by default) and a ray with nothing within
``detection_max`` as ``no_return`` (``+inf``). A too-close hit is never raised to ``range_min`` and
never published as though it were measured. A device model whose real driver publishes something
else declares it, e.g. ``urg_node`` publishes a Hokuyo's too-close error code as ``0.004`` and its
no-return code as ``65.533``. A consumer uses a range only where ``range_min <= r <= range_max`` and
the value is finite, as REP 117 prescribes.

**The header and the physics are separate keys** because they differ on real hardware: a driver
publishes a hard-coded ``range_min``/``range_max`` (``neo_sick_s300`` publishes 0.01/29.0 for a
scanner that measures 0.05 to 30 m), while whether a return exists is the device's physical limit.
``range_min``/``max_range`` are what the header says; ``detection_min``/``detection_max`` decide
which rays are too close, measured, or no return. Each defaults to its header counterpart.

**Layout.** ``rays`` samples run from ``angle_min`` to ``angle_max`` inclusive, ``angle_increment =
(angle_max - angle_min) / (rays - 1)``, which is how every 2D scanner driver lays out a
``LaserScan``. A full turn is written the way its driver writes it: ``angle_max`` one increment short
of ``angle_min + 2*pi`` where no bearing repeats (the LDS-01's ``hls_lfcd_lds_driver``), or
``-pi .. pi`` where the first and last ray share a bearing (the RPLIDAR C1's ``sllidar_ros2``).

**Noise** is drawn on the true distance of every hit and published on measured and ``raw``
too-close returns; a constant ``too_close``/``no_return`` carries none. A noisy distance is floored at
0 and then quantised to ``range_resolution``. It is not re-classified: a measured return that noise
carries past a header limit is published as the device would publish it, outside
``[range_min, range_max]``.

``frame_id`` defaults to ``site``; set it when the robot's real description names the frame
differently (the TurtleBot 4's URDF calls it ``rplidar_link``).

The static mount TF (``emit_static_tf``, on by default) is ``tf_parent -> frame_id``, measured from
that body. ``tf_parent`` defaults to the resolved ``exclude_body``, else ``world``; set it when the
excluded body is the scanner's own housing rather than the link its frame hangs from. A device
model mounted with ``spawn_sensor`` sets ``emit_static_tf: false``: the mount publishes the chain.
"""

from __future__ import annotations

import math
import numbers

import numpy as np

from .lidar_common import RayCastSensorPlugin
from .payloads import LaserScan

#: Spellings ``too_close``/``no_return`` accept besides a number. ``None`` is ``raw``.
_OUTPUT_WORDS = {"-inf": -math.inf, "+inf": math.inf, "inf": math.inf, "nan": math.nan, "raw": None}


def _output_value(
    key: str, value, *, allow_raw: bool, allow_neg_inf: bool
) -> tuple[float | None, str]:
    """``(published value or None for raw, error)`` for a ``too_close``/``no_return`` setting."""
    if isinstance(value, str):
        word = value.strip().lower()
        if word not in _OUTPUT_WORDS:
            return None, f"'{key}' must be a number or one of {_allowed(allow_raw, allow_neg_inf)}"
        out = _OUTPUT_WORDS[word]
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        out = float(value)
    else:
        return None, f"'{key}' must be a number or one of {_allowed(allow_raw, allow_neg_inf)}"
    if out is None and not allow_raw:
        return None, f"'{key}' cannot be 'raw': there is no distance to publish"
    if out is not None and out == -math.inf and not allow_neg_inf:
        return (
            None,
            f"'{key}' cannot be -inf: REP 117 reserves -inf for a return too close to measure",
        )
    if out is not None and math.isfinite(out) and out < 0:
        return None, f"'{key}' must be >= 0 when it is a number"
    return out, ""


def _allowed(allow_raw: bool, allow_neg_inf: bool) -> str:
    words = [w for w in ("-inf", "+inf", "nan", "raw") if (w != "raw" or allow_raw)]
    return ", ".join(w for w in words if w != "-inf" or allow_neg_inf)


class LidarPlugin(RayCastSensorPlugin):
    ENDPOINT_NAME = "scan"
    ROS_TYPE = "sensor_msgs.msg.LaserScan"
    DEFAULT_TOPIC = "scan"
    PLUGIN_LABEL = "lidar"

    DEFAULT_SITE = "lidar"
    DEFAULT_RANGE_MIN = 0.164
    DEFAULT_MAX_RANGE = 20.0
    DEFAULT_RAYS = 360
    DEFAULT_TOO_CLOSE = "-inf"
    DEFAULT_NO_RETURN = "+inf"

    #: The detection limits are read per frame. Each stores ``None`` while it follows its header
    #: counterpart, so a fault writing ``max_range`` still moves the far limit of a device that does
    #: not declare its own.
    LIVE_WRITABLE = {
        "detection_min": "_detection_min",
        "detection_max": "_detection_max",
    }

    #: The 2D fan's geometry. A ``LaserScan`` is a FIXED-LENGTH array whose ``angle_increment`` a
    #: consumer reads once, so writing any of these mid-run changes the array's length or the bearing
    #: its indices mean -- and a costmap would rebuild against a scan it cannot compare with the
    #: previous one. Set them in the world; sweep them as a campaign factor.
    REFUSED_WRITES = {
        "rays": "it changes the LaserScan's length mid-run, and a fixed length is the one thing "
        "every consumer of it relies on.",
        "angle_min": "it changes angle_increment, so the same index means a different bearing "
        "before and after the write.",
        "angle_max": "see 'angle_min'.",
    }

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        cfg = config or {}
        # Before the base: its fault bookkeeping reads every LIVE_WRITABLE attribute.
        self._detection_min = _optional_float(cfg.get("detection_min"))
        self._detection_max = _optional_float(cfg.get("detection_max"))
        super().__init__(config, name=name, entity=entity, label=label)
        self._num_rays = int(self.config.get("rays", self.DEFAULT_RAYS))
        self.angle_min = float(self.config.get("angle_min", 0.0))
        self.angle_max = float(
            self.config.get("angle_max", self._full_turn_max(self.angle_min, self._num_rays))
        )
        self._angle_increment = (
            (self.angle_max - self.angle_min) / (self._num_rays - 1) if self._num_rays > 1 else 0.0
        )
        self.too_close, _ = _output_value(
            "too_close",
            self.config.get("too_close", self.DEFAULT_TOO_CLOSE),
            allow_raw=True,
            allow_neg_inf=True,
        )
        no_return, _ = _output_value(
            "no_return",
            self.config.get("no_return", self.DEFAULT_NO_RETURN),
            allow_raw=False,
            allow_neg_inf=False,
        )
        self.no_return = math.inf if no_return is None else no_return

    @staticmethod
    def _full_turn_max(angle_min: float, rays: int) -> float:
        """``angle_max`` of a full turn sampled ``rays`` times with no bearing published twice."""
        rays = max(rays, 1)
        return angle_min + 2.0 * math.pi * (rays - 1) / rays

    @property
    def num_rays(self) -> int:
        return self._num_rays

    @property
    def angle_increment(self) -> float:
        return self._angle_increment

    @property
    def detection_min(self) -> float:
        return self.range_min if self._detection_min is None else self._detection_min

    @property
    def detection_max(self) -> float:
        return self.range_max if self._detection_max is None else self._detection_max

    def _validate_extra(self, config: dict) -> list[str]:
        errors = []
        rays = int(config.get("rays", self.DEFAULT_RAYS))
        if rays <= 0:
            errors.append("'rays' must be > 0")
        angle_min = float(config.get("angle_min", 0.0))
        angle_max = float(config.get("angle_max", self._full_turn_max(angle_min, rays)))
        if rays == 1 and angle_max != angle_min:
            errors.append("a single ray has no increment: 'angle_max' must equal 'angle_min'")
        if rays > 1 and angle_max <= angle_min:
            errors.append("'angle_max' must be > 'angle_min'")
        for key, allow_raw, allow_neg_inf, default in (
            ("too_close", True, True, self.DEFAULT_TOO_CLOSE),
            ("no_return", False, False, self.DEFAULT_NO_RETURN),
        ):
            _, error = _output_value(
                key, config.get(key, default), allow_raw=allow_raw, allow_neg_inf=allow_neg_inf
            )
            if error:
                errors.append(error)
        near = float(config.get("detection_min", config.get("range_min", self.DEFAULT_RANGE_MIN)))
        far = float(config.get("detection_max", config.get("max_range", self.DEFAULT_MAX_RANGE)))
        # A limit that follows its header key is already checked there; only its own key is here.
        if "detection_min" in config and near < 0:
            errors.append("'detection_min' must be >= 0")
        if "detection_max" in config and far <= 0:
            errors.append("'detection_max' must be > 0")
        if near >= 0 and far > 0 and near >= far:
            errors.append(
                f"the detection limits are empty: 'detection_min' ({near}) must be < "
                f"'detection_max' ({far}); each defaults to 'range_min' / 'max_range'"
            )
        return errors

    def _build_directions(self) -> np.ndarray:
        """A horizontal fan from ``angle_min`` to ``angle_max``, both endpoints sampled."""
        angles = np.linspace(self.angle_min, self.angle_max, self._num_rays)
        return np.stack([np.cos(angles), np.sin(angles), np.zeros(self._num_rays)], axis=1)

    def _payload(self, dist: np.ndarray, valid: np.ndarray, near: np.ndarray) -> LaserScan:
        ranges = np.full(self._num_rays, self.no_return, dtype=np.float64)
        ranges[valid] = dist[valid]
        ranges[near] = dist[near] if self.too_close is None else self.too_close
        return LaserScan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            angle_increment=self._angle_increment,
            range_min=self.range_min,
            range_max=self.range_max,
        )


def _optional_float(value) -> float | None:
    return None if value is None else float(value)
