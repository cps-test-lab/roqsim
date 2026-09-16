"""Sensor plugin: a GNSS receiver -- local ENU position converted to WGS84, with drift.

**Why this is not part of the flight-stack bridge.** Two reasons, and both are about ownership.
The *datum* -- where on Earth the world's origin is -- is a property of the world, not of the
autopilot attached to one vehicle in it: two drones in the same world must agree on it, and a
recorded run must carry it whether or not a flight stack was ever loaded. And **GNSS denial is a
first-class experiment factor**: an urban-canyon or jamming study varies ``denied`` (and the noise
terms) across a campaign, which requires the receiver to be a plugin with its own config rather
than a config block nested inside a bridge that a GNSS-free experiment would not load at all.

So the receiver stands alone, publishes ``NavSatFix``, and puts a handle on the blackboard at
``gnss:<robot>`` that :mod:`roqsim_aerial.plugins.px4_sitl` reads for ``HIL_GPS``.

Config::

    gnss:
      datum: {lat: 47.397742, lon: 8.545594, alt: 488.0}   # REQUIRED -- world origin on Earth
      body: x500                 # the body whose position is measured (default: the entity root)
      rate: 10.0                 # Hz -- real GNSS is slow, and EKF2's behaviour depends on that
      horizontal_noise: 0.5      # m, 1-sigma
      vertical_noise: 1.0        # m, 1-sigma
      velocity_noise: 0.1        # m/s, 1-sigma per axis
      bias_time: 60.0            # s, correlation time of the slowly-drifting position bias
      satellites: 12
      eph: 0.5                   # m, reported horizontal accuracy (default: horizontal_noise)
      epv: 1.0                   # m, reported vertical accuracy (default: vertical_noise)
      fix_type: 3                # 3 = 3D fix
      denied: false              # true -> no fix at all, without removing the plugin

**The projection is deliberately the cheap one.** ``dlat = north / R``, ``dlon = east / (R cos
lat)`` with ``R = 6378137.0`` m (the WGS84 semi-major axis) -- an equirectangular tangent-plane
approximation. Its error grows with the square of the offset from the datum, and over the
kilometre-scale extents a MuJoCo world can actually hold it stays well under a metre, i.e. below
this receiver's own noise floor. **It is not a general geodesy routine**: it ignores the ellipsoid's
flattening, uses geodetic altitude as if it were MSL, and would be wrong at continental range. Any
experiment that needs true geodesy needs pyproj and a real projection, not a bigger constant here.

**Noise is a drifting bias plus white noise, not white noise alone.** A pure-white GNSS is the
input that makes an EKF look best: averaging kills it, so position error shrinks with the filter's
window and the estimator appears to have solved a problem it has not. Real GNSS error is dominated
by slowly-correlated terms (ionosphere, ephemeris, multipath geometry), and *that* is what a
navigation experiment is about -- how the estimator behaves when its position reference wanders
metres over a minute. The bias here is an Ornstein-Uhlenbeck process with correlation time
``bias_time``; the white term rides on top.

**Randomness comes from ``ctx.rng_for``**, like every other stochastic plugin, so the fix is a pure
function of ``(sim.seed, episode, sim_time)`` and this plugin has no ``seed`` key of its own.

**The fix is held between updates.** At ``rate`` Hz the value changes 10 times a second and is
constant in between, because that is what the sensor does -- a consumer that interpolated it would
be inventing measurements the receiver never made, and would hide exactly the latency an estimator
has to cope with.

**What this does not model**: multipath and its geometry-dependent structure, ionospheric and
tropospheric delay, constellation geometry (so ``satellites`` and the reported ``eph``/``epv`` are
declared constants, not computed DOP), RTK/carrier-phase modes, receiver clock error, and
acquisition time -- the fix is available on the first tick.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin

#: WGS84 semi-major axis, m. The single constant the tangent-plane projection uses.
R_EARTH = 6378137.0

_DEFAULTS = {
    "rate": 10.0,
    "horizontal_noise": 0.5,
    "vertical_noise": 1.0,
    "velocity_noise": 0.1,
    "bias_time": 60.0,
    "satellites": 12,
    "fix_type": 3,
    "denied": False,
}


@dataclass
class GnssHandle:
    """In-process handle a bridge reads instead of importing this plugin."""

    name: str
    read_fix: Callable[[], dict]
    #: The receiver's own update rate, so a consumer can send at the sensor's cadence rather than
    #: at the physics rate -- resending a held fix at 250 Hz tells an estimator it has 250 Hz of
    #: independent GNSS, which is a lie it will happily believe.
    rate: float
    datum: dict


class GnssPlugin(Plugin):
    #: Measures one entity's body, so it belongs inside that entity's ``components:`` block.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self._bid = -1
        self._bias = np.zeros(3)
        self._next_update = 0.0
        self._fix: dict | None = None

    def cfg(self, key):
        return self.config.get(key, _DEFAULTS[key])

    # -- config ----------------------------------------------------------------------------------
    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        datum = config.get("datum")
        if datum is None:
            errors.append(
                "'datum' is required: {lat, lon, alt}. There is no defensible default -- a fix "
                "computed against an invented origin is not a missing measurement, it is a "
                "plausible-looking wrong one, and every position it reports would be confidently "
                "somewhere else on Earth."
            )
        elif not isinstance(datum, dict):
            errors.append("'datum' must be a mapping {lat, lon, alt}")
        else:
            for key in ("lat", "lon", "alt"):
                if key not in datum:
                    errors.append(f"'datum.{key}' is required")
            lat = datum.get("lat")
            if lat is not None and not -90.0 <= float(lat) <= 90.0:
                errors.append("'datum.lat' must be within [-90, 90] degrees")
            lon = datum.get("lon")
            if lon is not None and not -180.0 <= float(lon) <= 180.0:
                errors.append("'datum.lon' must be within [-180, 180] degrees")
        if "seed" in config:
            errors.append(
                "'seed' is not a key here: the noise draws from the run's seed (sim.seed / "
                "roqsim sim --seed) through ctx.rng_for, so the whole world reproduces together"
            )
        if "rate" in config and float(config["rate"]) <= 0:
            errors.append("'rate' must be > 0 Hz")
        if "bias_time" in config and float(config["bias_time"]) <= 0:
            errors.append("'bias_time' must be > 0 s (it is a correlation time)")
        for key in ("horizontal_noise", "vertical_noise", "velocity_noise"):
            if key in config and float(config[key]) < 0:
                errors.append(f"'{key}' must be >= 0")
        if "satellites" in config and int(config["satellites"]) < 0:
            errors.append("'satellites' must be >= 0")
        return errors

    # -- lifecycle -------------------------------------------------------------------------------
    def _resolve_body(self, entity, prefix: str) -> str:
        """The body to measure, as a name in the COMPILED model.

        The entity's root body is ``Entity.body`` -- the first-class attribute ``spawn_robot`` fills
        in, and already prefixed, because it is resolved against the compiled model rather than
        written by hand. A config ``body:`` names a body in the MODEL's own namespace, so it takes
        the prefix here; ``entity.body`` must not, or the prefix would be applied twice.
        """
        configured = self.config.get("body")
        if configured:
            return prefix + str(configured)
        if entity is not None and entity.body:
            return entity.body
        raise RuntimeError(
            f"gnss ({self.name}): no body to measure -- entity {self.robot!r} has no root body, "
            f"so name one in this plugin's 'body:'"
        )

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")

        body = self._resolve_body(entity, prefix)
        self._bid = mujoco.mj_name2id(ctx.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if self._bid < 0:
            raise RuntimeError(
                f"gnss ({self.name}): body {body!r} is not in the compiled model"
            )

        datum = self.config["datum"]
        self._datum = {
            "lat": float(datum["lat"]),
            "lon": float(datum["lon"]),
            "alt": float(datum["alt"]),
        }
        # cos(lat) of the datum, not of the current fix: the projection is a tangent plane pinned at
        # the origin, and recomputing it per-fix would make the scale factor depend on the vehicle's
        # own noise.
        self._cos_lat = math.cos(math.radians(self._datum["lat"]))

        ctx.blackboard.set(
            f"gnss:{self.robot}",
            GnssHandle(
                name=self.robot,
                read_fix=self.read_fix,
                rate=float(self.cfg("rate")),
                datum=dict(self._datum),
            ),
        )
        ctx.interface.add(
            Endpoint(
                name="fix",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self.read_fix,
                rate_hz=float(self.cfg("rate")),
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.NavSatFix",
                        "topic": self.topic_override("fix") or "fix",
                    }
                },
            )
        )

    def on_reset(self, ctx: SimContext) -> None:
        self._bias = np.zeros(3)
        self._next_update = 0.0
        self._fix = None

    def post_step(self, ctx: SimContext) -> None:
        """Sample after the step, so the fix describes the state the tick produced."""
        if ctx.sim_time + 1e-12 < self._next_update:
            return
        self._next_update = ctx.sim_time + 1.0 / float(self.cfg("rate"))
        self._fix = self._sample(ctx)

    # -- the measurement -------------------------------------------------------------------------
    def read_fix(self) -> dict:
        """The last fix the receiver produced; held between updates.

        Returns a no-fix reading before the first update (and always, when ``denied``), because a
        receiver that has not measured yet has nothing to report and a consumer must see that
        rather than a zero that looks like the datum.
        """
        if self._fix is None:
            return self._no_fix()
        return dict(self._fix)

    def _no_fix(self) -> dict:
        """A receiver reporting nothing. Position fields are zeroed rather than last-known: a stale
        position carried under ``valid: False`` is the one a careless consumer publishes anyway."""
        return {
            "lat": 0.0,
            "lon": 0.0,
            "alt": 0.0,
            "eph": 0.0,
            "epv": 0.0,
            "vel_n": 0.0,
            "vel_e": 0.0,
            "vel_d": 0.0,
            "vel": 0.0,
            "cog": 0.0,
            "fix_type": 0,
            "satellites": 0,
            "valid": False,
        }

    def _sample(self, ctx: SimContext) -> dict:
        if bool(self.cfg("denied")):
            return self._no_fix()

        pos = np.array(ctx.data.xpos[self._bid], dtype=float)  # world ENU, m
        vel = np.array(ctx.data.cvel[self._bid][3:6], dtype=float)  # world ENU, m/s

        sigma_h = float(self.cfg("horizontal_noise"))
        sigma_v = float(self.cfg("vertical_noise"))
        sigma_vel = float(self.cfg("velocity_noise"))
        rng = ctx.rng_for(f"gnss:{self.name}")

        # Ornstein-Uhlenbeck bias. `a` is the fraction of the correlation that decays per UPDATE
        # (not per physics tick -- the bias only advances when the receiver reports), and the
        # sqrt(2a) keeps the stationary spread at sigma whatever the rate, so lowering the receiver
        # rate does not quietly make the world's GNSS better.
        scale = np.array([sigma_h, sigma_h, sigma_v])
        a = min((1.0 / float(self.cfg("rate"))) / float(self.cfg("bias_time")), 1.0)
        self._bias = (1.0 - a) * self._bias + scale * math.sqrt(2.0 * a) * rng.standard_normal(3)
        white = scale * rng.standard_normal(3)
        east, north, up = pos + self._bias + white

        lat = self._datum["lat"] + math.degrees(north / R_EARTH)
        lon = self._datum["lon"] + math.degrees(east / (R_EARTH * max(self._cos_lat, 1e-9)))
        alt = self._datum["alt"] + up

        v_noise = sigma_vel * rng.standard_normal(3)
        vel_e, vel_n, vel_u = vel + v_noise
        vel_d = -vel_u
        ground = math.hypot(vel_n, vel_e)
        cog = math.degrees(math.atan2(vel_e, vel_n)) % 360.0

        return {
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "eph": float(self.config.get("eph", sigma_h)),
            "epv": float(self.config.get("epv", sigma_v)),
            "vel_n": float(vel_n),
            "vel_e": float(vel_e),
            "vel_d": float(vel_d),
            "vel": float(ground),
            "cog": float(cog),
            "fix_type": int(self.cfg("fix_type")),
            "satellites": int(self.cfg("satellites")),
            "valid": True,
        }
