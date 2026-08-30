"""World plugin: time-varying wind -- steady flow, a discrete gust, and Dryden turbulence.

``sim.wind`` sets a *constant* wind vector before compile, which is the right shape for stating the
conditions a world is in. It is the wrong shape for testing disturbance rejection, because a constant
is the one wind a controller never has to reject: the drone trims against it once and the error goes
to zero. What defeats an attitude loop is wind that *changes* -- a gust front, and the continuous
turbulence underneath it.

This plugin writes ``model.opt.wind`` each tick, so MuJoCo's existing drag terms do the work; nothing
here applies a force of its own.

Config::

    wind_field:
      steady: [2.0, 0.0, 0.0]     # m/s, world frame -- the mean flow
      gust:                       # optional 1-cosine discrete gust (MIL-F-8785C shape)
        magnitude: 4.0            # m/s at the peak
        axis: [1.0, 0.0, 0.0]     # direction (normalised; defaults to `steady`, else +x)
        onset: 5.0                # s, when it starts
        duration: 1.0             # s, rise-and-fall length
      turbulence:                 # optional continuous Dryden turbulence
        intensity: 0.6            # sigma, m/s (per horizontal axis)
        length_scale: 5.0         # L, m -- larger = slower, more correlated gusting
        vertical: 0.5             # sigma scale on the w axis (Dryden's low-altitude form)

**Why a plugin and not a runtime override.** ``model_override`` refuses every ``opt.*`` global on
purpose (see its module docstring, and ``docs/architecture.rst``): the ``sim:`` block owns them, it
owns them *before compile*, and a runtime write would make the value a run recorded differ from the
value that ran. That argument is about a fault-injection hook poking a global whose recorded value is
a single number. It does not extend to wind, because wind here is not a value -- it is a *signal*,
fully determined by the parameters above plus the run's seed, all of which are recorded with the
world. Re-run the same world with the same seed and the same wind happens, tick for tick.

**One owner per knob**, which is why declaring both ``sim.wind`` and this plugin is refused rather
than merged. Two owners of one global is precisely the situation that rule exists to prevent: the
compiled model would say one thing, the first tick would say another, and the run's provenance would
record the value that was immediately overwritten. State the mean flow in ``steady:`` instead --
that is the same wind, owned once.

**Randomness comes from ``ctx.rng_for``, like every other stochastic plugin here.** Turbulence is
therefore a pure function of ``(sim.seed, episode, sim_time)``, randomly accessible rather than
replayed, and the run's own seed governs it -- so this plugin has no ``seed`` key of its own. Note
the consequence, which is deliberate substrate behaviour rather than an accident: because ``episode``
is part of the key, repetitions of one configuration see *different* turbulence. Repetitions are
samples of the weather, not copies of it, which is what makes averaging over them mean anything.

**Wind is inert in a vacuum.** MuJoCo applies wind as a relative velocity into the density and
viscosity drag terms, so with both at 0 this plugin has no effect whatever. That is a silent,
plausible-looking failure -- the drone still flies, it just ignores the weather -- so it warns, the
same way ``quadrotor_controller`` warns about the same missing medium.
"""

from __future__ import annotations

import logging

import numpy as np

from roqsim.context import SimContext
from roqsim.plugin import Plugin

logger = logging.getLogger(__name__)

#: Dryden's low-altitude form makes the vertical component less energetic than the horizontal ones.
_DEFAULT_VERTICAL_SCALE = 0.5

#: The convection speed the Dryden filter is discretised against, when the mean flow is too slow to
#: supply one. Turbulence is carried past the vehicle at the airspeed; a hovering drone in still air
#: has none, and the filter would freeze. This is the "there is still weather when it is calm" floor.
_MIN_CONVECTION_MPS = 1.0


class WindFieldPlugin(Plugin):
    #: Writes a world global and holds only its own filter state; it owns no entity.
    parallel_safe = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self._steady = self._vec3(self.config.get("steady"), (0.0, 0.0, 0.0))
        self._gust = self.config.get("gust") or None
        self._turbulence = self.config.get("turbulence") or None
        self._turb = np.zeros(3)

    # -- config ----------------------------------------------------------------------------------
    @staticmethod
    def _vec3(value, default):
        if value is None:
            return np.array(default, dtype=float)
        return np.array([float(v) for v in value], dtype=float)

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if "steady" in config and len(config["steady"]) != 3:
            errors.append("'steady' must be 3 numbers (m/s, world frame)")
        if "seed" in config:
            errors.append(
                "'seed' is not a key here: turbulence draws from the run's seed (sim.seed / "
                "roqsim sim --seed) through ctx.rng_for, so the whole world reproduces together"
            )
        gust = config.get("gust")
        if gust is not None:
            if not isinstance(gust, dict):
                errors.append("'gust' must be a mapping")
            else:
                if "magnitude" not in gust:
                    errors.append("'gust.magnitude' is required (m/s)")
                if float(gust.get("duration", 1.0)) <= 0:
                    errors.append("'gust.duration' must be > 0 s")
                if "axis" in gust and len(gust["axis"]) != 3:
                    errors.append("'gust.axis' must be 3 numbers")
        turb = config.get("turbulence")
        if turb is not None:
            if not isinstance(turb, dict):
                errors.append("'turbulence' must be a mapping")
            else:
                if float(turb.get("intensity", 0.0)) < 0:
                    errors.append("'turbulence.intensity' must be >= 0 m/s")
                if float(turb.get("length_scale", 5.0)) <= 0:
                    errors.append("'turbulence.length_scale' must be > 0 m")
        return errors

    # -- lifecycle -------------------------------------------------------------------------------
    def configure(self, ctx: SimContext) -> None:
        model = ctx.model
        if float(model.opt.density) == 0.0 and float(model.opt.viscosity) == 0.0:
            logger.warning(
                "wind_field (%s): the world has no medium (density and viscosity are 0), so wind "
                "has no effect at all -- MuJoCo applies it through the drag terms. Set "
                "sim: {density: 1.225, viscosity: 1.8e-5} for air.",
                self.name,
            )
        # `sim.wind` has already been applied to the compiled model, so a non-zero value here is a
        # world declaring the same knob twice. Refuse rather than pick a winner.
        if float(np.linalg.norm(model.opt.wind)) > 0.0:
            raise RuntimeError(
                f"wind_field ({self.name}): the world also sets 'sim.wind' "
                f"({list(model.opt.wind)}), and this plugin overwrites it on the first tick. "
                f"One owner per knob: move that vector into this plugin's 'steady:' and remove "
                f"'sim.wind'."
            )
        self._apply(ctx, self._steady)

    def on_reset(self, ctx: SimContext) -> None:
        self._turb = np.zeros(3)
        self._apply(ctx, self._steady)

    def pre_step(self, ctx: SimContext) -> None:
        wind = np.array(self._steady, dtype=float)
        if self._gust:
            wind = wind + self._gust_at(ctx.sim_time)
        if self._turbulence:
            self._advance_turbulence(ctx, wind)
            wind = wind + self._turb
        self._apply(ctx, wind)

    @staticmethod
    def _apply(ctx: SimContext, wind) -> None:
        ctx.model.opt.wind = [float(v) for v in wind]

    # -- the two signals -------------------------------------------------------------------------
    def _gust_at(self, t: float) -> np.ndarray:
        """The MIL-F-8785C 1-cosine discrete gust: a smooth rise to a peak and back to zero."""
        onset = float(self._gust.get("onset", 0.0))
        duration = float(self._gust.get("duration", 1.0))
        if not onset <= t < onset + duration:
            return np.zeros(3)
        axis = self._gust.get("axis")
        if axis:
            direction = self._vec3(axis, (0.0, 0.0, 0.0))
        else:
            direction = np.array(self._steady, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            direction, norm = np.array([1.0, 0.0, 0.0]), 1.0
        magnitude = float(self._gust["magnitude"])
        # 0 -> 1 -> 0 over the gust length, with zero slope at both ends: the point of the 1-cosine
        # shape is that it does not hand the controller a step discontinuity to ring on.
        shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t - onset) / duration))
        return direction / norm * magnitude * shape

    def _advance_turbulence(self, ctx: SimContext, mean_wind: np.ndarray) -> None:
        """One step of the Dryden shaping filter, per axis.

        The Dryden spectrum's first-order form is an exponentially-correlated (Ornstein-Uhlenbeck)
        process whose correlation time is ``L / V`` -- the time the vehicle takes to traverse one
        turbulence length scale. Discretised at ``dt``::

            a      = V * dt / L                     (how much of the correlation decays per tick)
            u[k+1] = (1 - a) * u[k] + sigma * sqrt(2a) * N(0, 1)

        The ``sqrt(2a)`` is what keeps the stationary variance at ``sigma^2`` independent of ``dt``,
        so halving the timestep refines the signal instead of changing how windy it is.
        """
        sigma = float(self._turbulence.get("intensity", 0.0))
        if sigma <= 0.0:
            self._turb = np.zeros(3)
            return
        length = float(self._turbulence.get("length_scale", 5.0))
        vertical = float(self._turbulence.get("vertical", _DEFAULT_VERTICAL_SCALE))

        convection = max(float(np.linalg.norm(mean_wind)), _MIN_CONVECTION_MPS)
        # Clamped: a timestep past the correlation time is memoryless, not unstable.
        a = min(convection * ctx.dt / length, 1.0)
        scale = np.array([sigma, sigma, sigma * vertical])
        draw = ctx.rng_for(f"wind_field:{self.name}").standard_normal(3)
        self._turb = (1.0 - a) * self._turb + scale * np.sqrt(2.0 * a) * draw
