"""Local avoidance: turning what each agent *wants* to do into what it can do around the others.

The local half of the global-plan / local-control split. The planner routes around walls; this
layer handles everything that moves, which a grid rasterized once cannot represent.

ORCA is *an* answer, not the answer -- social-force models, velocity obstacles, a sampled local
planner and a learned policy are all plausible successors. So models are resolved by name from the
``roqsim_nav.avoidance`` entry-point group (or ``module:Class`` / ``file.py:Class``), and nothing in
this package branches on the name of one. The interface below is deliberately shaped to survive that
replacement rather than to fit the implementation that happens to ship.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .._resolve import RegistryError, registered, resolve

ENTRY_POINT_GROUP = "roqsim_nav.avoidance"

#: The id returned for an agent when no model is loaded. Every call is then a no-op and ``result``
#: is the identity, so a world without an avoidance model behaves as one where nobody yields.
NO_AGENT = -1


class AvoidanceModel(ABC):
    """One local-avoidance policy for a world.

    **The contract, and the whole of it: preferred velocity in, achievable velocity out**, world
    frame, m/s. Forces, accelerations, sampled rollouts, a network's logits -- all of that is the
    implementation's business and none of it appears here. A model that thinks in accelerations
    integrates them itself. That is what keeps this from becoming ORCA's API under another name.

    Six things about the shape below are load-bearing, and each is something a narrower interface
    would have got wrong:

    * :meth:`submit`, :meth:`solve` and :meth:`result` are **three phases, not one call**. ORCA's
      native API is incremental and per-agent; baking that in would force every future model to be
      sequential. Split, a batched or vectorised model computes all agents at once -- and because
      ``solve`` runs before any ``result`` in a step, the order plugins appear in a world does not
      matter.
    * ``yields`` is **semantic**, not "is it the robot": it says whether this model may move the
      agent at all. It is derived, never configured -- see the ``avoidance`` plugin.
    * Static geometry arrives as the **planner's own polygons**, so the global and local layers
      cannot disagree about where a wall is. A model wanting a distance field derives one in
      :meth:`configure`.
    * ``params`` is opaque, but :attr:`params_schema` is not, so a key left behind after switching
      models is refused at load rather than ignored at runtime.
    * ``present`` honours :mod:`roqsim.presence`, so an entity made absent stops deflecting others,
      exactly as it stops being seen by every raycaster.
    * :meth:`reset` exists because episodes do: agents and walls survive one, velocities must not.

    **The one known limit, stated rather than smuggled:** an agent is a **disc**. A long cart is its
    circumscribed circle. Widening to footprints is a real interface change -- a new ``add_agent``
    argument and a migration for every implementation -- and should be made deliberately when
    something needs it, not slipped through ``params`` where only one model would honour it.
    """

    #: Keys this model accepts, world-level and per-agent, so ``validate_config`` can refuse a typo
    #: at load time. Empty means "anything", which a model should only choose deliberately.
    params_schema: tuple[str, ...] = ()

    def configure(self, ctx, params: dict) -> None:
        """Set up for this world. ``ctx`` is available for ``rng_for`` and the compiled model."""

    @abstractmethod
    def add_agent(self, key: str, *, radius: float, max_speed: float, yields: bool, params: dict):
        """Register a participant and return its id.

        ``yields=False`` means this model may never move it: its state is overwritten from ground
        truth every step and the others must go round it.
        """

    @abstractmethod
    def add_static(self, polygons) -> None:
        """The world's static geometry, as CCW footprints from
        :func:`~roqsim_nav.obstacles.wall_polygons`."""

    @abstractmethod
    def submit(self, aid, pos, vel, pref_vel, *, present: bool = True) -> None:
        """This agent's ground-truth state and what it wants, for the next :meth:`solve`."""

    @abstractmethod
    def solve(self, dt: float) -> None:
        """Compute every agent's outcome from the submissions."""

    @abstractmethod
    def result(self, aid) -> np.ndarray:
        """The velocity ``aid`` should execute.

        Must equal the submitted ``pref_vel`` for a non-yielding agent, and for any agent the model
        cannot improve on.
        """

    def reset(self) -> None:
        """Drop per-episode state. Agents and static geometry survive; velocities do not."""


class NullAvoidance(AvoidanceModel):
    """What a world with no ``avoidance:`` entry gets: everyone executes what they wanted.

    A real object rather than a ``None`` the navigator has to test for, so there is one code path
    through the tick whether or not a world declares avoidance.
    """

    def add_agent(self, key, *, radius, max_speed, yields, params):
        return NO_AGENT

    def add_static(self, polygons) -> None: ...

    def submit(self, aid, pos, vel, pref_vel, *, present: bool = True) -> None:
        self._last = np.asarray(pref_vel, dtype=float)

    def solve(self, dt: float) -> None: ...

    def result(self, aid) -> np.ndarray:
        return getattr(self, "_last", np.zeros(2))


class AvoidanceService:
    """The world's model, plus the once-per-step solve.

    The solve is stamped with the step it ran for and triggered by whichever party reaches it first
    -- the ``avoidance`` plugin's own ``pre_step``, or a navigator about to read a result. Without
    that, the model would be solved wherever its entry happened to sit in the world file, and a
    navigator declared *before* it would read a result one step staler than one declared after. A
    world author would get a different trajectory for moving a line in a file, which is the kind of
    difference nobody thinks to look for.
    """

    def __init__(self, model: AvoidanceModel):
        self.model = model
        self._solved_step = -1

    def ensure_solved(self, ctx) -> None:
        step = round(ctx.sim_time / ctx.dt) if ctx.dt else 0
        if step == self._solved_step:
            return
        self._solved_step = step
        self.model.solve(float(ctx.dt))

    def reset(self) -> None:
        self._solved_step = -1
        self.model.reset()

    # -- the model's own surface, forwarded so a caller holds one object ------------------------
    def add_agent(self, key, **kw):
        return self.model.add_agent(key, **kw)

    def add_static(self, polygons) -> None:
        self.model.add_static(polygons)

    def submit(self, aid, pos, vel, pref_vel, *, present: bool = True) -> None:
        self.model.submit(aid, pos, vel, pref_vel, present=present)

    def result(self, aid) -> np.ndarray:
        return self.model.result(aid)


def resolve_model(ref: str, base_dir: Path | None = None) -> type[AvoidanceModel]:
    """Resolve ``ref`` to an :class:`AvoidanceModel` subclass."""
    return resolve(
        ref, group=ENTRY_POINT_GROUP, base=AvoidanceModel, kind="avoidance model", base_dir=base_dir
    )


def available() -> list[str]:
    """Registered model names, for error messages."""
    return registered(ENTRY_POINT_GROUP)


__all__ = [
    "NO_AGENT",
    "AvoidanceModel",
    "AvoidanceService",
    "NullAvoidance",
    "RegistryError",
    "available",
    "resolve_model",
]
