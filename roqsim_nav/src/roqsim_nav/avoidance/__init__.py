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

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .._resolve import RegistryError, registered, resolve

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "roqsim_nav.avoidance"

#: The model a mover gets when it asks to be steered without naming one -- a walker's
#: `avoidance: true`. A navigator names its model with `avoidance: {steer: ...}`; without `steer`
#: it is steered by none.
DEFAULT_MODEL = "give_way"

#: Where the world's shared model lives. Shared by necessity rather than by preference: agents can
#: only avoid each other inside one solver, so a world has exactly one.
SERVICE_KEY = "nav:avoidance"

#: The id of a mover that has joined no model -- in a world where no mover asked to be steered.
NO_AGENT = -1

#: How low a thing that moves must reach to stand on the floor and join the model as an agent the
#: others go round: the default floor of a mover's obstacle band. A box on a shelf or a tote on a
#: table does not; a robot, a cart, a person does.
FLOOR_BAND = 0.1


def _free_root(model, root: int) -> bool:
    """Whether the root body ``root`` moves freely: a mocap body, or one whose joint is free."""
    import mujoco  # noqa: PLC0415

    if root <= 0:
        return False
    if int(model.body_mocapid[root]) >= 0:
        return True
    first, count = int(model.body_jntadr[root]), int(model.body_jntnum[root])
    return count > 0 and int(model.jnt_type[first]) == mujoco.mjtJoint.mjJNT_FREE


def _planar_radius(model, data, geoms, centre) -> float:
    """The radius around ``centre`` (xy) the plan view of ``geoms`` reaches: for each, its centre's
    distance and its own reach in the plane -- a sphere's radius, a capsule's or a cylinder's radius
    and the part of its half length that lies in the plane, a box's and any other shape's farthest
    corner of its bounding box (``geom_aabb``) seen from above."""
    import mujoco  # noqa: PLC0415

    centre = np.asarray(centre, dtype=float)[:2]
    best = 0.0
    for g in geoms:
        rot = data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        kind = int(model.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            at, reach = data.geom_xpos[g][:2], float(size[0])
        elif kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            axis = rot[:2, 2]
            at, reach = data.geom_xpos[g][:2], float(size[0] + size[1] * np.linalg.norm(axis))
        else:
            box = model.geom_aabb[g]
            mid = data.geom_xpos[g] + rot @ box[:3]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            pts = mid[:2] + (corners * box[3:]) @ rot[:2, :].T
            best = max(best, float(np.linalg.norm(pts - centre, axis=1).max()))
            continue
        best = max(best, float(np.linalg.norm(np.asarray(at) - centre)) + reach)
    return best


def _lowest(model, data, geoms) -> float:
    """How low the lowest of ``geoms`` reaches (m), each taken as its bounding sphere."""
    return min(float(data.geom_xpos[g][2]) - float(model.geom_rbound[g]) for g in geoms)


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
      agent at all. It is derived, never configured: a navigator with ``steer`` set yields, and
      every other mover -- a navigator that steers by none, and anything that moves and stands on
      the floor with no navigator at all (:meth:`AvoidanceService._join_passive`) -- joins as an
      agent the others go round.
    * Static geometry arrives as the **planner's own polygons**, so the global and local layers
      cannot disagree about where a wall is. A model wanting a distance field derives one in
      :meth:`configure`.
    * ``params`` is opaque, but :attr:`params_schema` is not, so a key left behind after switching
      models is refused at load rather than ignored at runtime.
    * ``present`` honours :mod:`roqsim.presence`, so an entity made absent stops deflecting others,
      exactly as it stops being seen by every raycaster.
    * :meth:`reset` exists because runs do: agents and walls survive one, velocities must not.

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
        """Drop per-run state. Agents and static geometry survive; velocities do not."""

    def exclude(self, aid, other) -> None:
        """Let ``aid`` never give way to ``other``: a mover's ``ignore``.

        A model that cannot leave one agent out of another's solve raises
        :class:`NotImplementedError`, and a world asking it to is refused rather than having the
        ignored agent steered round regardless.
        """
        raise NotImplementedError


class AvoidanceService:
    """The world's model, plus the once-per-step solve.

    The solve is stamped with the step it ran for and triggered by whichever navigator is first to
    read a result in that step. Without that, the model would be solved wherever one mover happened
    to sit in the world file, and a navigator declared *before* it would read a result one step
    staler than one declared after. A world author would get a different trajectory for moving a
    line in a file, which is the kind of difference nobody thinks to look for.
    """

    def __init__(self, model: AvoidanceModel, *, ref: str = "", params: dict | None = None):
        self.model = model
        #: What was asked for, so a second navigator asking for something else can be refused.
        self.ref = ref
        self.params = dict(params or {})
        self._solved_step = -1
        self._reset_generation = -1
        #: Agent id by key, for every agent: a navigator's, and each passive one's.
        self._aids: dict[str, int] = {}
        #: The things that move with no navigator of their own: key -> (aid, root body id, geoms).
        self._passive: dict[str, tuple[int, int, list]] = {}
        self._joined_passive = False
        #: Where each passive agent was on the tick before, for its velocity.
        self._last: dict[str, tuple[float, np.ndarray]] = {}
        #: ``ignore`` of each navigator, applied once every agent is known: (aid, key).
        self._exclusions: list[tuple[int, str]] = []

    def ensure_solved(self, ctx) -> None:
        step = round(ctx.sim_time / ctx.dt) if ctx.dt else 0
        if step == self._solved_step:
            return
        self._solved_step = step
        if not self._joined_passive:
            self._join_passive(ctx)
        self._submit_passive(ctx)
        self.model.solve(float(ctx.dt))

    def _join_passive(self, ctx) -> None:
        """Join, as agents the others go round, the things that move and stand on the floor but
        have no navigator: a robot driven by its own stack, one standing still, a pushed cart.

        At the first solve of a run, once every navigator has joined in its ``on_reset``. What
        joins is a thing whose root body moves freely: a mocap body, or one with a free joint. A
        thing hinged or sliding on the world -- a door, a gate, a drawer -- moves only where it is
        fixed, and the planner's walls and the probe deal with it. Only an entity whose body is
        that root joins: one nested in another -- a sensor on a robot, an arm on a mobile base --
        rides inside its carrier's disc. Each is a disc around its root body, of the radius its geoms
        reach in the plane (its collidable ones, or every one where none collides), each by its
        shape (:func:`_planar_radius`).
        """
        import mujoco  # noqa: PLC0415 - the module stays importable without a model

        from roqsim.presence import entity_geoms  # noqa: PLC0415

        self._joined_passive = True
        model, data = getattr(ctx, "model", None), getattr(ctx, "data", None)
        entities = getattr(ctx, "entities", None)
        if model is None or entities is None:
            return
        mujoco.mj_forward(model, data)
        for entity in entities.all():
            if entity.name in self._aids or not entity.body:
                continue
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body)
            if bid < 0 or int(model.body_rootid[bid]) != bid or not _free_root(model, bid):
                continue
            geoms = entity_geoms(model, entity, strict=False)
            solid = [g for g in geoms if model.geom_contype[g] or model.geom_conaffinity[g]]
            geoms = solid or geoms
            if not geoms:
                continue
            if _lowest(model, data, geoms) > FLOOR_BAND:
                continue
            radius = _planar_radius(model, data, geoms, data.xpos[bid][:2])
            aid = self.model.add_agent(
                entity.name, radius=radius, max_speed=0.0, yields=False, params={}
            )
            self._aids[entity.name] = aid
            self._passive[entity.name] = (aid, bid, list(geoms))
        for aid, key in self._exclusions:
            other = self._aids.get(key)
            if other is None:
                continue
            try:
                self.model.exclude(aid, other)
            except NotImplementedError:
                raise RegistryError(
                    f"avoidance model {self.ref!r} cannot leave one agent out of another's solve, "
                    f"so a mover's `ignore: [{key}]` cannot apply to its steering. Steer that mover "
                    f"with a model that can (give_way), or drop the ignore."
                ) from None

    def _submit_passive(self, ctx) -> None:
        """Each passive agent where it is, at the velocity it moves at, measured from the tick
        before (a body whose pose is written has no velocity of its own). One lifted off the floor
        -- a tote being carried -- is absent to the others while it is."""
        if not self._passive:
            return
        model, data = ctx.model, ctx.data
        now = float(ctx.sim_time)
        for key, (aid, bid, geoms) in self._passive.items():
            pos = np.array(data.xpos[bid][:2], dtype=float)
            last = self._last.get(key)
            vel = (pos - last[1]) / (now - last[0]) if last and now > last[0] else np.zeros(2)
            self._last[key] = (now, pos)
            entity = ctx.entities.get(key)
            present = bool(entity.present) if entity is not None else True
            lifted = _lowest(model, data, geoms) > FLOOR_BAND
            self.model.submit(aid, pos, vel, np.zeros(2), present=present and not lifted)

    def exclude(self, aid, key: str) -> None:
        """Let agent ``aid`` never give way to the entity ``key``: its navigator's ``ignore``.
        Applied once every agent has joined, at the first solve of the run."""
        self._exclusions.append((aid, key))

    def ensure_reset(self, ctx) -> None:
        """Reset the model once per reset of the world, whichever mover gets here first.

        Stamped like the solve, and for the same reason: every navigator resets, and the shared
        model must be cleared exactly once -- not once per mover, which would wipe the state of the
        movers that reset before it.
        """
        generation = int(ctx.reset_count)
        if generation == self._reset_generation:
            return
        self._reset_generation = generation
        self.reset()

    def reset(self) -> None:
        self._solved_step = -1
        self._last = {}
        self.model.reset()

    # -- the model's own surface, forwarded so a caller holds one object ------------------------
    def add_agent(self, key, **kw):
        aid = self.model.add_agent(key, **kw)
        self._aids[str(key)] = aid
        return aid

    def add_static(self, polygons) -> None:
        self.model.add_static(polygons)

    def submit(self, aid, pos, vel, pref_vel, *, present: bool = True) -> None:
        self.model.submit(aid, pos, vel, pref_vel, present=present)

    def result(self, aid) -> np.ndarray:
        return self.model.result(aid)


def service_for(ctx, spec, base_dir=None) -> AvoidanceService:
    """The world's shared avoidance model, created on first use.

    There is no world-level entry to declare: a navigator that wants to yield says so, and the model
    appears. That is what makes avoidance a property of the mover rather than a checklist item --
    a world-level entry is one a world can leave commented out, with every mover politely
    yielding to nobody.

    ``spec`` is the NAME of a model, or a mapping naming it under ``model`` with its parameters; a
    navigator passes its ``avoidance.steer`` and ``avoidance.params``. The model itself is shared,
    because agents can only avoid each other inside one solver, so **two navigators asking for
    different models is refused** --
    silently giving the second one its own solver would make the two invisible to each other, which
    looks exactly like avoidance that does not work.
    """
    if isinstance(spec, dict):
        wanted = dict(spec)
        ref = str(wanted.pop("model", DEFAULT_MODEL))
    else:
        wanted, ref = {}, str(spec)
    service = ctx.blackboard.get(SERVICE_KEY)
    if service is not None:
        if service.ref != ref:
            raise RegistryError(
                f"this world asks for two different avoidance models, {service.ref!r} and {ref!r}. "
                f"Agents can only avoid each other inside one model, so the movers using one would "
                f"be invisible to the movers using the other. Pick one."
            )
        if wanted and service.params and wanted != service.params:
            raise RegistryError(
                f"avoidance model {ref!r} is configured twice with different parameters "
                f"({service.params} and {wanted}). It is one shared model, so there is one setting."
            )
        return service
    model = resolve_model(ref, base_dir)()
    model.configure(ctx, wanted)
    # The planner's own wall footprints, so a model that reasons about static geometry cannot push an
    # agent through a wall the plan carefully went around. Fed here rather than by each mover: the
    # walls belong to the world, and adding them once per mover would stack duplicates.
    import mujoco  # noqa: PLC0415 - only needed when a model is actually created

    from ..obstacles import wall_polygons  # noqa: PLC0415 - avoids an import cycle at module load

    if ctx is not None and getattr(ctx, "model", None) is not None:
        mujoco.mj_forward(ctx.model, ctx.data)  # geom_xpos must be valid to read footprints
        model.add_static(wall_polygons(ctx.model, ctx.data))
    service = AvoidanceService(model, ref=ref, params=wanted)
    ctx.blackboard.set(SERVICE_KEY, service)
    logger.info("avoidance: %s%s", ref, f" {wanted}" if wanted else "")
    return service


def resolve_model(ref: str, base_dir: Path | None = None) -> type[AvoidanceModel]:
    """Resolve ``ref`` to an :class:`AvoidanceModel` subclass."""
    return resolve(
        ref, group=ENTRY_POINT_GROUP, base=AvoidanceModel, kind="avoidance model", base_dir=base_dir
    )


def available() -> list[str]:
    """Registered model names, for error messages."""
    return registered(ENTRY_POINT_GROUP)


__all__ = [
    "DEFAULT_MODEL",
    "NO_AGENT",
    "SERVICE_KEY",
    "AvoidanceModel",
    "AvoidanceService",
    "RegistryError",
    "available",
    "resolve_model",
    "service_for",
]
