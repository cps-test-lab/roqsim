"""A walker's motion layers and its look: clips played over the gait on some joints, and the head
turned toward a point.

The gait (idle, walk, run, turns) poses the whole body from where the navigator moves it. A layer
plays a clip over it on the joints it names -- a group (``upper``, ``arms``, ``arm_r``, ``head``,
``legs``, ``all``, ...; :data:`roqsim_walker.humanoid.JOINT_GROUPS`) or joints by name -- so a
worker can wave with one arm while walking, or a clip of the whole body can play while it stands.
Layers stack in the order they were started; a hand's reach and the moves of a walker with hands
come over them, and the look over everything. Every method runs on the physics thread.
"""

from __future__ import annotations

import itertools
import math
import numbers

import numpy as np

from roqsim_walker.animation import MotionLayer, load_layer_clip
from roqsim_walker.blueprint import BlueprintError, resolve_clip
from roqsim_walker.humanoid import JOINT_NAMES, joints_of
from roqsim_walker.motion import ClipError


class Layers:
    """One walker's clip player and look (published on its handle, ``walker:<name>``)."""

    def __init__(self, st, blueprint: str, ctx=None):
        self.st = st
        self.blueprint = blueprint
        self.ctx = ctx  # the running world, for a target looked at by name
        self._ids = itertools.count(1)
        self._clips: dict[str, object] = {}

    def clip(self, clip: str):
        """The clip ``clip`` names (a clip of the walker's anim set by name, or a ``.npz`` path),
        grounded on this walker's feet; loaded once."""
        try:
            path = resolve_clip(self.blueprint, clip)
        except BlueprintError as exc:
            raise ValueError(str(exc)) from exc
        if path not in self._clips:
            try:
                self._clips[path] = load_layer_clip(path, self.st.skeleton)
            except ClipError as exc:
                raise ValueError(str(exc)) from exc
        return self._clips[path]

    def play(
        self,
        clip: str,
        joints=("all",),
        weight: float = 1.0,
        loop: bool = False,
        fade: float = 0.3,
        speed: float = 1.0,
    ) -> int:
        """Play ``clip`` over the gait on ``joints`` (groups and joint names), at ``weight`` (0..1)
        over what is under it, once or with ``loop`` round and round, fading in and out over
        ``fade`` seconds, at ``speed`` times its own pace. Returns the layer's id; raises
        :class:`ValueError` for a clip it cannot find or read, or an argument out of range."""
        names, weight, fade, speed = check(joints, weight, loop, fade, speed)
        c = self.clip(clip)
        layer = MotionLayer(
            id=next(self._ids),
            name=str(clip),
            clip=c,
            joints=tuple(JOINT_NAMES.index(n) for n in names),
            weight=weight,
            loop=loop,
            # A clip played once fades in and out within its own length.
            fade=fade if loop else min(fade, (c.num_frames - 1) / c.fps / speed / 2.0),
            speed=speed,
        )
        self.st.layers.append(layer)
        return layer.id

    def stop(self, clip="") -> list[int]:
        """Fade out the layers playing ``clip`` (its name as played, or a layer's id), or every
        layer with ``""``; the ids stopped."""
        stopped = []
        for layer in self.st.layers:
            if clip in ("", None) or clip == layer.name or clip == layer.id:
                layer.stopping = True
                stopped.append(layer.id)
        return stopped

    def ended(self, layer: int) -> bool:
        """Whether the layer ``layer`` has faded out and gone."""
        return layer in self.st.layers_ended

    def playing(self) -> list[dict]:
        """The layers playing now, each as a record (id, clip, joints, loop, t, weight, stopping)."""
        return [layer.describe() for layer in self.st.layers]

    def look_at(self, point) -> None:
        """Turn the head toward ``point``, within its limits, following it as it moves: a target
        as a move names one (a site, body or entity by name, or a world pose;
        :mod:`roqsim_motion.targets`), a world point (3 finite numbers), or a callable giving one.
        ``None`` or ``""`` turns it back to the gait's. Raises :class:`ValueError` for one it
        cannot look at."""
        self.st.look.settled = False  # a new point: not on it until the head has turned
        if point is None or (isinstance(point, str) and point == ""):
            self.st.look.point = None
            return
        if callable(point):
            self.st.look.point = point
            return
        if isinstance(point, (str, dict)):
            from roqsim_motion.targets import make_target

            if self.ctx is None:
                raise ValueError(f"look_at {point!r}: the walker is not in a running world")
            target = make_target(self.ctx.model, self.ctx.entities, point)
            ctx = self.ctx
            self.st.look.point = lambda: target.resolve(ctx.model, ctx.data)[0]
            return
        try:
            values = [float(v) for v in point]
        except (TypeError, ValueError):
            values = []
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError(
                f"look_at: a world point of 3 finite numbers, or None to look where it walks -- "
                f"not {point!r}"
            )
        self.st.look.point = np.array(values)

    def looking(self) -> bool:
        """Whether the head has turned onto its point (or back to the gait), within a degree."""
        lk = self.st.look
        if lk.point is None:
            return lk.w <= 1e-6
        return lk.w >= 1.0 - 1e-6 and lk.settled

    def reset(self) -> None:
        """No layer playing, and the head the gait's."""
        self.st.layers.clear()
        self.st.layers_ended.clear()
        lk = self.st.look
        lk.point, lk.w, lk.rot, lk.settled = None, 0.0, None, False


def check(joints=("all",), weight=1.0, loop=False, fade=0.3, speed=1.0):
    """``play``'s arguments other than the clip, checked: ``(joint names, weight, fade, speed)``;
    raises :class:`ValueError` for one out of range."""
    names = joints_of([joints] if isinstance(joints, str) else list(joints))
    if not isinstance(loop, bool):
        raise ValueError(f"play: 'loop' is true or false, not {loop!r}")
    return (
        names,
        _number("weight", weight, 0.0, 1.0),
        _number("fade", fade, 0.0, math.inf),
        _number("speed", speed, 1e-3, math.inf),
    )


def _number(name: str, value, lo: float, hi: float) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"play: {name!r} is a number, not {value!r}")
    if not lo <= value <= hi:
        bound = f"from {lo:g} to {hi:g}" if hi != math.inf else f"{lo:g} or more"
        raise ValueError(f"play: {name!r} is {bound}, not {value!r}")
    return float(value)
