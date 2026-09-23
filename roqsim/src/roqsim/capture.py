"""Capturing a run: the sample rate, and (see :class:`StateRecorder`) the recording itself.

A sample can only be taken **on a physics step boundary**, so the achievable capture rates are exactly
the rationals ``1/(k·dt)`` for integer ``k >= 1``. Asking for anything else silently mislabels the
result: with the common ``timestep: 0.002`` (500 Hz), 30 fps needs 16.67 steps, the gate fires every 17,
and the real spacing is 29.41 Hz -- a 2% timing error and ~1.2 s of drift per simulated minute.

That is a reason to **snap and report, not refuse**, because ffmpeg's ``-r`` accepts an exact rational:
declaring ``-r 500/17`` gives a timebase equal to the real spacing, with no rounding and no drift. So a
requested rate is snapped to the grid, the *effective* rate is what gets declared everywhere, and how
loudly that is announced depends on how far it moved (see :func:`CaptureRate.report`).

The rate is mostly a **disk** decision. Measured end-to-end (30 s of sim, ``--pacing asap``, so wall
time *is* the loop cost): the default 25 fps is indistinguishable from not recording, and 500 fps --
every single step at ``dt=0.002``, the worst case there is -- costs about **5.5%**. The underlying
``mj_getState`` is a ~0.001 ms memcpy; at 500 Hz what shows up instead is per-sample Python and numpy
overhead (~20 us: the float32 cast, the copy into the write buffer, the binding call). So lower the rate
because the file is large, and know that only an every-step rate is measurable at all.

A sample goes **straight to the file** (see :class:`StateRecorder`): the recording is one mcap file,
written chunk by chunk as the run proceeds, so the footprint does not grow with the run's length and
a run that is killed keeps every chunk that was closed before the kill.

For scale, the thing this design keeps *out* of the loop: one rendered frame is 2-6 ms depending on how
much of the world is in shot (scene-geometry bound, so resolution barely matters). That is three orders
of magnitude more than a sample.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from importlib import metadata
from pathlib import Path

import mujoco
import numpy as np

from . import keys
from .kinematics import body_twist
from .mcap_format import (
    CHANNEL_CLOCK,
    CHANNEL_JOINTS,
    CHANNEL_POSES,
    CHANNEL_STATE,
    CHUNK_SECONDS,
    FORMAT_VERSION,
    JSON_CHANNELS,
    META_ENTITIES,
    META_RECORDING,
    PROFILE,
    ChunkedWriter,
    RecordingError,
    json_bytes,
    ns,
    recording_path,
    register_channels,
)
from .rates import (
    SNAP_NOTABLE,
    SNAP_QUIET,
    GridRate,
    RateError,
    parse_rate,
    physics_rate,
    snap_rate,
)

log = logging.getLogger(__name__)

#: Frames per *simulated* second when nobody says otherwise. On the grid for every timestep in this
#: repo (0.001/0.002/0.0025/0.004/0.005 -> k = 40/20/16/10/8), so the default path never has to
#: announce a snap.
DEFAULT_FPS = 25

#: A capture rate that cannot exist in this world (see the message). The name this module's callers
#: catch it by, and the one :mod:`roqsim.rates` raises for every caller on the grid: the arithmetic is
#: shared, so the error has to be the same class or half of it would escape an ``except`` here.
CaptureError = RateError


def parse_fps(text: str | int | float | Fraction) -> Fraction:
    """Parse a capture rate, naming the flag it was typed on when it cannot be read.

    :func:`roqsim.rates.parse_rate` with the flag in front of its message: a rate that came in through
    ``--capture-fps`` is one a person can retype, and a message that does not say where it came from
    does not help them.
    """
    try:
        return parse_rate(text)
    except RateError as err:
        raise CaptureError(f"--capture-fps {err}") from err


@dataclass(frozen=True)
class CaptureRate(GridRate):
    """A capture rate that exists in this world: ``every`` steps, i.e. exactly ``fps`` per sim second.

    A :class:`~roqsim.rates.GridRate` in a capture's own vocabulary -- ``fps`` for the rate, ffmpeg's
    rational for the timebase, and a report worded for the flag the rate was typed on.
    """

    @property
    def fps(self) -> Fraction:
        """The effective rate as a capture spells it: what ffmpeg is told and a recording carries."""
        return self.hz

    def ffmpeg_rate(self) -> str:
        """The rate as an exact rational for ffmpeg's ``-r``, e.g. ``500/17``.

        Never a rounded decimal: ``-r 29.41`` on a stream whose real spacing is ``500/17`` drifts, which
        is the whole defect this exists to avoid.
        """
        return self.rational()

    def report(self, logger: logging.Logger | None = None) -> None:
        """Announce the snap in proportion to how far it moved: silent, a note, or a warning.

        The bands matter because the common case must not be noisy (25 fps is on every real timestep's
        grid, so it snaps by nothing) while a 30 -> 29.412 move is a 2% timing difference somebody may
        care about, and one they cannot see any other way.
        """
        logger = logger or log
        if self.deviation <= 0:
            return
        detail = (
            f"{float(self.fps):.3f} fps (every {self.every} steps; exactly {self.ffmpeg_rate()})"
        )
        if self.deviation < SNAP_QUIET:
            logger.debug("capture: %s -> %s", float(self.requested), detail)
        elif self.deviation < SNAP_NOTABLE:
            logger.info("capture: --capture-fps %s snapped to %s", float(self.requested), detail)
        else:
            nearby = ", ".join(f"{float(n.fps):g} (k={n.every})" for n in self.neighbours())
            logger.warning(
                "capture: --capture-fps %s snapped to %s -- samples land on physics steps and this "
                "world steps at %s Hz, so %s/%s = %.2f is not reachable. The timebase is exact, so "
                "there is no drift. Nearby: %s; a world stepping at a multiple of %s Hz would hold "
                "the requested rate exactly.",
                float(self.requested),
                detail,
                float(self.physics),
                float(self.physics),
                float(self.requested),
                float(self.physics / self.requested),
                nearby,
                float(self.requested),
            )


def snap_fps(fps: str | int | float | Fraction, dt: float) -> CaptureRate:
    """Snap a requested capture rate onto this world's physics grid. Refuses only the impossible.

    Hard errors are limited to rates that cannot exist at all -- non-positive, or faster than the
    simulation steps -- because everything else has a nearest achievable answer, and an exact rational
    timebase makes taking it harmless. A capture rate is typed as a flag, so refusing it names the
    flag and the caller can type another; see :func:`roqsim.rates.snap_rate` for the same grid without
    that door, which is what a bridge binding an endpoint needs.
    """
    requested = parse_fps(fps)
    rate = physics_rate(dt)
    if requested <= 0:
        raise CaptureError(f"--capture-fps {float(requested):g}: must be positive")
    if requested > rate:
        raise CaptureError(
            f"--capture-fps {float(requested):g} is faster than this world steps "
            f"({float(rate):g} Hz, timestep {dt:g}): a sample can only be taken on a physics step. "
            f"Use at most {float(rate):g}, or lower the world's sim.timestep."
        )
    snapped = snap_rate(requested, dt)
    return CaptureRate(snapped.hz, snapped.every, snapped.requested, snapped.physics)


# ==================================================================================================
# The recording
# ==================================================================================================

#: The state a recording stores, and **not** ``mjSTATE_FULLPHYSICS``.
#:
#: This is MuJoCo's *own* notion of a saved state: ``mjModel``'s keyframe fields are exactly
#: ``key_time, key_qpos, key_qvel, key_act, key_ctrl, key_mpos, key_mquat``, so one sample here is one
#: MuJoCo keyframe, plus ``plugin`` and ``eq_active`` (both change the pose solution and both are state
#: MuJoCo exposes). ``mjSTATE_FULLPHYSICS`` is the thing that *deviates* from that notion: it is
#: ``time|qpos|qvel|act|plugin`` and drops ``ctrl`` and the mocap fields.
#:
#: Dropping them is a **correctness bug, not an optimisation**, because this substrate drives a lot
#: through mocap and ``ctrl``: ``walker`` uses one mocap body per skeleton joint (17 per pedestrian),
#: ``moving_box`` writes a mocap pose every step, and ``door`` writes ``data.ctrl``. A ``FULLPHYSICS``
#: recording therefore replays every pedestrian and moving prop **frozen at its compile-time pose**, and
#: every door driven toward 0 instead of its commanded opening, while the robot moves correctly -- a
#: plausible-looking video that is wrong.
#:
#: ``xfrc_applied`` is deliberately excluded: it is ``6*nbody`` (1716 near-always-zero values on a
#: 286-body world) and would dominate the file for nothing. That is also why ``mjSTATE_INTEGRATION``,
#: which includes it, is the wrong shortcut.
STATE_SPEC = int(
    mujoco.mjtState.mjSTATE_TIME
    | mujoco.mjtState.mjSTATE_QPOS
    | mujoco.mjtState.mjSTATE_QVEL
    | mujoco.mjtState.mjSTATE_ACT
    | mujoco.mjtState.mjSTATE_CTRL
    | mujoco.mjtState.mjSTATE_MOCAP_POS
    | mujoco.mjtState.mjSTATE_MOCAP_QUAT
    | mujoco.mjtState.mjSTATE_PLUGIN
    | mujoco.mjtState.mjSTATE_EQ_ACTIVE
)

#: The spec's fields in the order ``mj_getState`` packs them, recorded in the provenance so a reader
#: never has to guess a layout, and so a MuJoCo release that reorders or extends the packing is
#: detectable rather than silently misread.
STATE_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "mocap_pos",
    "mocap_quat",
    "plugin",
    "eq_active",
)

#: What ``w`` is measured from. **Elapsed seconds, never a Unix timestamp**: the origin is the moment
#: this recorder was constructed, so the first sample is a few milliseconds rather than 1.7e9. Two
#: reasons the epoch is the wrong choice here. A float64 holding 1.7e9 has ~0.2 us of resolution left,
#: which is coarse against per-step costs measured in microseconds, whereas an elapsed value keeps
#: nanoseconds all run. And ``perf_counter`` is *monotonic*: an NTP step or a DST change mid-run cannot
#: make the column go backwards, which a wall calendar can. The ``poses``, ``joints`` and ``clock``
#: channels carry the epoch instead, because they exist for readers *outside* the process, who have
#: calendar stamps of their own to relate to them; ``wall_start_epoch`` in the provenance ties the two.
WALL_CLOCK_ORIGIN = "recorder start (elapsed seconds from time.perf_counter, monotonic)"

#: Camera-track width: type, fixedcamid, trackbodyid, lookat(3), distance, azimuth, elevation.
CAMERA_WIDTH = 9

#: Decimals a pose, twist or joint value is written with in the JSON channels. Micrometres and
#: microradians: below anything a trial is judged on, and a third of the text a full float costs.
_JSON_DECIMALS = 6

#: MuJoCo joint types that reduce to one scalar: what the ``joints`` channel carries.
_SCALAR_JOINTS = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))

#: The environment variables that narrow what the ``poses`` and ``joints`` channels carry.
RECORD_TRACKS_VAR = "ROQSIM_RECORD_TRACKS"
RECORD_EXCLUDE_VAR = "ROQSIM_RECORD_EXCLUDE"

_PROVENANCE_PACKAGES = ("roqsim", "mujoco", "numpy")


def _named_bodies(model) -> tuple[list[tuple[int, str]], list[str]]:
    """Every named body, in body order, and the parents of the unnamed ones left out.

    All of them rather than only those parented to the world: what a trial's success rule reads is
    often welded below a robot -- a tool on a flange, a workpiece in a gripper -- and a consumer
    cannot know in advance which one it will need. The price is rows, several times more on a
    manipulator world than on a mobile one. The ``state`` channel remains the complete record (sites,
    and anything between samples, are derivable only from it).

    An unnamed body has no key to be recorded under, so it is left out and reported instead: a tool
    missing from the record then shows up in the run log rather than as an absent row.
    """
    out, skipped = [], []
    for bid in range(1, model.nbody):  # 0 is the world body itself
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name:
            out.append((bid, name))
        else:
            parent = int(model.body_parentid[bid])
            skipped.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) or "world")
    return out, skipped


def _scalar_joints(model) -> list[tuple[int, str, int]]:
    """Every named hinge/slide joint as ``(id, name, qposadr)``, in model order.

    An unnamed joint is skipped: the channel is keyed by name, so a value a reader cannot key onto
    its joint is dead weight in the file.
    """
    out = []
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) not in _SCALAR_JOINTS:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name:
            out.append((jid, name, int(model.jnt_qposadr[jid])))
    return out


def package_versions() -> dict:
    """Versions that decide whether a recording can be reproduced elsewhere.

    ``numpy`` is in here for a specific reason: ``rng.choice(..., replace=False)``'s consumption is
    implementation-dependent, so bit-identical *noise* replay is pinned to a numpy version even though
    the physics is not.
    """
    out = {}
    for name in _PROVENANCE_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # pragma: no cover - a source checkout
            out[name] = "unknown"
    return out


# -- which bodies and joints are recorded ------------------------------------------------------------


def parse_patterns(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """A comma-separated string (the environment's spelling) or a sequence, as clean patterns."""
    if value is None:
        return ()
    items = value.split(",") if isinstance(value, str) else [str(v) for v in value]
    return tuple(item.strip() for item in items if item and item.strip())


def _segments_match(pattern: list[str], key: list[str]) -> bool:
    """Match ``/``-separated segments: ``*`` is one segment, ``**`` is one or more, else literal."""
    if not pattern:
        return not key
    head, rest = pattern[0], pattern[1:]
    if head == "**":
        return any(_segments_match(rest, key[i:]) for i in range(1, len(key) + 1))
    if not key:
        return False
    if head != "*" and head != key[0]:
        return False
    return _segments_match(rest, key[1:])


def pattern_matches(pattern: str, key: str) -> bool:
    """Whether a track pattern selects ``key`` (``<entity>/<local name>`` or a bare name)."""
    return _segments_match(pattern.split("/"), key.split("/"))


def _local_name(name: str, entity: str, prefix: str) -> str:
    """A body's or joint's name in the entity's own words: its spawn prefix or ``<entity>/`` removed."""
    if prefix and name.startswith(prefix):
        return name[len(prefix) :]
    if name.startswith(entity + "/"):
        return name[len(entity) + 1 :]
    return name


def _entities_of(model, registry) -> list[tuple[str, str, set[int]]]:
    """Every registered entity with a body in this model: ``(name, prefix, body ids in its subtree)``.

    The prefix is the one its spawn plugin recorded in ``meta`` -- what turns ``robot/base_link`` in
    the compiled model back into ``base_link``, so a pattern is written in the words a world document
    uses rather than the ones the model happens to carry.
    """
    if registry is None:
        return []
    try:
        entities = list(registry.all())
    except Exception:  # noqa: BLE001 - a driver with no usable registry records unnamed tracks
        return []
    parents = [int(p) for p in model.body_parentid]

    def under(bid: int, root: int) -> bool:
        while bid > 0:
            if bid == root:
                return True
            bid = parents[bid]
        return False

    out = []
    for entity in entities:
        if not entity.body:
            continue
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(entity.body))
        if root < 0:
            continue
        meta = entity.meta if isinstance(getattr(entity, "meta", None), dict) else {}
        subtree = {bid for bid in range(1, model.nbody) if under(bid, root)}
        out.append((str(entity.name), str(meta.get("prefix") or ""), subtree))
    return out


def select_tracks(
    model,
    registry,
    tracks: str | Sequence[str] | None = None,
    exclude: str | Sequence[str] | None = None,
) -> tuple[list[tuple[int, str]], list[tuple[int, str, int]], list[str]]:
    """The bodies and joints the ``poses`` and ``joints`` channels carry, and the unnamed bodies left out.

    Every named body and every named hinge or slide joint by default. ``tracks`` narrows that to what
    its patterns select and ``exclude`` removes what its patterns select, an exclude winning over an
    include. A pattern is ``<entity>/<body-or-joint>`` -- ``**`` for everything of that entity, ``*``
    for one name segment -- or a bare name, which selects the body and the joint of that name alike.

    A pattern that selects nothing **refuses**, naming the entities and the bodies and joints that
    exist: a typo would otherwise record a run that looks complete and lacks the one track the trial
    is judged on. The ``state`` and ``clock`` channels are never narrowed.
    """
    bodies, skipped = _named_bodies(model)
    joints = _scalar_joints(model)
    includes, excludes = parse_patterns(tracks), parse_patterns(exclude)
    if not includes and not excludes:
        return bodies, joints, skipped

    entities = _entities_of(model, registry)
    # Every name a body or joint answers to: its own, and ``<entity>/<local>`` per entity it is under.
    body_keys: dict[str, set[str]] = {name: {name} for _, name in bodies}
    joint_keys: dict[str, set[str]] = {name: {name} for _, name, _ in joints}
    for entity, prefix, subtree in entities:
        for bid, name in bodies:
            if bid in subtree:
                body_keys[name].add(f"{entity}/{_local_name(name, entity, prefix)}")
        for jid, name, _ in joints:
            if int(model.jnt_bodyid[jid]) in subtree:
                joint_keys[name].add(f"{entity}/{_local_name(name, entity, prefix)}")

    def selected(pattern: str) -> tuple[set[str], set[str]]:
        hit_bodies = {
            n for n, ks in body_keys.items() if any(pattern_matches(pattern, k) for k in ks)
        }
        hit_joints = {
            n for n, ks in joint_keys.items() if any(pattern_matches(pattern, k) for k in ks)
        }
        if not hit_bodies and not hit_joints:
            named = ", ".join(f"{n} (prefix {p!r})" if p else n for n, p, _ in entities)
            raise RecordingError(
                f"record track pattern {pattern!r} matches no body and no joint of this world. "
                f"Entities: {named or 'none registered'}. "
                f"Bodies: {', '.join(n for _, n in bodies) or 'none'}. "
                f"Joints: {', '.join(n for _, n, _ in joints) or 'none'}. A pattern is "
                "<entity>/<body-or-joint> (** for everything of the entity, * for one name "
                "segment) or a bare body or joint name."
            )
        return hit_bodies, hit_joints

    keep_bodies = {n for _, n in bodies}
    keep_joints = {n for _, n, _ in joints}
    if includes:
        keep_bodies, keep_joints = set(), set()
        for pattern in includes:
            hit_bodies, hit_joints = selected(pattern)
            keep_bodies |= hit_bodies
            keep_joints |= hit_joints
    for pattern in excludes:
        hit_bodies, hit_joints = selected(pattern)
        keep_bodies -= hit_bodies
        keep_joints -= hit_joints
    return (
        [(bid, n) for bid, n in bodies if n in keep_bodies],
        [(jid, n, adr) for jid, n, adr in joints if n in keep_joints],
        skipped,
    )


# -- the provenance ----------------------------------------------------------------------------------


def _actuator_record(ctx) -> dict:
    """The resolved actuator table per entity, as plain data, or ``{}`` when nothing filled it.

    Reads what the spawn plugins published at ``configure``; a world whose spawn plugins predate this
    simply records nothing rather than failing, which is what keeps an embedding driver working.
    """
    tables = getattr(ctx, "actuator_tables", None) or {}
    return {entity: [row.as_record() for row in rows] for entity, rows in tables.items() if rows}


def _endpoint_rate_record(ctx) -> list:
    """What each published endpoint actually goes out at, as plain data, or ``[]`` when nothing bound.

    Reads what the bridges wrote at ``configure``; a run with no transport records nothing rather
    than failing. The rate is in there twice on purpose: ``requested_hz`` is the number the world
    asked for and quotes everywhere, ``realised_hz`` is the one the run published at, and they differ
    whenever the request is not a whole number of physics steps. Nobody reading a rate afterwards can
    tell those apart from the world document alone, and the exact rational is recoverable from
    ``every_steps`` and the ``timestep`` recorded beside this.
    """
    rows = getattr(ctx, "endpoint_rates", None) or []
    return [dict(row) for row in rows]


def _roster(registry) -> list[dict] | None:
    """The entity registry as the ``roqsim.entities`` document, or ``None`` without a usable one."""
    if registry is None:
        return None
    try:
        return [
            {"name": e.name, "kind": e.kind, "body": e.body, "present": bool(e.present)}
            for e in registry.all()
        ]
    except Exception:  # noqa: BLE001 - a driver with no usable registry, not a failure
        return None


def decimated(rec, factor: int, out: str | Path) -> Path:
    """Write a copy of ``rec`` keeping every ``factor``-th sample, at 1/``factor`` of its rate.

    For a recording captured far above the rate anything will play it back at. ``roqsim render``
    renders one frame per sample and never decimates -- deliberately, so every frame in a video is a
    state the simulation actually had -- which means a 250 Hz recording rendered for a 30 fps video
    draws eight frames for every one that survives. Dropping them first costs the same pictures and a
    fraction of the rendering.

    That invariant is preserved here rather than traded away: the samples that remain are untouched
    messages, so every frame drawn from the result is still a state the simulation had. What changes
    is only how many of them there are, and the declared rate that says so. The ``poses``, ``joints``
    and ``clock`` messages of the kept samples travel with them.

    The rate stays exact because ``capture_fps`` is a ``[numerator, denominator]`` pair: 250 Hz
    decimated by 8 is ``[250, 8]``, i.e. 31.25 fps, not a rounded 31.
    """
    factor = int(factor)
    if factor < 1:
        raise RecordingError(f"decimate factor must be 1 or more, got {factor}")
    kept = range(0, len(rec), factor)
    if len(kept) < 2:
        raise RecordingError(
            f"decimating {rec.path} by {factor} would leave {len(kept)} sample(s) of its "
            f"{len(rec)}; a recording needs at least two to have a span."
        )
    num, den = rec.meta["capture_fps"]
    meta = {**rec.meta, "capture_fps": [int(num), int(den) * factor], "samples": len(kept)}
    expected = record_dtype(int(rec.meta["state_size"]), rec.has_camera)
    if rec.samples.dtype != expected:
        raise RecordingError(
            f"{rec.path}: samples are {rec.samples.dtype}, but its provenance describes {expected}."
        )
    out = recording_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = ChunkedWriter(out)
    try:
        writer.start(PROFILE, library=_library())
        writer.add_json_metadata(META_RECORDING, meta)
        if rec.entities is not None:
            writer.add_json_metadata(META_ENTITIES, {"entities": rec.entities})
        channels = register_channels(writer)
        state_times = rec.message_times(CHANNEL_STATE)
        for index in kept:
            log_time, publish_time = state_times[index]
            writer.add_message(
                channels[CHANNEL_STATE], log_time, rec.samples[index].tobytes(), publish_time
            )
            for topic, _name, _schema in JSON_CHANNELS:
                message = rec.message(topic, index)
                if message is not None:
                    writer.add_message(
                        channels[topic], message.log_time, message.data, message.publish_time
                    )
        writer.finish()
    except BaseException:
        writer.abandon()
        raise
    return out


def _library() -> str:
    return f"roqsim {package_versions().get('roqsim', 'unknown')}"


class StateRecorder:
    """Sample MuJoCo state into an mcap file while a run proceeds. A **driver** object, not a plugin.

    Capture is a session concern, not an experiment one -- the same footing as ``sim.headless`` (which
    the world YAML explicitly rejects), ``--left-ui`` and ``--manual-control``. So this is constructed by
    a driver and ``sample``\\ d from the loop the driver already runs: no lifecycle hooks, nothing
    injected into a parsed world, and no second route through the world YAML.

    Cost on the run is one ``mj_getState`` (~0.001 ms, about a fiftieth of a physics step), one
    ``mj_objectVelocity`` per recorded body, and the JSON of the three decoded channels. Everything
    expensive -- rebuilding the world, rendering, encoding -- happens afterwards, from the file (see
    :mod:`roqsim.recording`).

    **One file, four channels, written as the run goes.** Every sample is one message on each of
    ``state`` (the MuJoCo state vector, the primary artifact), ``poses`` (every recorded body's world
    pose and twist), ``joints`` (every recorded scalar joint) and ``clock`` (the wall/sim pair a reader
    outside the process relates its own stamps to). The provenance and the entity roster are metadata
    records. The file is chunked and compressed, and a chunk is closed and flushed **at least once per
    wall second** (:data:`roqsim.mcap_format.CHUNK_SECONDS`), so the run's memory does not grow with
    its length and a hard kill loses at most the open chunk.

    Every stop that must work reaches :meth:`close` through the driver's existing ``finally``: closing
    the viewer window drops ``viewer.is_running()``, and Ctrl+C **or a supervisor's SIGTERM** is caught
    by ``_graceful_stop``, which sets ``QUITTING`` rather than raising. SIGTERM matters as much as
    Ctrl+C here, because it is how a supervised run ends -- a container teardown, a ``docker stop``, an
    eviction, a campaign timeout. ``close`` writes the summary section that marks a finished file;
    **SIGKILL** skips it, and the file it leaves still opens with every closed chunk in it. Its missing
    summary is what says the run did not end on purpose.
    """

    def __init__(
        self,
        ctx,
        path: str | Path,
        rate: CaptureRate,
        *,
        world: str | None = None,
        overrides: dict | None = None,
        config=None,
        camera: bool = False,
        tracks: str | Sequence[str] | None = None,
        exclude: str | Sequence[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.path = recording_path(path)
        self.rate = rate
        self.log = logger or log
        self._model = ctx.model
        self._size = mujoco.mj_stateSize(ctx.model, STATE_SPEC)
        self._buf = np.empty(self._size)  # mj_getState needs float64; samples are cast on write
        self._camera = bool(camera)
        # The record layout is decided here rather than at close: by then there is nothing left in
        # memory to infer it from.
        self._record = np.zeros(1, dtype=record_dtype(self._size, camera))
        # Views onto the record's fields, taken once. Naming a field of a structured array builds a
        # new view every time, which measured as much again as the write it feeds.
        self._t, self._w, self._s = self._record["t"], self._record["w"], self._record["s"]
        self._cam = self._record["cam"] if camera else None
        # The roster that says what the pose rows are. Held as a live reference to the registry, not
        # a copy: an entity spawned or removed mid-run changes the answer, and a snapshot taken at
        # construction would describe a world the trial has since left.
        self._registry = getattr(ctx, "entities", None)
        # Which bodies and joints the decoded channels carry. Decided at construction, so a pattern
        # that selects nothing fails the run's start rather than its analysis.
        self._bodies, self._joints, self._skipped = select_tracks(
            ctx.model, self._registry, tracks, exclude
        )
        #: Last roster written, so the record is rewritten when it changes and not once per sample.
        self._entities_sig: tuple | None = None
        self._ctx = ctx
        self._writer: ChunkedWriter | None = None
        self._channels: dict[str, int] = {}
        self._count = 0
        # Span endpoints, kept because the closing log line reports them and nothing else remembers.
        self._first_t = self._last_t = 0.0
        self._first_w = self._last_w = 0.0
        self._next_due = 0.0
        self._closed = False
        self._last_chunk = 0.0
        # Origin for the wall column, taken before any sample so the series starts at ~0. A *take*
        # started by F9 mid-session gets its own origin, which is what makes each take's real-time
        # factor its own rather than the session's.
        self._origin = time.perf_counter()
        self._wall_start_epoch = time.time()
        self._provenance = {
            "format_version": FORMAT_VERSION,
            # The seed belongs in the provenance because it is what makes a *sensor* replay exact: a
            # recomputed lidar scan needs the same noise the live run drew. The episode travels with
            # it for the same reason and is useless without it: the noise key is (seed, episode,
            # sim_time, sensor), so a recording of the third trial replayed as the first would draw
            # a different -- and entirely plausible-looking -- scan.
            "seed": getattr(ctx, "seed", None),
            "episode": int(getattr(ctx, "episode", 0)),
            "world": world,
            # The recipe: what a reader needs to see how this run was asked for, and what an external
            # consumer uses as world identity. Replay does NOT re-interpret it -- see `world_model`.
            "overrides": overrides or {},
            # What actually ran: the resolved component tree and the sim block it ran with. Recorded
            # outright so rebuilding is a read rather than a re-resolution, which is what keeps a
            # recording valid across a change to the override grammar.
            "world_model": config.as_record() if config is not None else None,
            # What each joint actually ran under. The world_model above carries what a world
            # DECLARED, which is only half the answer: a model's own gains are the other half, and
            # an `actuators:` block that changes one joint leaves the rest reported by nothing. This
            # is the resolved table -- every actuator, its law, its gains, and whether the value came
            # from the model or from the world -- so a reader can state the gains a run used without
            # opening the MJCF and re-deriving them.
            "actuators": _actuator_record(ctx),
            # What each published endpoint went out at, requested and realised. A publish lands on a
            # physics step, so a rate that is not a whole number of steps is served at a neighbouring
            # one; this is where that shows without reading a log. Written again at close, since a
            # bridge may bind after the first sample.
            "endpoint_rates": _endpoint_rate_record(ctx),
            "packages": package_versions(),
            "state_spec": STATE_SPEC,
            "state_fields": list(STATE_FIELDS),
            "state_size": self._size,
            "dtype": "float32",
            "capture_fps": [rate.fps.numerator, rate.fps.denominator],
            "capture_every_steps": rate.every,
            "timestep": float(ctx.model.opt.timestep),
            # Both spellings of the same fact: ``camera`` states the ``state`` message's layout
            # beside ``state_size``/``state_fields``; ``camera_track`` is the name a reader asks by.
            "camera": self._camera,
            "camera_track": self._camera,
            "wall_clock_origin": WALL_CLOCK_ORIGIN,
            # The calendar instant ``w == 0`` corresponds to. ``w`` itself stays elapsed and
            # monotonic for the reasons WALL_CLOCK_ORIGIN gives -- nanosecond resolution, and
            # immunity to an NTP step mid-run -- but a *reader* outside this process has wall
            # stamps of its own to relate to it: a container log's lines, a rosbag's receive
            # times. One number completes the record; converting the column would break it.
            "wall_start_epoch": self._wall_start_epoch,
            # What the decoded channels carry, so a reader can tell "not recorded" from "did not move".
            "tracks": {
                "bodies": [name for _, name in self._bodies],
                "joints": [name for _, name, _ in self._joints],
            },
            "model": {
                "name": _model_name(ctx.model),
                "nbody": int(ctx.model.nbody),
                "ngeom": int(ctx.model.ngeom),
                "nq": int(ctx.model.nq),
                "nv": int(ctx.model.nv),
                "nu": int(ctx.model.nu),
                "nmocap": int(ctx.model.nmocap),
            },
        }

    # -- recording -------------------------------------------------------------------------------

    @property
    def frames(self) -> int:
        return self._count

    def _open(self) -> None:
        """Create the file: header, provenance, roster, channels. On the first sample, not before.

        Lazily, so a recorder that never samples leaves no file for an existence check to trip
        over -- a run that ends before its first sample is due has nothing to say.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = ChunkedWriter(self.path)
        try:
            writer.start(PROFILE, library=_library())
            writer.add_json_metadata(META_RECORDING, self._provenance)
            self._channels = register_channels(writer)
            # On disk before the first chunk closes: a reader can already see what the file is.
            writer.flush()
        except BaseException:
            writer.abandon()
            raise
        self._writer = writer
        self._last_chunk = time.perf_counter()
        if self._skipped:
            skipped = sorted(set(self._skipped))
            self.log.info(
                "recording: the poses channel carries %d named bodies; %d unnamed bodies have no "
                "entry (under %s)",
                len(self._bodies),
                len(self._skipped),
                ", ".join(skipped),
            )
        self._write_entities()

    def sample(self, ctx, cam=None) -> bool:
        """Take a sample if one is due. Cheap enough to call every step; returns whether it did.

        Gated on ``ctx.sim_time``, so the rate is per *simulated* second and a recording plays back at
        1x sim time whatever wall-clock pacing the run used. The wall clock is *recorded* rather than
        gated on, which is what makes the pacing itself a measurable property of the run instead of a
        thing the sample schedule hides.
        """
        now = ctx.sim_time
        if now + 1e-12 < self._next_due:
            return False
        # Absolute schedule, so a long step cannot drag the whole series late; resynchronise after a
        # gap (a reset, a paused window) rather than firing a burst of catch-up samples.
        period = self.rate.period
        self._next_due = now + period if self._next_due <= 0 else self._next_due + period
        if self._next_due <= now:
            self._next_due = now + period

        # Stamped next to the state copy, so ``w`` says when this state was taken and not when the
        # step's bookkeeping around it finished.
        wall = time.perf_counter() - self._origin
        sim = float(now)
        mujoco.mj_getState(ctx.model, ctx.data, self._buf, STATE_SPEC)
        self._t[0] = sim
        self._w[0] = wall
        # float64 -> float32 happens in the assignment, which is why there is no per-sample astype.
        self._s[0] = self._buf
        if self._cam is not None:
            self._cam[0] = _camera_row(cam)
        try:
            if self._writer is None:
                self._open()
            self._write_sample(ctx, sim, wall)
        except OSError as err:
            raise RecordingError(
                f"{self.path}: the recording could not be written ({err}). This is the recording "
                "itself, so the run stops here rather than continuing toward a file that would "
                "silently be missing samples."
            ) from err
        self._count += 1
        if self._count == 1:
            self._first_t, self._first_w = sim, wall
        self._last_t, self._last_w = sim, wall
        return True

    def _write_sample(self, ctx, sim: float, wall: float) -> None:
        """One message per channel for this sample, then the roster if it changed, then the chunk."""
        writer = self._writer
        epoch = self._wall_start_epoch + wall
        log_time, publish_time = ns(sim), ns(epoch)
        writer.add_message(
            self._channels[CHANNEL_STATE], log_time, self._record.tobytes(), publish_time
        )
        writer.add_message(
            self._channels[CHANNEL_POSES],
            log_time,
            json_bytes({"t": sim, "w": epoch, "bodies": self._pose_rows(ctx)}),
            publish_time,
        )
        qpos = ctx.data.qpos
        writer.add_message(
            self._channels[CHANNEL_JOINTS],
            log_time,
            json_bytes(
                {
                    "t": sim,
                    "w": epoch,
                    "q": {
                        name: round(float(qpos[adr]), _JSON_DECIMALS)
                        for _, name, adr in self._joints
                    },
                }
            ),
            publish_time,
        )
        writer.add_message(
            self._channels[CHANNEL_CLOCK],
            log_time,
            json_bytes({"wall_ts": epoch, "sim_ts": sim}),
            publish_time,
        )
        self._write_entities()
        # Judged against the clock now rather than the sample's stamp: on the first sample the file
        # was opened between the two, and the reference was taken at the open.
        now = time.perf_counter()
        if now - self._last_chunk >= CHUNK_SECONDS:
            writer.close_chunk()
            self._last_chunk = now

    def _pose_rows(self, ctx) -> dict[str, list[float]]:
        """This sample's world pose and twist, one row per recorded body.

        The velocities come from the solver rather than from differencing the positions, which is the
        point of the channel: a difference is only ever as good as the interval it is divided by, and
        an arrival-time interval is not the interval the motion happened over. It is also the only
        pose series a **stepped** run produces at all -- with no ROS there is no rosbag and so no TF
        to derive poses from afterwards.

        One convention, stated because it is easy to get wrong and impossible to see: ``mj_step``
        integrates ``qpos`` and then leaves ``xpos`` holding the pose from *before* that integration,
        so the row is a coherent snapshot of ``sim - dt`` carrying the label ``sim``. That is
        deliberately the same one-step lag the ``ground_truth_pose`` plugin publishes with, so this
        channel and the TF one describe the same instant and any difference between them is transport
        rather than convention. It cancels in every derivative.
        """
        data = ctx.data
        rows: dict[str, list[float]] = {}
        r = _JSON_DECIMALS
        for bid, name in self._bodies:
            pos, quat = data.xpos[bid], data.xquat[bid]
            twist = body_twist(ctx.model, data, bid)
            rows[name] = [
                round(float(pos[0]), r),
                round(float(pos[1]), r),
                round(float(pos[2]), r),
                # MuJoCo orders a quaternion (w, x, y, z); the channel orders it (x, y, z, w), as
                # ROS does. Reordered here rather than at the reader, once.
                round(float(quat[1]), r),
                round(float(quat[2]), r),
                round(float(quat[3]), r),
                round(float(quat[0]), r),
                round(twist.linear[0], r),
                round(twist.linear[1], r),
                round(twist.linear[2], r),
                round(twist.angular[0], r),
                round(twist.angular[1], r),
                round(twist.angular[2], r),
            ]
        return rows

    def _write_entities(self) -> None:
        """Keep the ``roqsim.entities`` metadata matching the registry, writing it only on a change.

        Rewritten on a change because a run can spawn, remove or hide an entity -- a roster written
        once at the first sample would describe the world the trial started in rather than the one
        it is in. A reader takes the last record of the name.

        The comparison is over the whole roster including ``present``, which costs an iteration of a
        registry holding tens of entities against a sample that has already copied the entire
        physics state and formatted a row per body.
        """
        roster = _roster(self._registry)
        if roster is None:
            self._registry = None
            return
        signature = tuple((e["name"], e["kind"], e["body"], e["present"]) for e in roster)
        if signature == self._entities_sig:
            return
        self._writer.add_json_metadata(META_ENTITIES, {"entities": roster})
        self._entities_sig = signature

    def on_reset(self) -> None:
        """Start the schedule over. A rebuilt or reset world is a new series, not a continuation.

        The wall origin deliberately does **not** move with it: real time did not restart, and a reset
        that took 4 s of rebuilding is exactly the kind of thing this column exists to show.
        """
        self._next_due = 0.0

    def replay(self, ctx):
        """Yield ``(sim_time, data)`` for every sample taken, posed on ``ctx``'s **live** model.

        The counterpart to :mod:`roqsim.recording` for a driver that still *has* the world: it restores
        each sample in place instead of rebuilding from the file, which is what lets a shutdown hook
        derive something from the run without paying seconds for a rebuild it does not need.

        Reads the samples back from the file, closing the open chunk first, so it works either side
        of :meth:`close`.

        **Destructive and terminal**: it overwrites ``ctx.data`` sample by sample, so it belongs after
        the loop is done. One ``MjData`` is re-posed throughout, so a consumer that keeps values must
        copy them.
        """
        if self._writer is None:
            return
        if not self._closed:
            self._writer.close_chunk()
        from .recording import open_recording

        samples = open_recording(self.path).samples
        buf = np.empty(self._size)
        for record in samples:
            t = float(record["t"])
            buf[:] = record["s"]
            mujoco.mj_setState(ctx.model, ctx.data, buf, STATE_SPEC)
            # xpos/xquat/site_xpos are derived, never stored -- this is what makes them available.
            mujoco.mj_forward(ctx.model, ctx.data)
            yield t, ctx.data

    def close(self) -> Path | None:
        """Finish the recording and return its path, or ``None`` when there was nothing to write.

        Idempotent, because every call site is a ``finally`` and a driver may unwind more than once.
        The provenance is written a second time here with what is only known at the end (the
        endpoint rates a bridge bound after the start, the sample count and span); a reader takes
        the last record.
        """
        if self._closed:
            return None
        self._closed = True
        if self._writer is None:
            # Nothing sampled: say so rather than leaving an empty file that passes an existence check.
            self.log.warning(
                "recording: no samples taken, so %s was not written. The run ended before the first "
                "sample was due at %s fps (every %.3f s of sim time).",
                self.path,
                float(self.rate.fps),
                self.rate.period,
            )
            return None
        sim_span = self._last_t - self._first_t
        wall_span = self._last_w - self._first_w
        self._provenance["endpoint_rates"] = _endpoint_rate_record(self._ctx)
        self._provenance["samples"] = self._count
        self._provenance["span"] = [self._first_t, self._last_t]
        try:
            self._writer.add_json_metadata(META_RECORDING, self._provenance)
            self._writer.finish()
        except OSError as err:
            self._writer.abandon()
            raise RecordingError(
                f"{self.path}: the recording could not be finished ({err})"
            ) from err
        self.log.info(
            # Wall gets two decimals where sim gets one: a fast `--pacing asap` run finishes in
            # hundredths of a second, and "in 0.0 s wall" would report a real measurement as nothing.
            "recording: %d samples at %s fps (%.1f s of sim time in %.2f s wall, %s) -> %s",
            self._count,
            float(self.rate.fps),
            sim_span,
            wall_span,
            f"{sim_span / wall_span:.2f}x real time" if wall_span > 0 else "instant",
            self.path,
        )
        return self.path


def _model_name(model) -> str:
    """MuJoCo has no model-name accessor, so read it out of the names blob's first entry."""
    try:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0) or ""
    except Exception:  # pragma: no cover - defensive; a name is provenance, not correctness
        return ""


def record_dtype(state_size: int, camera: bool) -> np.dtype:
    """The structured record -- one row per sample: **both clocks**, the state, optionally the camera.

    Exactly the bytes of one ``state`` message, in order: ``t`` is simulated seconds (MuJoCo's
    ``data.time``) and ``w`` is wall seconds elapsed since the recorder started
    (:data:`WALL_CLOCK_ORIGIN`). Both, because neither answers the other's questions: ``t`` is what
    the physics means and the only one a replay can be indexed by, while ``w`` is the only one that
    shows what the run *cost* -- the real-time factor, a step that stalled on a slow sensor, the gap
    where a viewer sat paused. Deriving ``w`` from ``t`` is impossible in either direction, since the
    ratio is exactly the thing that varies.

    Both stay ``f8`` while the state is ``f4``: a float32 second degrades to ~1 ms of resolution within
    a couple of hours of run time, which would quantise away the millisecond differences ``w`` exists to
    show.

    One structured array rather than parallel ``states``/``times`` arrays, so time and state cannot
    desynchronise and the layout is declared once in the provenance.
    """
    fields = [("t", "<f8"), ("w", "<f8"), ("s", "<f4", (state_size,))]
    if camera:
        fields.append(("cam", "<f4", (CAMERA_WIDTH,)))
    return np.dtype(fields)


def _camera_row(cam) -> np.ndarray:
    """A viewer camera as a fixed-width row, in the order :func:`camera_from_row` reads it back."""
    if cam is None:
        return np.zeros(CAMERA_WIDTH, dtype=np.float32)
    return np.array(
        [
            float(cam.type),
            float(cam.fixedcamid),
            float(cam.trackbodyid),
            *[float(v) for v in cam.lookat],
            float(cam.distance),
            float(cam.azimuth),
            float(cam.elevation),
        ],
        dtype=np.float32,
    )


def camera_from_row(row) -> mujoco.MjvCamera:
    """Rebuild an ``MjvCamera`` from a recorded row (the inverse of :func:`_camera_row`)."""
    cam = mujoco.MjvCamera()
    cam.type = int(row[0])
    cam.fixedcamid = int(row[1])
    cam.trackbodyid = int(row[2])
    cam.lookat[:] = [float(row[3]), float(row[4]), float(row[5])]
    cam.distance = float(row[6])
    cam.azimuth = float(row[7])
    cam.elevation = float(row[8])
    return cam


class RecordToggle:
    """F9 in the viewer: start or stop a recording take, mid-session.

    **Travel needs hold state; a toggle needs edges** -- and roqsim has both sources, for opposite reasons.
    :class:`roqsim.viewer.WalkKeys` deliberately avoids the key-event stream, because the passive viewer
    "forwards presses and auto-repeats but no releases, and the repeats are throttled and gappy (~5 Hz
    with second-long holes)", so it reads the live X11 keymap instead. For a toggle that is exactly
    wrong -- a momentary press would be missed between polls -- so this uses the **event stream**, which
    is the one thing that stream is reliable for.

    Two threads, and the split matters. :meth:`key_callback` runs on MuJoCo's UI thread and does nothing
    but debounce and set a flag; :meth:`take_pending` is called by the driver on the physics thread,
    which is where the recorder is actually started or stopped. The recorder does no GL, but keeping the
    state change on the physics thread is what preserves the single-writer rule.
    """

    #: What this handler answers to, and what the window's F1 list says of it. Declared in
    #: :mod:`roqsim.keys`, with every other key roqsim binds and why these are the ones it may take.
    key_bindings = (keys.RECORD_TAKE,)

    #: GLFW keycode for F9, from the binding above -- so the key that records and the key the list
    #: names cannot become two different keys.
    KEY_F9 = keys.KEY_F9

    #: Auto-repeat means a held key delivers several press events (~0.2 s apart), which would toggle
    #: several times. Above that interval, below a deliberate double-press.
    DEBOUNCE_S = keys.DEBOUNCE_S

    def __init__(self, chain=None) -> None:
        self._chain = chain
        self._pending = 0
        self._last_accepted = 0.0

    def key_callback(self, keycode: int) -> None:
        """UI thread. Debounce, count, return -- no sampling, no file I/O, no rendering."""
        if self._chain is not None:
            self._chain(keycode)
        if int(keycode) != self.KEY_F9:
            return
        now = time.monotonic()
        if now - self._last_accepted < self.DEBOUNCE_S:
            return
        self._last_accepted = now
        self._pending += 1

    def take_pending(self) -> bool:
        """Physics thread. Whether a toggle is due, collapsing everything since the last check into one."""
        if not self._pending:
            return False
        self._pending = 0
        return True


class TakeRecorder:
    """Numbered recording takes driven by :class:`RecordToggle`, so F9 can be pressed repeatedly.

    Each stop *finalises* a take, which is a write rather than an encode -- so stopping is cheap and
    needs none of the background machinery a live encoder would have required.

    F9 works in any windowed run, with or without ``--record``: a press with no recorder configured
    starts one at the default path. That is the easy-activation property, and it is free because
    recording is a memcpy.
    """

    def __init__(self, ctx, path: str | Path, rate: CaptureRate, *, logger=None, **kwargs) -> None:
        self._ctx = ctx
        self._base = Path(path)
        self._rate = rate
        self._kwargs = kwargs
        self.log = logger or log
        self._take = 0
        self._active: StateRecorder | None = None
        self.written: list[Path] = []

    @property
    def recording(self) -> bool:
        return self._active is not None

    def _next_path(self) -> Path:
        """``run.mcap``, ``run-2.mcap``, ... so a second take never overwrites the first."""
        self._take += 1
        if self._take == 1:
            return self._base
        return self._base.with_name(f"{self._base.stem}-{self._take}{self._base.suffix}")

    def start(self) -> None:
        if self._active is not None:
            return
        path = self._next_path()
        self._active = StateRecorder(self._ctx, path, self._rate, logger=self.log, **self._kwargs)
        self.log.info("recording: take %d started -> %s (F9 to stop)", self._take, path)

    def stop(self) -> None:
        if self._active is None:
            self.log.debug("recording: F9 with nothing recording")
            return
        active, self._active = self._active, None
        written = active.close()
        if written is not None:
            self.written.append(written)

    def toggle(self) -> None:
        self.stop() if self.recording else self.start()

    def sample(self, ctx, cam=None) -> bool:
        return self._active.sample(ctx, cam=cam) if self._active is not None else False

    def close(self) -> list[Path]:
        """Finalise whatever is running and return every take written. Idempotent."""
        self.stop()
        return list(self.written)
