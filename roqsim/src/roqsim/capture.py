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
overhead (~20 us: the float32 cast, the append, the binding call). So lower the rate because the file is
large, and know that only an every-step rate is measurable at all.

For scale, the thing this design keeps *out* of the loop: one rendered frame is 2-6 ms depending on how
much of the world is in shot (scene-geometry bound, so resolution barely matters). That is three orders
of magnitude more than a sample.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from fractions import Fraction
from importlib import metadata
from pathlib import Path

import mujoco
import numpy as np

from .kinematics import body_twist

log = logging.getLogger(__name__)

#: Frames per *simulated* second when nobody says otherwise. On the grid for every timestep in this
#: repo (0.001/0.002/0.0025/0.004/0.005 -> k = 40/20/16/10/8), so the default path never has to
#: announce a snap.
DEFAULT_FPS = 25

#: Denominator bound for recovering a timestep's intended exact value from its float. A world writes
#: ``timestep: 0.002``, which is not exactly representable; ``Fraction(0.002)`` is a 60-digit monster
#: whereas ``limit_denominator(1e9)`` is exactly ``1/500``. Verified to recover the intent for every
#: timestep in use here, including ``1/240``.
_DT_DENOM_LIMIT = 10**9

#: How far a snap may move the rate before it is worth saying so, and before it is worth a warning.
_QUIET = 0.001  # 0.1%: the caller got what they asked for
_NOTABLE = 0.01  # 1%: above this, name the neighbours


class CaptureError(ValueError):
    """A capture rate that cannot exist in this world (see the message)."""


def parse_fps(text: str | int | float | Fraction) -> Fraction:
    """Parse a rate as an exact :class:`~fractions.Fraction`: ``25``, ``29.412``, ``500/17``, ``1/3``.

    Accepting a fraction literally is what makes every rate this module *prints* re-enterable -- the
    snap messages below suggest rates like ``500/17``, and a suggestion you cannot type back is not a
    suggestion. It also removes the only reason to write a repeating decimal by hand.
    """
    if isinstance(text, Fraction):
        return text
    if isinstance(text, int):
        return Fraction(text)
    if isinstance(text, float):
        return Fraction(text).limit_denominator(_DT_DENOM_LIMIT)
    try:
        return Fraction(str(text).strip())
    except (ValueError, ZeroDivisionError) as err:
        raise CaptureError(
            f"--capture-fps {text!r}: expected a number or a fraction, e.g. 25, 29.412, 500/17, 1/3"
        ) from err


def physics_rate(dt: float) -> Fraction:
    """A world's step rate as an exact rational, recovered from its float timestep."""
    if dt <= 0:
        raise CaptureError(f"timestep must be positive, got {dt!r}")
    return 1 / Fraction(dt).limit_denominator(_DT_DENOM_LIMIT)


@dataclass(frozen=True)
class CaptureRate:
    """A capture rate that exists in this world: ``every`` steps, i.e. exactly ``fps`` per sim second."""

    fps: Fraction  # the effective rate -- what gets declared to ffmpeg and written to a recording
    every: int  # k: sample once per this many physics steps
    requested: Fraction  # what the caller asked for, kept so the report can compare
    physics: Fraction  # the world's step rate, kept for the message

    @property
    def deviation(self) -> float:
        """How far the snap moved the rate, as a fraction of the request."""
        return abs(float(self.fps - self.requested) / float(self.requested))

    @property
    def period(self) -> float:
        """Seconds of *simulated* time between samples."""
        return float(1 / self.fps)

    def ffmpeg_rate(self) -> str:
        """The rate as an exact rational for ffmpeg's ``-r``, e.g. ``500/17``.

        Never a rounded decimal: ``-r 29.41`` on a stream whose real spacing is ``500/17`` drifts, which
        is the whole defect this module exists to avoid.
        """
        return f"{self.fps.numerator}/{self.fps.denominator}"

    def neighbours(self, count: int = 3) -> list[CaptureRate]:
        """Achievable rates either side of this one, nearest first -- the suggestions in the report."""
        out: list[CaptureRate] = []
        for offset in _spiral(count):
            k = self.every + offset
            if k >= 1 and k != self.every:
                out.append(CaptureRate(self.physics / k, k, self.requested, self.physics))
        return out[:count]

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
        if self.deviation < _QUIET:
            logger.debug("capture: %s -> %s", float(self.requested), detail)
        elif self.deviation < _NOTABLE:
            logger.info("capture: --capture-fps %s snapped to %s", float(self.requested), detail)
        else:
            nearby = ", ".join(f"{float(n.fps):g} (k={n.every})" for n in self.neighbours())
            logger.warning(
                "capture: --capture-fps %s snapped to %s -- samples land on physics steps and this "
                "world steps at %s Hz, so %s/%s = %.2f is not reachable. The timebase is exact, so "
                "there is no drift. Nearby: %s.",
                float(self.requested),
                detail,
                float(self.physics),
                float(self.physics),
                float(self.requested),
                float(self.physics / self.requested),
                nearby,
            )


def _spiral(count: int):
    """Step offsets nearest-first: 1, -1, 2, -2, ... so suggestions stay close to what was asked."""
    for i in range(1, count + 2):
        yield i
        yield -i


def snap_fps(fps: str | int | float | Fraction, dt: float) -> CaptureRate:
    """Snap a requested rate onto this world's physics grid. Refuses only the impossible.

    Hard errors are limited to rates that cannot exist at all -- non-positive, or faster than the
    simulation steps -- because everything else has a nearest achievable answer, and an exact rational
    timebase makes taking it harmless.
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
    every = max(1, round(float(rate / requested)))
    return CaptureRate(rate / every, every, requested, rate)


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

#: The recording's own format version, so a future change is refused by name rather than misread.
FORMAT_VERSION = 1

#: Header of the streamed clock record, written beside the recording while the run proceeds.
CLOCK_MAP_FIELDS = ("wall_ts", "sim_ts")

#: Filename of that record, next to the ``.npz`` (``run.npz`` -> ``run.clock_map.csv``).
CLOCK_MAP_SUFFIX = ".clock_map.csv"

#: What ``w`` is measured from. **Elapsed seconds, never a Unix timestamp**: the origin is the moment
#: this recorder was constructed, so the first sample is a few milliseconds rather than 1.7e9. Two
#: reasons the epoch is the wrong choice here. A float64 holding 1.7e9 has ~0.2 us of resolution left,
#: which is coarse against per-step costs measured in microseconds, whereas an elapsed value keeps
#: nanoseconds all run. And ``perf_counter`` is *monotonic*: an NTP step or a DST change mid-run cannot
#: make the column go backwards, which a wall calendar can. A run that needs to be placed on the
#: calendar has the file's mtime and the campaign's own metadata for that.
WALL_CLOCK_ORIGIN = "recorder start (elapsed seconds from time.perf_counter, monotonic)"

#: Columns of the streamed pose record -- RoboVAST's pose-table contract, which is a published
#: schema rather than a shared import (roqsim depends on nothing downstream). Long form: one row per
#: body per sample, so a world that gains a body grows rows and not columns, and the table can be
#: read with the same SQL whatever it contains.
#:
#: Orientation is a quaternion and only a quaternion. Euler angles are lossy the moment a body
#: pitches or rolls -- a drone, a tilting arm, a robot on a ramp -- and yaw is a projection any
#: consumer that wants it can take.
SIM_POSE_FIELDS = (
    "timestamp",
    "wall_time",
    "frame",
    "position.x",
    "position.y",
    "position.z",
    "orientation.x",
    "orientation.y",
    "orientation.z",
    "orientation.w",
    "twist.linear.x",
    "twist.linear.y",
    "twist.linear.z",
    "twist.angular.x",
    "twist.angular.y",
    "twist.angular.z",
)

#: Filename of that record, beside the ``.npz`` rather than named after it. Deliberately NOT
#: ``<recording>.sim_poses.csv`` the way the clock map is named: a consumer that ingests run
#: directories names its table after the file stem, so ``run.sim_poses.csv`` arrives as
#: ``run_sim_poses`` -- the recording's name leaking into an observable's. This file is the
#: observable, so it is named for that and nothing else.
SIM_POSE_FILENAME = "sim_poses.csv"

#: Camera-track width: type, fixedcamid, trackbodyid, lookat(3), distance, azimuth, elevation.
CAMERA_WIDTH = 9

_PROVENANCE_PACKAGES = ("roqsim", "mujoco", "numpy")


class RecordingError(RuntimeError):
    """A recording cannot be written or read (see the message)."""


def env_flag(name: str) -> bool:
    """Read a capture on/off switch from the environment.

    A *session* switch, like every other capture setting: set by whatever launched the run, never by
    the world YAML. Anything but the explicit off-words counts as on, so ``=1``, ``=true`` and a bare
    ``=`` -less presence all work and nobody has to guess the spelling.
    """
    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


def _root_bodies(model) -> list[tuple[int, str]]:
    """Named bodies parented directly to the world -- the free-standing things a trial is about.

    Every robot base, prop and walker is one of these; a wheel, a link or a gripper finger is not.
    That is the line worth drawing by default: the whole body list is mostly a robot's internal
    kinematics, which multiplies the row count by an order of magnitude to record what the joint
    columns already imply.
    """
    out = []
    for bid in range(1, model.nbody):  # 0 is the world body itself
        if int(model.body_parentid[bid]) != 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name:
            out.append((bid, name))
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


class StateRecorder:
    """Sample MuJoCo state into a ``.npz`` while a run proceeds. A **driver** object, not a plugin.

    Capture is a session concern, not an experiment one -- the same footing as ``sim.headless`` (which
    the world YAML explicitly rejects), ``--left-ui`` and ``--manual-control``. So this is constructed by
    a driver and ``sample``\\ d from the loop the driver already runs: no lifecycle hooks, nothing
    injected into a parsed world, and no second route through the world YAML.

    Cost on the run is one ``mj_getState`` (~0.001 ms, about a fiftieth of a physics step) plus a list
    append. Everything expensive -- rebuilding the world, rendering, encoding -- happens afterwards, from
    the file (see :mod:`roqsim.recording`).

    Written once at :meth:`close`. Every stop that must work reaches it through the driver's existing
    ``finally``: closing the viewer window drops ``viewer.is_running()``, and Ctrl+C **or a supervisor's
    SIGTERM** is caught by ``_graceful_stop``, which sets ``QUITTING`` rather than raising. SIGTERM
    matters as much as Ctrl+C here, because it is how a supervised run ends -- a container teardown, a
    ``docker stop``, an eviction, a campaign timeout -- and its default action would kill the process
    with no ``finally`` at all. Only **SIGKILL** still loses the recording, since an ``.npz`` is a zip
    whose index is written at the end; that is the accepted trade for a standard container.
    """

    def __init__(
        self,
        ctx,
        path: str | Path,
        rate: CaptureRate,
        *,
        world: str | None = None,
        overrides: dict | None = None,
        camera: bool = False,
        sim_poses: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.path = Path(path)
        self.rate = rate
        self.log = logger or log
        self._model = ctx.model
        self._size = mujoco.mj_stateSize(ctx.model, STATE_SPEC)
        self._buf = np.empty(self._size)  # mj_getState needs float64; samples are cast on append
        self._states: list[np.ndarray] = []
        self._times: list[float] = []
        self._walls: list[float] = []
        self._cams: list[np.ndarray] = []
        self._camera = camera
        self._next_due = 0.0
        self._closed = False
        # Origin for the wall column, taken before any sample so the series starts at ~0. A *take*
        # started by F9 mid-session gets its own origin, which is what makes each take's real-time
        # factor its own rather than the session's.
        self._origin = time.perf_counter()
        self._wall_start_epoch = time.time()
        # The clock record, streamed and flushed per sample. The .npz is written once, at close(),
        # so a run killed by a timeout leaves none at all -- and that is exactly the run whose log
        # somebody needs to place in time. This file survives, because every line is already on
        # disk when the process dies. Opened lazily on the first sample so a recorder that never
        # samples leaves no empty file for an existence check to trip over.
        self._clock_path = Path(str(self.path).removesuffix(".npz") + CLOCK_MAP_SUFFIX)
        self._clock_file = None
        # The pose record, streamed on the same terms and for the same reasons. Off unless a driver
        # asks for it, because it is a second file per run and only a campaign wants one.
        self._pose_path = (self.path.parent / SIM_POSE_FILENAME) if sim_poses else None
        self._pose_file = None
        self._pose_bodies = _root_bodies(ctx.model) if sim_poses else []
        self._provenance = {
            "format_version": FORMAT_VERSION,
            # The seed belongs in the provenance because it is what makes a *sensor* replay exact: a
            # recomputed lidar scan needs the same noise the live run drew.
            "seed": getattr(ctx, "seed", None),
            "world": world,
            "overrides": overrides or {},
            "packages": package_versions(),
            "state_spec": STATE_SPEC,
            "state_fields": list(STATE_FIELDS),
            "state_size": self._size,
            "dtype": "float32",
            "capture_fps": [rate.fps.numerator, rate.fps.denominator],
            "capture_every_steps": rate.every,
            "timestep": float(ctx.model.opt.timestep),
            "camera_track": bool(camera),
            "wall_clock_origin": WALL_CLOCK_ORIGIN,
            # The calendar instant ``w == 0`` corresponds to. ``w`` itself stays elapsed and
            # monotonic for the reasons WALL_CLOCK_ORIGIN gives -- nanosecond resolution, and
            # immunity to an NTP step mid-run -- but a *reader* outside this process has wall
            # stamps of its own to relate to it: a container log's lines, a rosbag's receive
            # times. One number completes the record; converting the column would break it.
            "wall_start_epoch": self._wall_start_epoch,
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
        return len(self._times)

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
        mujoco.mj_getState(ctx.model, ctx.data, self._buf, STATE_SPEC)
        self._states.append(self._buf.astype(np.float32))
        self._times.append(float(now))
        self._walls.append(wall)
        self._write_clock_sample(wall, float(now))
        self._write_sim_pose_sample(ctx, wall, float(now))
        if self._camera:
            self._cams.append(_camera_row(cam))
        return True

    def _write_clock_sample(self, wall: float, sim: float) -> None:
        """Append one ``(epoch wall, sim)`` pair to the streamed clock record.

        Epoch here, not the elapsed ``w``: this file exists for readers *outside* the process, who
        have calendar stamps of their own to relate to it and no way to learn this run's origin.
        Never fatal -- a recording that cannot write its clock record is still a recording, and
        failing the run over a diagnostic would be the wrong trade.
        """
        try:
            if self._clock_file is None:
                self._clock_path.parent.mkdir(parents=True, exist_ok=True)
                self._clock_file = open(  # pylint: disable=consider-using-with
                    self._clock_path, "w", encoding="utf-8", buffering=1
                )
                self._clock_file.write(",".join(CLOCK_MAP_FIELDS) + "\n")
            self._clock_file.write(f"{self._wall_start_epoch + wall:.6f},{sim:.6f}\n")
            self._clock_file.flush()
        except OSError as err:
            self.log.debug("recording: clock record not written (%s)", err)
            self._clock_file = None

    def _write_sim_pose_sample(self, ctx, wall: float, sim: float) -> None:
        """Append this sample's world pose and twist, one row per root body.

        Streamed and flushed per row for the reason :meth:`_write_clock_sample` gives, and one more:
        this file is a *run's ground truth*, so it is exactly what somebody wants from the run that
        died. It is also the only pose data a **stepped** run produces at all -- with no ROS there is
        no rosbag and so no TF to derive poses from afterwards.

        The velocities come from the solver rather than from differencing the positions, which is the
        point of the file: a difference is only ever as good as the interval it is divided by, and an
        arrival-time interval is not the interval the motion happened over.

        One convention, stated because it is easy to get wrong and impossible to see: ``mj_step``
        integrates ``qpos`` and then leaves ``xpos`` holding the pose from *before* that integration,
        so the row is a coherent snapshot of ``sim - dt`` carrying the label ``sim``. That is
        deliberately the same one-step lag the ``ground_truth_pose`` plugin publishes with, so this
        table and the TF one describe the same instant and any difference between them is transport
        rather than convention. It cancels in every derivative.
        """
        if self._pose_path is None:
            return
        try:
            if self._pose_file is None:
                self._pose_path.parent.mkdir(parents=True, exist_ok=True)
                self._pose_file = open(  # pylint: disable=consider-using-with
                    self._pose_path, "w", encoding="utf-8", buffering=1
                )
                self._pose_file.write(",".join(SIM_POSE_FIELDS) + "\n")
            data = ctx.data
            for bid, name in self._pose_bodies:
                pos, quat = data.xpos[bid], data.xquat[bid]
                twist = body_twist(ctx.model, data, bid)
                self._pose_file.write(
                    f"{sim:.6f},{self._wall_start_epoch + wall:.6f},{name},"
                    # MuJoCo orders a quaternion (w, x, y, z); the contract orders it (x, y, z, w),
                    # as ROS does. Reordered here rather than at the reader, once.
                    f"{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f},"
                    f"{quat[1]:.6f},{quat[2]:.6f},{quat[3]:.6f},{quat[0]:.6f},"
                    f"{twist.linear[0]:.6f},{twist.linear[1]:.6f},{twist.linear[2]:.6f},"
                    f"{twist.angular[0]:.6f},{twist.angular[1]:.6f},{twist.angular[2]:.6f}\n"
                )
            self._pose_file.flush()
        except OSError as err:
            self.log.debug("recording: pose record not written (%s)", err)
            self._pose_file = None

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
        derive something from the run (a browser run capture, say) without paying seconds for a
        rebuild it does not need. The state layout stays known in exactly one module either way.

        **Destructive and terminal**: it overwrites ``ctx.data`` sample by sample, so it belongs after
        the loop is done. One ``MjData`` is re-posed throughout, so a consumer that keeps values must
        copy them.
        """
        buf = np.empty(self._size)
        for t, state in zip(self._times, self._states, strict=True):
            buf[:] = state
            mujoco.mj_setState(ctx.model, ctx.data, buf, STATE_SPEC)
            # xpos/xquat/site_xpos are derived, never stored -- this is what makes them available.
            mujoco.mj_forward(ctx.model, ctx.data)
            yield t, ctx.data

    def close(self) -> Path | None:
        """Write the recording and return its path, or ``None`` when there was nothing to write.

        Idempotent, because every call site is a ``finally`` and a driver may unwind more than once.
        """
        if self._closed:
            return None
        self._closed = True
        for attr in ("_clock_file", "_pose_file"):
            handle = getattr(self, attr)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
                setattr(self, attr, None)
        if not self._times:
            # Nothing sampled: say so rather than leaving an empty file that passes an existence check.
            self.log.warning(
                "recording: no samples taken, so %s was not written. The run ended before the first "
                "sample was due at %s fps (every %.3f s of sim time).",
                self.path,
                float(self.rate.fps),
                self.rate.period,
            )
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        samples = _pack(
            self._times, self._walls, self._states, self._cams if self._camera else None
        )
        # Plain savez, never savez_compressed: float mantissas of smooth signals are incompressible
        # noise in the low bits (measured: 4% on float64), and float32 has already removed those bits.
        # It would cost real CPU inside the run for nothing.
        np.savez(
            self.path,
            meta=np.array(json.dumps(self._provenance)),
            samples=samples,
        )
        sim_span = self._times[-1] - self._times[0]
        wall_span = self._walls[-1] - self._walls[0]
        self.log.info(
            # Wall gets two decimals where sim gets one: a fast `--pacing asap` run finishes in
            # hundredths of a second, and "in 0.0 s wall" would report a real measurement as nothing.
            "recording: %d samples at %s fps (%.1f s of sim time in %.2f s wall, %s) -> %s",
            len(self._times),
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

    ``t`` is simulated seconds (MuJoCo's ``data.time``) and ``w`` is wall seconds elapsed since the
    recorder started (:data:`WALL_CLOCK_ORIGIN`). Both, because neither answers the other's questions:
    ``t`` is what the physics means and the only one a replay can be indexed by, while ``w`` is the only
    one that shows what the run *cost* -- the real-time factor, a step that stalled on a slow sensor,
    the gap where a viewer sat paused. Deriving ``w`` from ``t`` is impossible in either direction,
    since the ratio is exactly the thing that varies.

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


def _pack(times, walls, states, cams) -> np.ndarray:
    out = np.empty(len(times), dtype=record_dtype(len(states[0]), cams is not None))
    out["t"] = times
    out["w"] = walls
    out["s"] = np.asarray(states, dtype=np.float32)
    if cams is not None:
        out["cam"] = np.asarray(cams, dtype=np.float32)
    return out


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

    #: GLFW keycode for F9. F1-F7 are Simulate's own (help/info/profiler/sensor/fullscreen/frame/label);
    #: F8 and F9 are unbound. The arrows, Page Up/Down and Shift belong to :class:`~roqsim.viewer.WalkKeys`,
    #: and letters are unsafe because Simulate claims W/A/S/D/E/Q for visualization flags.
    KEY_F9 = 298

    #: Auto-repeat means a held key delivers several press events (~0.2 s apart), which would toggle
    #: several times. Above that interval, below a deliberate double-press.
    DEBOUNCE_S = 0.4

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
        """``run.npz``, ``run-2.npz``, ... so a second take never overwrites the first."""
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
