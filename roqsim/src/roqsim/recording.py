"""Read a recording back: rebuild its world, verify it, and hand out restored states by time.

A **public API**, not an internal helper. Two commands are thin layers on it -- ``roqsim render --state``
turns a restored state into pixels and ``roqsim state`` turns one into numbers -- and it is also the answer
to "how do I compute something you did not think of":

    from roqsim.recording import open_recording

    rec = open_recording("run.npz")
    for sample in rec.range(8.0, 20.0):
        sample.sim_time, sample.wall_time, sample.index, sample.data
        ...                       # any numpy/mujoco computation over a real restored state

Everything subtle lives here exactly once, so the two commands cannot drift on it: the world rebuild,
the provenance check, nearest-sample selection by time, and reporting *which* sample a request actually
landed on.

The file itself is a plain ``.npz`` with a ``meta`` member (JSON provenance) and a ``samples`` member
(one structured record per sample: ``t`` sim seconds, ``w`` elapsed wall seconds, ``s`` state, and
``cam`` when the camera was tracked), so a reader with numpy and no ``roqsim`` at all can open it:
``np.load("run.npz")`` returns both.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import mujoco
import numpy as np

from .capture import STATE_SPEC, RecordingError, record_dtype

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    """One restored moment: its two clocks, its index in the recording, and MuJoCo data posed at it.

    ``sim_time`` is what everything selects by; ``wall_time`` is elapsed real seconds from the start of
    the recording (see :data:`roqsim.capture.WALL_CLOCK_ORIGIN`) and carries no meaning for the physics --
    it is there to say what the run cost.
    """

    sim_time: float
    index: int
    data: mujoco.MjData
    camera: mujoco.MjvCamera | None = None
    wall_time: float = 0.0


class Recording:
    """An opened recording, with its world rebuilt and checked against what produced it."""

    def __init__(self, path: Path, meta: dict, samples: np.ndarray) -> None:
        self.path = path
        self.meta = meta
        self._samples = samples
        self._model: mujoco.MjModel | None = None
        self._ctx = None
        self._data: mujoco.MjData | None = None
        self._buf: np.ndarray | None = None
        self._view: dict | None = None
        self._run_sensors = False

    # -- what the file says about itself ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def times(self) -> np.ndarray:
        """Simulated seconds, one per sample -- the axis every selector works in."""
        return self._samples["t"]

    @property
    def wall_times(self) -> np.ndarray:
        """Elapsed real seconds, one per sample, from the start of the recording (never a timestamp)."""
        return self._samples["w"]

    @property
    def span(self) -> tuple[float, float]:
        return float(self.times[0]), float(self.times[-1])

    @property
    def wall_span(self) -> tuple[float, float]:
        return float(self.wall_times[0]), float(self.wall_times[-1])

    @property
    def real_time_factor(self) -> float | None:
        """Simulated seconds per real second over the recording, or ``None`` for a single sample.

        Above 1 the run outpaced real time (``--pacing asap`` on a cheap world); below it the world was
        too expensive to keep up. ``None`` rather than a divide-by-zero when there is no span to divide.
        """
        w0, w1 = self.wall_span
        t0, t1 = self.span
        return (t1 - t0) / (w1 - w0) if w1 > w0 else None

    @property
    def fps(self) -> Fraction:
        num, den = self.meta["capture_fps"]
        return Fraction(int(num), int(den))

    @property
    def has_camera(self) -> bool:
        return bool(self.meta.get("camera_track")) and "cam" in (self._samples.dtype.names or ())

    @property
    def world(self) -> str | None:
        return self.meta.get("world")

    @property
    def view(self) -> dict | None:
        """The rebuilt world's ``sim.view``, so replaying a world looks like rendering it.

        Available only after :meth:`build`. Without this a replay auto-frames instead of honouring the
        camera the world states, and the same world would look different depending on whether you
        rendered it live or from a recording.
        """
        return self._view

    def describe(self) -> dict:
        """A JSON-safe summary -- what ``--check`` reports, and what an MCP caller can read cheaply."""
        t0, t1 = self.span
        w0, w1 = self.wall_span
        return {
            "path": str(self.path),
            "samples": len(self),
            "span": [t0, t1],
            "wall_span": [w0, w1],
            "real_time_factor": self.real_time_factor,
            "fps": float(self.fps),
            "world": self.world,
            "camera_track": self.has_camera,
            "model": self.meta.get("model", {}),
            "packages": self.meta.get("packages", {}),
        }

    # -- the world -------------------------------------------------------------------------------

    def build(self, target: str | None = None, *, no_ceiling: bool = False):
        """Rebuild the recorded world (or ``target`` if given) and check it can hold these states.

        The world target is **optional**: the provenance names it, so a caller need not repeat what the
        file already knows. Passing one explicitly wins -- that is how you render a recording whose world
        was renamed or relocated -- but it is still size-checked, so an override cannot silently become a
        way to render the *wrong* world.
        """
        if self._model is not None:
            return self._model, self._ctx

        ref = target or self.world
        if not ref:
            raise RecordingError(
                f"{self.path} does not name the world it was recorded from, so it cannot be rebuilt. "
                "Pass the world explicitly."
            )
        from .render import build_target

        try:
            model, data, ctx, view, _cam = build_target(
                ref, self.meta.get("overrides") or None, no_ceiling=no_ceiling
            )
        except Exception as err:
            raise RecordingError(self._rebuild_help(ref, err)) from err

        self._check_size(model, ref)
        # The recorded seed must reach the rebuilt world, or a re-run sensor draws different noise than
        # the live run published -- which is the entire point of recording the seed. Storing it and then
        # not applying it made every replayed scan differ while looking perfectly plausible.
        if ctx is not None and self.meta.get("seed") is not None:
            ctx.seed = int(self.meta["seed"])
        self._model, self._ctx, self._data, self._view = model, ctx, data, view
        self._buf = np.empty(int(self.meta["state_size"]))
        return model, ctx

    def _rebuild_help(self, ref: str, err: Exception) -> str:
        """Distinguish "the world moved" from "the wrong packages are installed"."""
        recorded = self.meta.get("model", {})
        versions = ", ".join(f"{k} {v}" for k, v in (self.meta.get("packages") or {}).items())
        return (
            f"could not rebuild the world {ref!r} this recording came from: {err}\n"
            f"It was recorded against {recorded.get('nbody', '?')} bodies / "
            f"{recorded.get('ngeom', '?')} geoms with {versions or 'unknown versions'}. Either the "
            f"world moved (pass its current name explicitly) or the installed packages differ from the "
            f"ones that recorded it."
        )

    def _check_size(self, model, ref: str) -> None:
        """Refuse a recording this world cannot hold, before anything is rendered from it.

        Without this a different world produces plausible garbage, which is worse than an error.

        **The spec is checked before the size**, because a spec mismatch fully explains a size mismatch
        and the size message would otherwise accuse the world falsely -- it prints the recorded and
        rebuilt dimensions, which for an old ``FULLPHYSICS`` recording are *identical*, so the reader is
        told the world does not match when the world is fine and the format is not.
        """
        spec = int(self.meta.get("state_spec", 0))
        if spec != STATE_SPEC:
            extra = ""
            if spec == int(mujoco.mjtState.mjSTATE_FULLPHYSICS):
                extra = (
                    " That is mjSTATE_FULLPHYSICS, which omits ctrl and the mocap fields, so replaying "
                    "it would show every walker and moving prop frozen at its compile-time pose and "
                    "every door driven toward 0 while the robot moved correctly."
                )
            raise RecordingError(
                f"{self.path} was recorded with state spec {spec}, but this roqsim reads "
                f"{STATE_SPEC}.{extra} Re-record it."
            )
        expected = int(self.meta["state_size"])
        actual = mujoco.mj_stateSize(model, spec)
        if actual != expected:
            m = self.meta.get("model", {})
            raise RecordingError(
                f"{ref} does not match this recording: its state is {actual} values, the recording "
                f"holds {expected}. Recorded against nq={m.get('nq')} nv={m.get('nv')} "
                f"nu={m.get('nu')} nmocap={m.get('nmocap')}; the rebuilt world has nq={model.nq} "
                f"nv={model.nv} nu={model.nu} nmocap={model.nmocap}."
            )

    # -- selecting moments -----------------------------------------------------------------------

    def _restore(self, index: int) -> Sample:
        model, ctx = self.build()
        row = self._samples[index]
        self._buf[:] = row["s"]  # float32 on disk -> float64 for mj_setState
        mujoco.mj_setState(
            model, self._data, self._buf, int(self.meta.get("state_spec", STATE_SPEC))
        )
        mujoco.mj_forward(model, self._data)
        if self._run_sensors and ctx is not None:
            self._drive_sensors(ctx)
        cam = None
        if self.has_camera:
            from .capture import camera_from_row

            cam = camera_from_row(row["cam"])
        return Sample(float(row["t"]), int(index), self._data, cam, float(row["w"]))

    def index_at(self, when: float) -> int:
        """The index of the sample **nearest** ``when``; ties resolve to the earlier one.

        Nearest rather than preceding, and never interpolated: blending two states would produce a pose
        the simulation never had. Out of range is an error rather than a clamp -- silently returning the
        first or last sample is what makes a wrong answer look like a right one.
        """
        t0, t1 = self.span
        period = float(1 / self.fps)
        if when < t0 - period or when > t1 + period:
            raise RecordingError(
                f"--at {when:g} is outside this recording: it spans {t0:.3f}..{t1:.3f} s of sim time "
                f"at {float(self.fps):g} fps ({len(self)} samples)."
            )
        # searchsorted + a single comparison, so a tie lands on the earlier sample.
        pos = int(np.searchsorted(self.times, when, side="left"))
        if pos == 0:
            return 0
        if pos >= len(self):
            return len(self) - 1
        before, after = self.times[pos - 1], self.times[pos]
        return pos - 1 if (when - before) <= (after - when) else pos

    def enable_sensors(self) -> None:
        """Run the world's observation plugins on each restored sample, so endpoints carry values.

        Off by default: rendering needs only the state, and running plugins it does not read would be
        pure cost. ``roqsim state --sensor`` turns it on.
        """
        self._run_sensors = True

    def _drive_sensors(self, ctx) -> None:
        """``post_step`` every plugin that has one, against the state just restored.

        This is the whole sensor-replay mechanism: a sensor registers its ports in ``configure`` and fills
        them in ``post_step``, both of which work against a compiled model with no engine loop -- so
        re-running one needs no lifecycle fiction. Plugins whose ``post_step`` depends on state MuJoCo does
        not carry (a controller's integrator, a walker's ORCA goals) will produce nonsense, which is why
        :func:`roqsim.state.replayable_sensors` only offers *outward* endpoints and why a stateful instance
        is refused by name rather than silently recomputed.
        """
        engine = getattr(ctx, "engine", None)
        if engine is None:
            return
        # ctx.data is the engine's own buffer, and _restore posed exactly that object.
        for plugin in engine.plugins:
            try:
                plugin.post_step(ctx)
            except Exception as err:  # noqa: BLE001 - one broken sensor must not stop the rest
                log.debug("replay: %s.post_step failed: %s", type(plugin).__name__, err)

    def at(self, when: float | None = None) -> Sample:
        """The sample nearest ``when``, or the last one when ``when`` is ``None``.

        The last sample is the useful default for "what did this run end up looking like".
        """
        return self._restore(len(self) - 1 if when is None else self.index_at(when))

    def range(self, start: float | None = None, stop: float | None = None):
        """Yield every sample in ``[start, stop]`` in order, as :class:`Sample`.

        The same ``MjData`` is re-posed for each, so a consumer that keeps values must copy them -- which
        is what makes iterating a long recording allocation-free.
        """
        lo = 0 if start is None else self.index_at(start)
        hi = len(self) - 1 if stop is None else self.index_at(stop)
        if hi < lo:
            lo, hi = hi, lo
        for index in range(lo, hi + 1):
            yield self._restore(index)

    def at_record(self, when: float | None, sample: Sample) -> dict:
        """The fields every caller reports about a time request, so ``--at`` is auditable.

        A caller must be able to see that it landed 12 ms early rather than assume it got exactly what it
        asked for: at 25 fps a request can be 20 ms off, which is enough to matter for "was the gripper
        closed yet?". ``at_error`` is in sim time, because that is what was requested; ``wall_time``
        rides along to say when in the run's real elapsed time that moment happened.
        """
        return {
            "sim_time": round(sample.sim_time, 6),
            "wall_time": round(sample.wall_time, 6),
            "sample_index": sample.index,
            "requested_at": None if when is None else round(float(when), 6),
            "at_error": None if when is None else round(sample.sim_time - float(when), 6),
        }


def open_recording(path: str | Path) -> Recording:
    """Open a ``.npz`` recording, validating its shape before anything expensive happens."""
    path = Path(path)
    if not path.exists():
        raise RecordingError(f"{path}: no such recording")
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as err:
        raise RecordingError(
            f"{path} is not a readable recording ({type(err).__name__}: {err}). A run killed with "
            "SIGKILL leaves an unreadable archive, because an .npz writes its index at the end."
        ) from err

    missing = {"meta", "samples"} - set(archive.files)
    if missing:
        raise RecordingError(
            f"{path} is missing {', '.join(sorted(missing))}; it has {sorted(archive.files)}. "
            "A recording holds a JSON 'meta' member and a structured 'samples' member."
        )
    meta = json.loads(str(archive["meta"]))
    samples = archive["samples"]
    if len(samples) == 0:
        raise RecordingError(f"{path} holds no samples")

    expected = record_dtype(int(meta["state_size"]), bool(meta.get("camera_track")))
    if samples.dtype != expected:
        raise RecordingError(
            f"{path}: samples are {samples.dtype}, but its provenance describes {expected}. "
            "The file and its declared layout disagree."
        )
    return Recording(path, meta, samples)
