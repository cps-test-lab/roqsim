"""Export a recorded run to a browser run capture: capture.json + capture.bin.

The motion half of "replay a run in the web UI". :mod:`roqsim.export_web` emits the *geometry* --
a body tree with named joints -- and this emits the *motion* that animates it: one track per joint
value and one per body pose that actually moved, keyed by the **name** the geometry uses.

The format is defined by the consumer that reads it -- RoboVAST, in *its own* ``docs/run_capture.rst``,
not a file in this tree -- and roqsim is one producer of it, the same relationship this package already
has with URDF and SRDF. That is why the
tracks are named values rather than a MuJoCo state vector: ``qpos`` is meaningless without this
model, so resolving names is the producer's job and no viewer ever learns what a ``qposadr`` is.

Two callers, one derivation, so they cannot drift:

* ``roqsim export capture --state run.npz`` -- after the fact, through :mod:`roqsim.recording` (one world
  rebuild, then ``mj_setState`` + ``mj_forward`` per sample);
* the scenario adapter at shutdown -- no rebuild needed, because it still holds the live model.

Both land in :func:`write_capture`, which takes a model plus an iterator of posed ``MjData``.

What travels, and what does not:

* **Joint tracks** -- one scalar per hinge/slide joint, in the joint's own unit (rad or m), read from
  ``data.qpos[jnt_qposadr]``. These are what let an *articulated* robot replay: a viewer composes them
  through the geometry's own FK metadata, which a pose-per-body stream cannot do (a child link's world
  pose says nothing a viewer can apply without its parent's).
* **Pose tracks** -- world-frame ``(pos, wxyz quat)`` for every body whose ``xpos``/``xquat`` actually
  varies and that no joint track already explains: free-jointed bodies, mocap bodies (a walker's
  bones), ball-jointed ones. Selected by *observation* rather than configuration, so a world that
  gains a walker or a movable prop captures it with no config change.
* Static bodies are omitted -- the geometry already carries their rest pose, and a track per wall
  would dwarf the file.

Poses are the simulator's **world** frame, matching the exported geometry 1:1, and the manifest says
so (``frame``). That is not a detail: a nav stack's ``base_link`` lives in the *map* frame, which can
be metres away from the world origin, and a viewer cannot tell the two apart from the numbers.
"""

from __future__ import annotations

import json
import logging
from importlib import metadata
from pathlib import Path

import mujoco
import numpy as np

log = logging.getLogger(__name__)

#: The format this writer implements. A consumer refuses an unknown pair rather than misreading it.
FORMAT = "robovast.run_capture"
FORMAT_VERSION = 1

#: Bodies whose world pose stays within this of its first sample are treated as static and get no
#: track. Well below anything visible at any sane zoom, and above float32 round-trip noise (a
#: recording stores state as float32, so a body at rest still jitters in the last mantissa bits).
_STATIC_EPS = 1e-6

#: MuJoCo joint types that reduce to one scalar a viewer can apply through the geometry's FK.
_SCALAR_JOINTS = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))

#: Per-joint-type unit, for the manifest. MuJoCo hinges are radians and slides are metres.
_JOINT_UNIT = {
    int(mujoco.mjtJoint.mjJNT_HINGE): "rad",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "m",
}


class CaptureExportError(RuntimeError):
    """A run capture cannot be written (see the message)."""


class _BinWriter:
    """Accumulate typed tracks into one ``capture.bin``, each aligned for its own dtype.

    Alignment is per dtype rather than a blanket 4 bytes: a browser's ``new Float64Array(buf, off, n)``
    *throws* unless ``off`` is a multiple of 8, so a float64 time track appended after float32 value
    tracks would be unreadable at an unlucky offset. Getting this wrong fails only for some
    recordings, which is worse than failing for all of them.

    Tracks are **sample-major** (all of sample 0's values, then sample 1's, ...). That is what makes a
    *time window* a contiguous byte range, so a consumer can range-read the part it is showing
    instead of the whole file -- the property the format promises and a live source needs.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def add(self, arr: np.ndarray, dtype: str) -> int:
        """Append ``arr`` as ``dtype``, returning its byte offset."""
        flat = np.ascontiguousarray(arr, dtype=dtype).ravel()
        align = np.dtype(dtype).itemsize
        while len(self._buf) % align != 0:
            self._buf.append(0)
        off = len(self._buf)
        self._buf.extend(flat.tobytes())
        return off

    def bytes(self) -> bytes:
        return bytes(self._buf)


def _producer_version() -> str:
    try:
        return metadata.version("roqsim")
    except metadata.PackageNotFoundError:  # pragma: no cover - a source checkout
        return "unknown"


def _scalar_joints(model: mujoco.MjModel) -> list[tuple[str, int, int]]:
    """Every hinge/slide joint as ``(name, qposadr, jnt_type)``, in model order.

    An unnamed joint is skipped: the whole addressing scheme is by name, so a track a consumer cannot
    key onto its joint is dead weight in the file.
    """
    out = []
    for i in range(model.njnt):
        jtype = int(model.jnt_type[i])
        if jtype not in _SCALAR_JOINTS:
            continue
        name = model.joint(i).name
        if not name:
            continue
        out.append((name, int(model.jnt_qposadr[i]), jtype))
    return out


def _body_joints(model: mujoco.MjModel) -> list[list[int]]:
    """Joint ids per body, in model order."""
    per: list[list[int]] = [[] for _ in range(model.nbody)]
    for j in range(model.njnt):
        per[int(model.jnt_bodyid[j])].append(j)
    return per


def _pose_bodies(
    model: mujoco.MjModel, xpos: np.ndarray, xquat: np.ndarray
) -> tuple[list[tuple[str, int]], list[str]]:
    """Bodies needing a world-pose track, as ``(name, body_id)``, plus the names of any holes.

    A body needs one only when a viewer could not otherwise place it, and "otherwise" is *transitive*:
    a link welded to a moving parent, or hanging off it through hinge joints, is already determined by
    that parent's transform composed with the joint tracks. Testing only whether a body owns a scalar
    joint gets this wrong in the common direction -- it emitted a redundant pose track for every
    gripper pad and sensor mount on a moving arm (36 tracks instead of 2 on the TIAGo pick), which is
    not merely wasteful: two sources writing one body's transform is last-one-wins.

    So this is a single forward pass over model order -- which MuJoCo guarantees puts a parent before
    its children -- carrying "can a viewer reconstruct this body yet?":

    * a body is reconstructable when its parent is *and* its own joints are all named hinge/slide
      *and* it is not mocap (a mocap body's placeholder rest pose is a lie by design);
    * one that is not gets a pose track -- which makes it reconstructable, and so its descendants too;
    * one that is not, but never moved, needs nothing: the geometry's rest pose is already right.

    That ordering is part of the format's contract, not an accident: a consumer turning a world pose
    into a local transform needs its parent already seated.
    """
    moved = (np.abs(xpos - xpos[0]).max(axis=(0, 2)) > _STATIC_EPS) | (
        np.abs(xquat - xquat[0]).max(axis=(0, 2)) > _STATIC_EPS
    )
    body_joints = _body_joints(model)
    reconstructable = [False] * model.nbody
    reconstructable[0] = True  # the world body

    out: list[tuple[str, int]] = []
    holes: list[str] = []
    for i in range(1, model.nbody):
        scalar_only = all(
            int(model.jnt_type[j]) in _SCALAR_JOINTS and model.joint(j).name for j in body_joints[i]
        )
        is_mocap = int(model.body_mocapid[i]) >= 0
        if reconstructable[int(model.body_parentid[i])] and scalar_only and not is_mocap:
            reconstructable[i] = True
            continue
        if not moved[i]:
            # Static and not derivable: the geometry's rest transform already places it.
            reconstructable[i] = True
            continue
        name = model.body(i).name
        if not name:
            # Addressing is by name throughout, so an unnamed mover cannot be expressed. Say which,
            # rather than shipping a capture that silently leaves it at its rest pose.
            holes.append(f"body {i} (unnamed)")
            continue
        out.append((name, i))
        reconstructable[i] = True
    return out, holes


def write_capture(
    model: mujoco.MjModel,
    samples,
    out_dir: str | Path,
    *,
    world: str | None = None,
    overrides: dict | None = None,
    packages: dict | None = None,
    seed: int | None = None,
    time_base: str = "sim",
    logger: logging.Logger | None = None,
) -> dict:
    """Write ``capture.json`` + ``capture.bin`` for ``samples`` into ``out_dir``.

    Args:
        model: the compiled model the samples belong to -- the source of every track *name*.
        samples: iterable of ``(sim_time, MjData)`` already posed and forwarded, in time order.
            ``mj_forward`` must have run: ``xpos``/``xquat`` are what pose tracks read.
        out_dir: directory to write into (created if absent).
        world: the world reference this run was built from.
        overrides: the ``world_overrides`` it was built with. Together with *world* this is the
            **world identity** -- what a consumer needs to obtain matching geometry, not merely
            provenance to display. Pass the real value even when empty (``{}`` means "none were
            applied"); ``None`` means "not recorded", which a consumer must not read as "none".
        packages: producer package versions, so a version mismatch is legible rather than mysterious.
            Defaults to *this* process's versions, which is right for a live export; a replay from a
            recording passes the ones that *recorded* it, which is the more truthful answer there.
        seed: the run's seed, for the manifest's provenance.
        time_base: ``"sim"`` (simulated seconds from the run's start) or ``"wall"``.

    Returns:
        The manifest that was written.
    """
    lg = logger or log
    out = Path(out_dir)
    if packages is None:
        from .capture import package_versions

        packages = package_versions()

    times: list[float] = []
    qpos: list[np.ndarray] = []
    xpos: list[np.ndarray] = []
    xquat: list[np.ndarray] = []
    for t, data in samples:
        times.append(float(t))
        qpos.append(np.array(data.qpos, dtype=np.float64))
        xpos.append(np.array(data.xpos, dtype=np.float64))
        xquat.append(np.array(data.xquat, dtype=np.float64))

    if not times:
        raise CaptureExportError(
            "nothing to export: the recording yielded no samples. A run killed hard (SIGKILL, a "
            "campaign timeout) writes no recording at all -- check that the run stopped cleanly."
        )

    t_arr = np.asarray(times, dtype=np.float64)
    qpos_arr = np.stack(qpos)  # (n, nq)
    xpos_arr = np.stack(xpos)  # (n, nbody, 3)
    xquat_arr = np.stack(xquat)  # (n, nbody, 4)

    joints = _scalar_joints(model)
    bodies, holes = _pose_bodies(model, xpos_arr, xquat_arr)
    if holes:
        lg.warning(
            "run capture cannot express %d moving body/bodies: %s. They will replay at their rest "
            "pose. Name them in the model to include them.",
            len(holes),
            ", ".join(holes),
        )

    binw = _BinWriter()
    n = len(times)
    time_track = {"off": binw.add(t_arr, "<f8"), "dtype": "f8", "samples": n, "width": 1}

    tracks: list[dict] = []
    for name, adr, jtype in joints:
        tracks.append(
            {
                "kind": "joint",
                "name": name,
                "unit": _JOINT_UNIT[jtype],
                "width": 1,
                "samples": n,
                "dtype": "f4",
                "off": binw.add(qpos_arr[:, adr], "<f4"),
            }
        )
    for name, bid in bodies:
        # (n, 7) sample-major: pos xyz then quat wxyz, which is MuJoCo's own quaternion order and
        # what the geometry's rest transforms already use.
        pose = np.concatenate([xpos_arr[:, bid, :], xquat_arr[:, bid, :]], axis=1)
        tracks.append(
            {
                "kind": "pose",
                "name": name,
                "width": 7,
                "samples": n,
                "dtype": "f4",
                "off": binw.add(pose, "<f4"),
            }
        )

    manifest = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        # A file is finished by construction. The flag exists so the same manifest shape can describe
        # a live source, whose upper bound moves and whose track list may still grow.
        "complete": True,
        "frame": "world",
        "time": {"base": time_base, "t0": float(t_arr[0]), "t1": float(t_arr[-1]), **time_track},
        "producer": "roqsim",
        "producer_version": _producer_version(),
        "world": world,
        # Distinguish {} ("no overrides applied") from absent ("this producer did not record them").
        # A consumer that conflated the two would compile the *unoverridden* world for a run that
        # varied it, and render confidently wrong geometry -- the failure this format exists to avoid.
        **({} if overrides is None else {"overrides": overrides}),
        **({} if packages is None else {"packages": packages}),
        "seed": None if seed is None else int(seed),
        "tracks": tracks,
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "capture.bin").write_bytes(binw.bytes())
    (out / "capture.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    lg.info(
        "wrote run capture to %s: %d samples, %d joint + %d pose tracks, %.1f KiB",
        out,
        n,
        len(joints),
        len(bodies),
        (out / "capture.bin").stat().st_size / 1024,
    )
    return manifest


def export_from_recording(
    state: str | Path,
    out_dir: str | Path,
    *,
    target: str | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    """Export a capture from a recorded ``.npz``, rebuilding its world once.

    The recording names the world it came from, so ``target`` is only for one that moved or was
    renamed -- and it is still size-checked, so it cannot become a way to replay the wrong world.
    """
    from .recording import open_recording

    rec = open_recording(state)
    model, _ctx = rec.build(target)

    def posed():
        # `range` re-poses one MjData per step, so nothing here may keep `data` past the yield --
        # write_capture copies what it needs out of it before asking for the next sample.
        for sample in rec.range(*rec.span):
            yield sample.sim_time, sample.data

    return write_capture(
        model,
        posed(),
        out_dir,
        world=rec.world,
        overrides=rec.meta.get("overrides") or {},
        packages=rec.meta.get("packages"),
        seed=rec.meta.get("seed"),
        logger=logger,
    )


def main(argv: list[str] | None = None) -> int:
    """``roqsim export capture`` -- write a run capture from a recording."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="roqsim export capture",
        description="Export a recorded run to a browser run capture (capture.json + capture.bin).",
    )
    ap.add_argument("--state", required=True, help="the recording (.npz) to export")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument(
        "--world",
        default=None,
        help="world reference, when the recording's own has moved or been renamed",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        manifest = export_from_recording(args.state, args.out, target=args.world)
    except (CaptureExportError, OSError) as err:
        print(f"error: {err}", file=__import__("sys").stderr)
        return 1
    # One line of JSON on stdout: a machine contract, as `roqsim render` has.
    print(
        json.dumps(
            {
                "out": str(Path(args.out)),
                "samples": manifest["time"]["samples"],
                "tracks": len(manifest["tracks"]),
                "span": [manifest["time"]["t0"], manifest["time"]["t1"]],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
