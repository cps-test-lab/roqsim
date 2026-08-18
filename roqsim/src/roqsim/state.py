"""Get numbers out of a recording: poses, joints, contacts, MJCF sensors, and a world's own sensors.

    roqsim state --state run.npz --at 12.5 --body base_link          # one moment  -> JSON
    roqsim state --state run.npz --body base_link --out base.csv     # whole run   -> CSV
    roqsim state --state run.npz --from 8 --to 20 --body 'tiago_*' --site grasp --out arm.csv
    roqsim state --state run.npz --sensor front --out scan.npz       # a world's lidar, re-run
    roqsim state --state run.npz --at 12.5 --contacts

The counterpart to ``roqsim render``: same recording, same :mod:`roqsim.recording` core -- so ``--at``,
``--from``/``--to``, the nearest-sample snapping and the provenance refusal are inherited, not
reimplemented -- but the output is numbers rather than pixels.

**Why not merged into ``roqsim render``.** They share all their machinery and almost none of their flags:
``--size``/``--view``/``--focus``/``--fps`` are meaningless for a CSV, and the selectors here are
meaningless for a PNG. One command would mean two mutually irrelevant flag clusters plus cross-validation
rules for nonsensical combinations, and someone hunting for pose data would not look under "render".

Output is JSON for one moment and CSV for a range, following this repo's existing convention and the
reason ``insertion_task`` states for it: *what the substrate owes is the observable the definitions are
computed from, at a stated rate, with the frame recorded in the header.*
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import io
import json
import logging
import sys
from pathlib import Path

import mujoco
import numpy as np

from . import logging_setup
from .kinematics import body_twist
from .recording import RecordingError, open_recording

log = logging.getLogger(__name__)

EXIT_BAD_ARGS = 2
EXIT_PROVENANCE = 4


class StateError(RuntimeError):
    """``roqsim state`` cannot produce the numbers it was asked for (see the message)."""


# -- selectors ------------------------------------------------------------------------------------


def _match(patterns: list[str], names: list[str], kind: str) -> list[str]:
    """Resolve glob patterns against a model's names, or fail naming the near misses.

    An unmatched selector is an **error**, never an empty column: a silently missing series is what makes
    a downstream analysis quietly wrong, and the caller cannot tell an absent column from a zero one.
    """
    out: list[str] = []
    for pattern in patterns:
        hits = (
            fnmatch.filter(names, pattern)
            if any(c in pattern for c in "*?[")
            else ([pattern] if pattern in names else [])
        )
        if not hits:
            close = _closest(pattern, names)
            hint = (
                f" Closest: {', '.join(close)}."
                if close
                else f" Available: {', '.join(names[:12])}."
            )
            raise StateError(f"--{kind} {pattern!r} matches nothing in this world.{hint}")
        out.extend(h for h in hits if h not in out)
    return out


def _closest(pattern: str, names: list[str], limit: int = 5) -> list[str]:
    import difflib

    return difflib.get_close_matches(pattern, names, n=limit, cutoff=0.4)


def _names_of(model, objtype) -> list[str]:
    count = {
        mujoco.mjtObj.mjOBJ_BODY: model.nbody,
        mujoco.mjtObj.mjOBJ_SITE: model.nsite,
        mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
        mujoco.mjtObj.mjOBJ_SENSOR: model.nsensor,
    }[objtype]
    return [n for n in (mujoco.mj_id2name(model, objtype, i) for i in range(count)) if n]


def body_columns(model, data, names: list[str]) -> dict:
    """World pose per body: position and orientation, the thing most callers actually want."""
    out = {}
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        pos, quat = data.xpos[bid], data.xquat[bid]
        out[f"{name}.pos.x"], out[f"{name}.pos.y"], out[f"{name}.pos.z"] = (float(v) for v in pos)
        roll, pitch, yaw = _rpy(quat)
        out[f"{name}.rot.roll"], out[f"{name}.rot.pitch"], out[f"{name}.rot.yaw"] = roll, pitch, yaw
    return out


def body_twist_columns(model, data, names: list[str]) -> dict:
    """World-frame linear and angular velocity per body -- the derivative the pose columns would
    otherwise have to be differenced for, which is both noisier and only as good as the sample
    spacing. Read straight out of the solver instead."""
    out = {}
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        twist = body_twist(model, data, bid)
        out[f"{name}.vel.x"], out[f"{name}.vel.y"], out[f"{name}.vel.z"] = twist.linear
        out[f"{name}.avel.x"], out[f"{name}.avel.y"], out[f"{name}.avel.z"] = twist.angular
    return out


def site_columns(model, data, names: list[str]) -> dict:
    out = {}
    for name in names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = data.site_xpos[sid]
        out[f"{name}.pos.x"], out[f"{name}.pos.y"], out[f"{name}.pos.z"] = (float(v) for v in pos)
    return out


def joint_columns(model, data, names: list[str]) -> dict:
    """Position and velocity per named joint, sliced by the joint's own address.

    Sliced rather than indexed by joint id because a free or ball joint occupies several ``qpos``
    entries, so ``qpos[jid]`` would silently read a neighbour's value.
    """
    out = {}
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
        width = _joint_width(model, jid)
        dof = _joint_dofs(model, jid)
        for i in range(width):
            suffix = "" if width == 1 else f".{i}"
            out[f"{name}.qpos{suffix}"] = float(data.qpos[qadr + i])
        for i in range(dof):
            suffix = "" if dof == 1 else f".{i}"
            out[f"{name}.qvel{suffix}"] = float(data.qvel[dadr + i])
    return out


_JOINT_WIDTH = {
    int(mujoco.mjtJoint.mjJNT_FREE): 7,
    int(mujoco.mjtJoint.mjJNT_BALL): 4,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}
_JOINT_DOFS = {
    int(mujoco.mjtJoint.mjJNT_FREE): 6,
    int(mujoco.mjtJoint.mjJNT_BALL): 3,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}


def _joint_width(model, jid: int) -> int:
    return _JOINT_WIDTH[int(model.jnt_type[jid])]


def _joint_dofs(model, jid: int) -> int:
    return _JOINT_DOFS[int(model.jnt_type[jid])]


def mjcf_sensor_columns(model, data, names: list[str]) -> dict:
    """MJCF ``<sensor>`` readings, sliced by each sensor's declared width."""
    out = {}
    for name in names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr, dim = int(model.sensor_adr[sid]), int(model.sensor_dim[sid])
        for i in range(dim):
            suffix = "" if dim == 1 else f".{i}"
            out[f"{name}{suffix}"] = float(data.sensordata[adr + i])
    return out


def contact_rows(model, data) -> list[dict]:
    """Every current contact: the geom pair, where it is, and how hard. One row per contact."""
    rows = []
    force = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        mujoco.mj_contactForce(model, data, i, force)
        rows.append(
            {
                "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or con.geom1,
                "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or con.geom2,
                "pos.x": float(con.pos[0]),
                "pos.y": float(con.pos[1]),
                "pos.z": float(con.pos[2]),
                "dist": float(con.dist),
                "force.normal": float(force[0]),
            }
        )
    return rows


def _rpy(quat) -> tuple[float, float, float]:
    """MuJoCo's ``(w, x, y, z)`` quaternion as roll/pitch/yaw, matching the usual robotics convention."""
    w, x, y, z = (float(v) for v in quat)
    import math

    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


# -- a world's own sensors, replayed through the endpoint registry ---------------------------------
#
# `roqsim state` is a second **backend** for the endpoint registry, where the ROS 2 bridge is a wire
# backend -- which is exactly what that registry is for. `Endpoint`'s own docstring says a transport
# plugin "reads the registry and wires each port to its wire protocol, so the robot and its bridge no
# longer duplicate a hand-maintained key contract"; writing to a file instead of a socket is the same
# job. The consequence is that a new sensor type needs no change here at all.
#
# **What a replayed sensor guarantees, measured.** It is *deterministic* -- the same recording replayed
# twice gives byte-identical output -- and it is *noise-correct*, because `ctx.rng_for` is keyed on
# simulated time and the recording carries the run's seed, so the draw is the one that moment would have
# had. It is **not** bit-identical to what the live run published at that timestamp, and the reason is
# structural rather than a defect: live, `post_step` runs at the physics rate (500 Hz on a 0.002 s
# world), so a sensor's own `rate_hz` gate fires at some step *between* two recorded samples and the
# endpoint then holds a scan computed slightly earlier than the sample that recorded it. A replay only
# sees the sampled states, so it recomputes at the sample's own state. Measured on an aligned 25 Hz
# sensor at 25 fps capture, about a quarter of scans coincide exactly; the rest are the same scan a
# moment later.
#
# Recovering exactness would mean recording every sensor firing -- i.e. bagging the topic, at 26x the
# size of the whole state recording. That is the trade this architecture deliberately refuses. So read a
# replayed sensor as "what this sensor sees in this state", which is what almost every question about a
# run actually wants, and reach for the live topic when you need the published byte stream itself.


#: Output kinds, decided by the *shape* of what an endpoint's ``read()`` returns -- never by a table of
#: sensor names in this module.
KIND_SCALAR = "scalar"  # a number or a short vector -> CSV columns
KIND_ARRAY = "array"  # a per-element array (a scan) -> .npz
KIND_IMAGE = "image"  # an (H, W, C) uint8 frame -> `roqsim render --camera`

#: The widest payload still written as CSV columns. Above this a per-sample row stops being a row.
_SCALAR_MAX = 32


def replayable_sensors(ctx) -> dict:
    """The world's outward endpoints, keyed by name, with the kind each one produces.

    Only ``direction == "out"``: an input port is something the world consumes, not something it can be
    asked for.
    """
    out = {}
    for endpoint in ctx.interface.all():
        if endpoint.direction != "out" or endpoint.read is None:
            continue
        out[endpoint.name] = endpoint
    return out


def endpoint_kind(endpoint) -> str:
    """What kind of output this endpoint produces, from its declaration or its payload's shape.

    A sensor that wants to be explicit says so in ``Endpoint.backend`` under a ``"file"`` key -- the same
    inert-metadata channel ``{"ros2": {"type": "sensor_msgs.msg.LaserScan"}}`` already uses, so this needs
    no new plugin API and no new hook. Otherwise the shape decides.
    """
    declared = (endpoint.backend or {}).get("file", {}).get("kind")
    if declared in (KIND_SCALAR, KIND_ARRAY, KIND_IMAGE):
        return declared
    return _kind_of(endpoint.read())


def _kind_of(payload) -> str:
    values = _numeric(payload)
    if values is None:
        return KIND_SCALAR  # a small dataclass/tuple; flattened into columns below
    array = np.asarray(values)
    if array.ndim >= 3 and array.dtype == np.uint8:
        return KIND_IMAGE
    if array.size > _SCALAR_MAX:
        return KIND_ARRAY
    return KIND_SCALAR


def _numeric(payload):
    """The numeric content of a neutral payload, or ``None`` if it is a small structured object.

    Endpoints traffic in "neutral payloads (numpy arrays, tuples, small dataclasses)", so this handles
    all three rather than assuming one.
    """
    if payload is None:
        return None
    if isinstance(payload, np.ndarray):
        return payload
    if isinstance(payload, (int, float)):
        return np.array([payload])
    for attr in ("ranges", "data", "values"):  # the common single-array carriers
        if hasattr(payload, attr):
            return np.asarray(getattr(payload, attr))
    if isinstance(payload, (tuple, list)) and payload and isinstance(payload[0], (int, float)):
        return np.asarray(payload)
    return None


def sensor_columns(endpoint) -> dict:
    """A scalar/short-vector endpoint as CSV columns."""
    values = _numeric(endpoint.read())
    if values is None:
        return _flatten_object(endpoint.name, endpoint.read())
    flat = np.asarray(values).ravel()
    if flat.size == 1:
        return {endpoint.name: float(flat[0])}
    return {f"{endpoint.name}.{i}": float(v) for i, v in enumerate(flat)}


def _flatten_object(prefix: str, payload) -> dict:
    """A small dataclass-ish payload as columns, so a pose or a wrench needs no special case."""
    out = {}
    for key in dir(payload):
        if key.startswith("_"):
            continue
        value = getattr(payload, key, None)
        if isinstance(value, (int, float)):
            out[f"{prefix}.{key}"] = float(value)
        elif isinstance(value, (list, tuple, np.ndarray)) and len(np.ravel(value)) <= 8:
            for i, v in enumerate(np.ravel(value)):
                out[f"{prefix}.{key}.{i}"] = float(v)
    return out


def check_sensor_kind(name: str, endpoint, out: Path | None) -> str:
    """Refuse a sensor whose output does not fit the requested file, naming what does fit."""
    kind = endpoint_kind(endpoint)
    suffix = out.suffix.lower() if out else ".json"
    if kind == KIND_IMAGE:
        raise StateError(
            f"--sensor {name!r} produces images, which are not numbers. Render them instead:\n"
            f"    roqsim render --state <recording> --camera <camera> --out out.webm"
        )
    if kind == KIND_ARRAY and suffix == ".csv":
        raise StateError(
            f"--sensor {name!r} produces an array per sample (a scan), which does not fit one row per "
            f"sample. Write it to .npz instead: --out {out.with_suffix('.npz').name if out else 'scan.npz'}"
        )
    return kind


# -- gathering, writing, and the CLI --------------------------------------------------------------


def _selected(model, bodies, sites, joints, mjcf_sensors):
    """Resolve every name selector once, up front, so a typo fails before any sample is restored."""
    return {
        "bodies": _match(bodies, _names_of(model, mujoco.mjtObj.mjOBJ_BODY), "body")
        if bodies
        else [],
        "sites": _match(sites, _names_of(model, mujoco.mjtObj.mjOBJ_SITE), "site") if sites else [],
        "joints": _match(joints, _names_of(model, mujoco.mjtObj.mjOBJ_JOINT), "joint")
        if joints
        else [],
        "mjcf": _match(mjcf_sensors, _names_of(model, mujoco.mjtObj.mjOBJ_SENSOR), "sensor")
        if mjcf_sensors
        else [],
    }


def _row(model, sample, chosen, endpoints, twist: bool = False) -> dict:
    # Both clocks lead every row: a series is often read to ask what the run *cost* at some point in it
    # (a controller stalling, a sensor going expensive), and that question is unanswerable from sim time.
    row = {
        "sim_time": round(sample.sim_time, 6),
        "wall_time": round(sample.wall_time, 6),
        "sample_index": sample.index,
    }
    row.update(body_columns(model, sample.data, chosen["bodies"]))
    if twist:
        row.update(body_twist_columns(model, sample.data, chosen["bodies"]))
    row.update(site_columns(model, sample.data, chosen["sites"]))
    row.update(joint_columns(model, sample.data, chosen["joints"]))
    row.update(mjcf_sensor_columns(model, sample.data, chosen["mjcf"]))
    for endpoint in endpoints:
        row.update(sensor_columns(endpoint))
    return row


def _write_csv(rows: list[dict], out: Path | None, header: dict) -> None:
    """CSV with a commented header naming the frame, the rate and the provenance.

    Following ``insertion_task``'s stated rule: the substrate owes the observable the definitions are
    computed from, *at a stated rate, with the frame recorded in the header* -- so a reader never has to
    guess what a column means or how often it was sampled.
    """
    if not rows:
        raise StateError("no samples in range, so there is nothing to write")
    buffer = io.StringIO()
    for key, value in header.items():
        buffer.write(f"# {key}: {value}\n")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    text = buffer.getvalue()
    if out is None or str(out) == "-":
        sys.stdout.write(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


def _write_npz(arrays: dict, times, walls, out: Path, header: dict) -> None:
    """Array series keyed by sensor name, with both clocks as their own members.

    ``times`` is sim seconds and ``wall_times`` elapsed real seconds, named rather than stacked so a
    reader cannot mistake one for the other -- the CSV path states them as two columns for the same
    reason.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        meta=np.array(json.dumps(header)),
        times=np.asarray(times),
        wall_times=np.asarray(walls),
        **arrays,
    )


def run_state(
    state,
    target: str | None = None,
    *,
    bodies=None,
    sites=None,
    joints=None,
    mjcf_sensors=None,
    sensors=None,
    twist: bool = False,
    contacts: bool = False,
    at: float | None = None,
    start: float | None = None,
    stop: float | None = None,
    out: str | Path | None = None,
    check: bool = False,
) -> dict:
    """Pull numbers out of a recording. Returns the JSON record the CLI prints."""
    rec = open_recording(state)
    model, ctx = rec.build(target)
    out_path = Path(out) if out and str(out) != "-" else None

    # An endpoint's kind can only be read off a *populated* payload, and a sensor fills its port in
    # post_step -- so one sample has to be restored with the plugins running before anything can be said
    # about what they produce. Probing an unpopulated endpoint reports every sensor as a scalar.
    rec.enable_sensors()
    rec.at()
    available = replayable_sensors(ctx)
    if check:
        summary = rec.describe()
        summary["sensors"] = {name: endpoint_kind(ep) for name, ep in sorted(available.items())}
        summary["bodies"] = len(_names_of(model, mujoco.mjtObj.mjOBJ_BODY))
        summary["joints"] = _names_of(model, mujoco.mjtObj.mjOBJ_JOINT)
        summary["mjcf_sensors"] = _names_of(model, mujoco.mjtObj.mjOBJ_SENSOR)
        return summary

    chosen = _selected(model, bodies or [], sites or [], joints or [], mjcf_sensors or [])
    endpoints = []
    for name in sensors or []:
        if name not in available:
            have = ", ".join(sorted(available)) or "none"
            raise StateError(
                f"--sensor {name!r}: this world declares no such sensor. It has: {have}."
            )
        check_sensor_kind(name, available[name], out_path)
        endpoints.append(available[name])

    if not any(chosen.values()) and not endpoints and not contacts:
        raise StateError(
            "nothing selected: pass --body/--site/--joint/--mjcf-sensor/--sensor/--contacts, "
            "or --check to see what this recording offers."
        )

    header = {
        "recording": str(rec.path),
        "world": rec.world,
        "frame": "world",
        # Stated in the header for the same reason the frame is: a wall_time column whose zero is not
        # named reads as a timestamp, and would be off by ~55 years if anyone treated it as one.
        "wall_clock_origin": rec.meta.get("wall_clock_origin"),
        "rate_fps": str(rec.fps),
        "samples": len(rec),
        "seed": rec.meta.get("seed"),
        "packages": rec.meta.get("packages"),
    }

    # One moment -> JSON; a range -> CSV (or .npz for array sensors).
    if at is not None or (start is None and stop is None and out_path is None):
        sample = rec.at(at)
        if at is None:
            # Same reasoning as `roqsim render`: the last sample is a choice the caller did not make.
            log.info(
                "no --at given, so reporting the LAST of %d samples (t=%.3f s); use --at T for "
                "another moment, or --from/--to with --out for a series",
                len(rec),
                sample.sim_time,
            )
        record = {**rec.at_record(at, sample), "header": header}
        if contacts:
            record["contacts"] = contact_rows(model, sample.data)
        record["values"] = _row(model, sample, chosen, endpoints, twist)
        return record

    rows, times, walls = [], [], []
    array_series: dict[str, list] = {}
    for sample in rec.range(start, stop):
        times.append(sample.sim_time)
        walls.append(sample.wall_time)
        if any(endpoint_kind(e) == KIND_ARRAY for e in endpoints):
            for endpoint in endpoints:
                if endpoint_kind(endpoint) == KIND_ARRAY:
                    array_series.setdefault(endpoint.name, []).append(
                        np.asarray(_numeric(endpoint.read()), dtype=np.float32).copy()
                    )
        rows.append(
            _row(
                model,
                sample,
                chosen,
                [e for e in endpoints if endpoint_kind(e) != KIND_ARRAY],
                twist,
            )
        )

    if array_series:
        target_path = out_path or Path("state.npz")
        _write_npz(
            {k: np.asarray(v) for k, v in array_series.items()},
            times,
            walls,
            target_path,
            header,
        )
        return {
            "path": str(target_path.resolve()),
            "samples": len(times),
            "arrays": {k: list(np.asarray(v).shape) for k, v in array_series.items()},
            "header": header,
        }

    _write_csv(rows, out_path, header)
    return {
        "path": str(out_path.resolve()) if out_path else "-",
        "rows": len(rows),
        "columns": list(rows[0]) if rows else [],
        "header": header,
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roqsim state", description=__doc__.split("\n")[0])
    parser.add_argument(
        "target",
        nargs="?",
        help="the world, if the recording's own name cannot be resolved (normally omitted)",
    )
    parser.add_argument(
        "--state", required=True, metavar="PATH", help="a `roqsim sim --record` recording"
    )
    parser.add_argument("--at", type=float, metavar="T", help="one moment (s of sim time) -> JSON")
    parser.add_argument("--from", dest="start", type=float, metavar="T", help="range start (s)")
    parser.add_argument("--to", dest="stop", type=float, metavar="T", help="range end (s)")
    parser.add_argument("--out", metavar="PATH", help="write here; '-' is stdout (a range -> CSV)")

    sel = parser.add_argument_group("what to extract")
    sel.add_argument(
        "--body", action="extend", nargs="+", metavar="NAME", help="world pose per body"
    )
    sel.add_argument(
        "--twist",
        action="store_true",
        help="add world linear/angular velocity for each --body (read from the solver, not "
        "differenced)",
    )
    sel.add_argument(
        "--site", action="extend", nargs="+", metavar="NAME", help="world position per site"
    )
    sel.add_argument(
        "--joint", action="extend", nargs="+", metavar="NAME", help="qpos/qvel per joint"
    )
    sel.add_argument(
        "--mjcf-sensor", action="extend", nargs="+", metavar="NAME", help="an MJCF <sensor> reading"
    )
    sel.add_argument(
        "--sensor",
        action="extend",
        nargs="+",
        metavar="NAME",
        help="a sensor the WORLD declares, re-run as it was configured (see --check for the list)",
    )
    sel.add_argument("--contacts", action="store_true", help="the contacts at that moment")
    parser.add_argument("--check", action="store_true", help="report what this recording offers")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging_setup.configure(
        verbose=args.verbose, stream=sys.stderr
    )  # stdout is the JSON/CSV payload

    try:
        record = run_state(
            args.state,
            args.target,
            bodies=args.body,
            sites=args.site,
            joints=args.joint,
            mjcf_sensors=args.mjcf_sensor,
            sensors=args.sensor,
            twist=args.twist,
            contacts=args.contacts,
            at=args.at,
            start=args.start,
            stop=args.stop,
            out=args.out,
            check=args.check,
        )
    except RecordingError as err:
        print(f"roqsim state: {err}", file=sys.stderr)
        return EXIT_PROVENANCE
    except StateError as err:
        print(f"roqsim state: {err}", file=sys.stderr)
        return EXIT_BAD_ARGS

    if not (args.out and str(args.out) != "-" or args.out == "-"):
        print(json.dumps(record))
    elif args.out and str(args.out) != "-":
        print(json.dumps(record))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
