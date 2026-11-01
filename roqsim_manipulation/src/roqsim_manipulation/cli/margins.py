"""How much room a recorded arm's poses left: to its joint limits, a singularity and its torque limits.

A cell that runs is not yet a cell with margin: a pose a few degrees from a joint's stop, a wrist
almost straight, a shoulder carrying most of what it can -- each still runs in a simulation, and is
the first thing to fail on the real arm. This reads a recording and reports, for the moments named
(``--at``), each pose's margins (:func:`roqsim_manipulation.planning.pose_margins`): the distance
of every joint to its limits, the Jacobian's smallest singular value and manipulability, the UR-style
wrist, elbow and shoulder singularity measures, and the static joint torques with a payload against
the drives' limits. Over the whole run it reports each joint's peak torque from the recorded motion
(inverse dynamics on the recorded positions and velocities, accelerations differenced between
samples), the payload included.

The payload is a point mass on a body, ``--payload BODY MASS X Y Z`` in that body's frame: a
gripper's own mass is in the model already, so the payload is what it holds.

Usage::

    roqsim arm margins run.mcap --arm ur5e --site ur5e_pinch --at 3.2 7.8 \\
        --payload ur5e_hande_left_finger 0.19 0 0 0.05 --json

Exits 0 with the report; 2 for a recording, arm, site or body that is not there.
"""

from __future__ import annotations

import argparse
import json
import sys

import mujoco
import numpy as np

from roqsim.exit_status import BAD_INPUT, OK
from roqsim.recording import RecordingError, open_recording

from ..planning.kinematics import ArmKinematics
from ..planning.margins import ArmDynamics, pose_margins


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="roqsim arm margins", description=__doc__.split("\n")[0])
    p.add_argument("recording", help="a recording written by roqsim sim --record")
    p.add_argument("--arm", required=True, help="the arm's entity name (its spawn_arm name)")
    p.add_argument("--site", required=True, help="the site the Jacobian is taken at (full name)")
    p.add_argument(
        "--at", type=float, nargs="*", default=[], help="sim times of the poses to report"
    )
    p.add_argument(
        "--payload",
        nargs=5,
        metavar=("BODY", "MASS", "X", "Y", "Z"),
        help="a point mass the arm carries, on BODY at (X, Y, Z) in its frame",
    )
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    return p


def margins_report(recording, arm: str, site: str, at=(), payload=None) -> dict:
    """The report :func:`main` prints, as a dict."""
    with open_recording(recording) as rec:
        model, ctx = rec.build()
        entity = ctx.entities.get(arm)
        handle = ctx.blackboard.get(f"arm:{arm}")
        if entity is None or handle is None:
            raise KeyError(f"the recording's world has no arm {arm!r}")
        prefix = entity.meta.get("prefix", "") if isinstance(entity.meta, dict) else ""
        joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n)
            for n in handle.joint_names
        ]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            raise KeyError(f"the recording's world has no site {site!r}")
        load = None
        if payload is not None:
            body, mass, x, y, z = payload
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if bid < 0:
                raise KeyError(f"the recording's world has no body {body!r}")
            load = (bid, float(mass), [float(x), float(y), float(z)])
        dofs = [int(model.jnt_dofadr[j]) for j in joints]
        qadr = [int(model.jnt_qposadr[j]) for j in joints]
        poses = []
        for when in at:
            sample = rec.at(when)
            kin = ArmKinematics(model, joints, sid, sample.data)
            dyn = ArmDynamics(model, joints, load, sample.data)
            q = sample.data.qpos[qadr].copy()
            poses.append({"t": sample.sim_time, **pose_margins(kin, q, dyn).as_dict()})
        first = rec.sample(0)
        dyn = ArmDynamics(model, joints, load, first.data)
        times = rec.times
        qpos = rec.qpos[:, qadr]
        qvel = rec.qvel[:, dofs]
        qacc = np.gradient(qvel, times, axis=0) if len(times) > 1 else np.zeros_like(qvel)
        peak = np.zeros(len(joints))
        peak_t = np.zeros(len(joints))
        for i in range(len(times)):
            tau = np.abs(dyn.torque(qpos[i], qvel[i], qacc[i]))
            better = tau > peak
            peak = np.where(better, tau, peak)
            peak_t = np.where(better, times[i], peak_t)
        return {
            "recording": str(recording),
            "arm": arm,
            "joints": list(handle.joint_names),
            "torque_limit": dyn.limits.tolist(),
            "payload": None
            if load is None
            else {"body": payload[0], "mass": load[1], "pos": load[2]},
            "poses": poses,
            "peak_torque": peak.tolist(),
            "peak_torque_ratio": (peak / dyn.limits).tolist(),
            "peak_torque_t": peak_t.tolist(),
        }


def _num(value, fmt: str) -> str:
    """A measure as text, or ``n/a`` where the arm has none (not of the UR kind)."""
    return "n/a" if value is None else format(value, fmt)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = margins_report(args.recording, args.arm, args.site, args.at, args.payload)
    except (RecordingError, KeyError) as err:
        print(f"roqsim arm margins: {err}", file=sys.stderr)
        return BAD_INPUT
    if args.json:
        print(json.dumps(report, indent=2))
        return OK
    print(f"{report['arm']}: joints {', '.join(report['joints'])}")
    for pose in report["poses"]:
        lim = ", ".join(f"{np.degrees(r):.0f}" for r in pose["limit_rad"])
        print(
            f"t={pose['t']:.2f} s  limit margin [deg] {lim}  sigma_min {pose['sigma_min']:.3f}  "
            f"wrist {_num(pose['wrist'], '.2f')}  elbow {_num(pose['elbow'], '.2f')}  "
            f"shoulder {_num(pose['shoulder_m'], '.3f')} m  "
            f"torque/limit {max(pose['torque_ratio']):.0%}"
        )
    peak = ", ".join(f"{r:.0%}" for r in report["peak_torque_ratio"])
    print(f"peak torque / limit over the run: {peak}")
    return OK


if __name__ == "__main__":
    raise SystemExit(main())
