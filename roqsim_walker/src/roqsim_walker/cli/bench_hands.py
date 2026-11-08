"""``roqsim walker bench-hands`` -- what a walker's hands cost: hands off against hands on.

Builds one world per case -- a robot with its lidar (``--robot``, default ``turtlebot3_waffle``) and
1 or 10 walkers patrolling across it -- and measures, per physics step or per call:

* ``mj_step``: MuJoCo's own step, on a copy of the run's state (best of five batches);
* ``posing``: the walkers' posing in Python (``roqsim_walker.animation.animate``), per physics step,
  averaged over a run; with hands on it includes the motion core and the fingers;
* ``lidar``: one 360-ray scan from the robot's lidar site through ``roqsim.raycast``;
* ``frame``: one rendered 640x480 camera frame of the skinned walkers (``--no-render`` skips it).

Cases: ``off`` (hands off), ``on`` (hands on, idle: no goal) and ``moving`` (hands on, both hands
on goals that move every step, so the solver runs on every update). Prints a table, or JSON with
``--json``. Numbers are this machine's: compare cases, not machines. ``--robot none`` measures
the scan from a fixed point instead of a robot, for a quick run (``--smoke``: few steps, no render).

    roqsim walker bench-hands --walkers 1 10
    roqsim walker bench-hands --smoke --robot none --json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import mujoco
import numpy as np

CASES = ("off", "on", "moving")


def _world(n: int, hands: bool, robot: str):
    from roqsim.config import load_config_from_dict

    comps = []
    if robot != "none":
        comps.append({"spawn_robot": {"model": robot, "pose": {"position": {"x": 0.0, "y": 0.0}}}, "name": "robot"})
    for i in range(n):
        a = 2 * math.pi * i / max(n, 1)
        x, y = 1.5 * math.cos(a), 1.5 * math.sin(a)
        comps.append(
            {
                "walker": {
                    "walker": "MaleVisitorWalk" if i % 2 == 0 else "FemaleVisitorWalk",
                    "hands": hands,
                    "pose": {"position": {"x": x, "y": y}},
                },
                "name": f"w{i}",
                "components": [
                    {"navigator": {"output": "walker", "speed": 1.0, "loop": True, "goals": [[-x, -y]],
                                   "avoidance": {"steer": "none", "stop": False}}}
                ],
            }
        )
    return load_config_from_dict({"sim": {"pacing": "asap", "timestep": 0.002}, "components": comps})


def measure(n: int, case: str, robot: str = "turtlebot3_waffle", steps: int = 1000, render: bool = True) -> dict:
    """One case's numbers (see the module docstring)."""
    from roqsim import raycast
    from roqsim.engine import Engine

    import roqsim_walker.output as output

    engine = Engine(_world(n, case != "off", robot))
    engine.ctx.seed = 1
    engine.setup()
    try:
        engine.reset()
        m, d = engine.ctx.model, engine.ctx.data
        bodies = [engine.ctx.blackboard.get(f"humanoid_body:w{i}") for i in range(n)] if case != "off" else []
        spent = [0.0, 0]
        animate = output.animate

        def timed(*a, **k):
            t = time.perf_counter()
            animate(*a, **k)
            spent[0] += time.perf_counter() - t
            spent[1] += 1

        output.animate = timed
        try:
            for k in range(steps // 3 + steps):
                if k == steps // 3:
                    spent[:] = [0.0, 0]
                if case == "moving":
                    for b in bodies:
                        x, y, yaw = b.root()
                        c, s = math.cos(yaw), math.sin(yaw)
                        wave = 0.1 * math.sin(2.0 * engine.ctx.sim_time)
                        for side, off in (("l", 0.2), ("r", -0.2)):
                            b.set_hand(side, [x + 0.45 * c - off * s, y + 0.45 * s + off * c, 1.0 + wave])
                engine.step()
        finally:
            output.animate = animate
        posing = spent[0] / steps * 1e6

        scratch = mujoco.MjData(m)
        scratch.qpos[:], scratch.qvel[:] = d.qpos, d.qvel
        scratch.mocap_pos[:], scratch.mocap_quat[:], scratch.ctrl[:] = d.mocap_pos, d.mocap_quat, d.ctrl
        batches = []
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(500):
                mujoco.mj_step(m, scratch)
            batches.append((time.perf_counter() - t) / 500 * 1e6)
        step = min(batches)

        origin = np.array([0.0, 0.0, 0.3])
        for i in range(m.nsite):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or ""
            if robot != "none" and (name.endswith("lidar") or name.endswith("base_scan")):
                origin = d.site_xpos[i].copy()
                break
        ang = np.linspace(-math.pi, math.pi, 360, endpoint=False)
        dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
        buf = raycast.buffers(360)
        batches = []
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(100):
                raycast.cast(m, d, origin, dirs, cutoff=12.0, out=buf)
            batches.append((time.perf_counter() - t) / 100 * 1e6)
        lidar = min(batches)

        frame = None
        if render:
            r = mujoco.Renderer(m, 480, 640)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.0, 0.0, 0.9]
            cam.distance, cam.elevation, cam.azimuth = 5.0, -20.0, 30.0
            for _ in range(3):
                r.update_scene(d, cam)
                r.render()
            t = time.perf_counter()
            for _ in range(20):
                r.update_scene(d, cam)
                r.render()
            frame = (time.perf_counter() - t) / 20 * 1e3
            r.close()
        return {
            "walkers": n,
            "case": case,
            "mj_step_us": round(step, 1),
            "posing_us_per_step": round(posing, 1),
            "lidar_us": round(lidar, 1),
            "frame_ms": None if frame is None else round(frame, 2),
            "nmocap": int(m.nmocap),
            "ngeom": int(m.ngeom),
            "skin_bones": int(m.nskinbone),
        }
    finally:
        engine.shutdown()


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--walkers", type=int, nargs="+", default=[1, 10], help="walker counts to measure")
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=CASES)
    ap.add_argument("--robot", default="turtlebot3_waffle", help="the robot with a lidar, or 'none'")
    ap.add_argument("--steps", type=int, default=1000, help="physics steps the posing is averaged over")
    ap.add_argument("--no-render", action="store_true", help="skip the camera frame")
    ap.add_argument("--smoke", action="store_true", help="a quick run: 60 steps, no render")
    ap.add_argument("--json", action="store_true", help="print JSON rows instead of a table")
    args = ap.parse_args(argv)
    steps = 60 if args.smoke else args.steps
    rows = [
        measure(n, case, args.robot, steps, render=not (args.no_render or args.smoke))
        for n in args.walkers
        for case in args.cases
    ]
    if args.json:
        print(json.dumps(rows))
        return
    print(f"{'walkers':>7} {'case':>7} {'mj_step us':>11} {'posing us':>10} {'lidar us':>9} {'frame ms':>9} {'bones':>6}")
    for r in rows:
        frame = "-" if r["frame_ms"] is None else f"{r['frame_ms']:.2f}"
        print(f"{r['walkers']:>7} {r['case']:>7} {r['mj_step_us']:>11.1f} {r['posing_us_per_step']:>10.1f} "
              f"{r['lidar_us']:>9.1f} {frame:>9} {r['skin_bones']:>6}")


if __name__ == "__main__":
    main()
