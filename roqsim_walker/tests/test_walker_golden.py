"""Recorded reference trajectories: the walker moves exactly as it does today.

This is a **refactor guard**, not a behaviour specification. It exists so that splitting the walker's
navigation, avoidance and animation into separate packages can be judged against what the walker
actually did, rather than against a reviewer's memory of it. Nothing here says the recorded numbers
are *right* -- only that they must not change by accident.

The walk is deterministic to the bit, which is what makes this possible: with no per-waypoint dwell
the only ``random`` draw in the nav stack (``behavior.py``'s dwell) is never taken, and everything
else -- A* over the wall grid, the path follower, the clip blendspace phased by distance travelled,
the forward kinematics onto the mocap bodies -- is a pure function of the world. Two runs in one
process and two runs in separate processes were verified byte-identical before these references were
recorded. **Do not add a dwell to these worlds**; it would make them unreproducible and the test
flaky in a way that looks like a real regression.

**Two cases, because one is not enough.** ``open`` patrols a bare walled room, which covers the
follower and the gait blendspace but *not* the planner: with nothing in the way A* returns a
straight line, so the search is unobservable there. ``divider`` puts a wall across the middle and
sends the walker from one side to the other, so the recorded path is one A* actually had to search
for -- making the diagonal step cost, the grid resolution and the inflation observable.

What these references do and do not pin down, measured rather than assumed. Caught: the follower's
waypoint radius, the body turn rate, the grid resolution, and a diagonal step cost changed enough to
pick different cells. **Not** caught: a diagonal cost nudged by 0.05%, which leaves A* choosing the
same cells and is then erased entirely by line-of-sight string-pulling. That is the planner being
robust to its own tie-breaking rather than a hole here -- the guard is over the *trajectory*, and
the trajectory genuinely does not change. A refactor that alters the search materially moves it.

Regenerating (only when a change to the walk is *intended*, and the diff says so)::

    ROQSIM_REGEN_GOLDEN=1 .venv/bin/python -m pytest roqsim_walker/tests/test_walker_golden.py

Then look at the reported drift before committing: a legitimate change moves the trajectory by
centimetres and says why in the commit message; a refactor meant to preserve behaviour moves it by
nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim_walker.humanoid import JOINT_NAMES

DATA = Path(__file__).parent / "data"

#: Sampled every ``SAMPLE_EVERY`` steps for ``STEPS`` steps -- 8 sim-seconds, long enough for the
#: walker to cross a leg at 1.2 m/s and turn onto the next, so straight-line following, the turn
#: blendspace and (in ``divider``) a planned detour are all in the reference.
STEPS = 4000
SAMPLE_EVERY = 50

#: Tight enough that any real change in the path or the gait fails (those move centimetres), loose
#: enough to survive a legitimate reassociation of floating-point arithmetic during a refactor.
TOLERANCE = 1e-6

_BASE = {
    "walker": "MaleVisitorWalk",
    "speed": 1.2,
    "loop": True,
    "arrival_radius": 0.25,
    "avoidance": False,
    "skin": False,  # capsules: the skin is a 5 MB OBJ and does not move the bones
}

CASES = {
    # A bare walled room: the follower and the gait, with the planner returning straight lines.
    "open": (None, [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]),
    # A wall across the middle: every leg is a detour A* had to search for.
    "divider": (DATA / "walled_room_divider.xml", [[-2.5, 0.0], [2.5, 0.0]]),
}


def _tracked_mocapids(model) -> list[int]:
    """Every mocap body the walker owns, in a fixed order: the nav root plus each skeleton bone."""
    ids = []
    for name in ["pelvis", *JOINT_NAMES]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ped/{name}")
        if bid >= 0 and model.body_mocapid[bid] >= 0:
            ids.append(int(model.body_mocapid[bid]))
    return ids


def _drift(recorded: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Per-sample, per-value difference -- comparing **rotations**, not their representations.

    A unit quaternion and its negation are the same rotation, and the blend that produces these
    picks a sign by whichever branch a near-tie falls on. Comparing raw components therefore reports
    a difference of up to 2.0 for a body that did not move at all -- which happened: one bone, one
    frame, ``q_new == -q_ref`` to the bit with identical positions and ``|dot| == 1``. A guard that
    fires on that is worse than no guard, because the first thing it teaches is to ignore it.

    Positions are compared directly; each 4-component quaternion block is compared against whichever
    of ``+q``/``-q`` is nearer, so a sign flip reads as zero and a real rotation still reads as
    itself.
    """
    a = recorded.reshape(len(recorded), -1, 7)
    b = expected.reshape(len(expected), -1, 7)
    pos = np.abs(a[..., :3] - b[..., :3])
    quat = np.minimum(np.abs(a[..., 3:] - b[..., 3:]), np.abs(a[..., 3:] + b[..., 3:]))
    return np.concatenate([pos, quat], axis=-1).reshape(len(recorded), -1)


def _record(world, waypoints) -> np.ndarray:
    """``(samples, bones * 7)`` of mocap position + quaternion, sampled along one run."""
    sim = {"pacing": "asap"}
    if world is not None:
        sim["world"] = str(world)
    engine = Engine(
        load_config_from_dict(
            {
                "sim": sim,
                "components": [{"walker": {**_BASE, "waypoints": waypoints}, "name": "ped"}],
            }
        )
    )
    engine.setup()
    engine.reset()
    model, data = engine.ctx.model, engine.ctx.data
    mocapids = _tracked_mocapids(model)
    assert mocapids, "no walker mocap bodies found -- the body naming changed"
    frames = []
    try:
        for step in range(STEPS):
            engine.step()
            if step % SAMPLE_EVERY == 0:
                frames.append(
                    np.concatenate(
                        [np.concatenate([data.mocap_pos[i], data.mocap_quat[i]]) for i in mocapids]
                    )
                )
    finally:
        engine.shutdown()
    return np.array(frames)


@pytest.mark.parametrize("case", sorted(CASES))
def test_walk_matches_the_recorded_reference(case):
    world, waypoints = CASES[case]
    reference = DATA / f"walker_golden_{case}.npz"
    recorded = _record(world, waypoints)

    if os.environ.get("ROQSIM_REGEN_GOLDEN"):
        DATA.mkdir(parents=True, exist_ok=True)
        if reference.exists():
            before = np.load(reference)["frames"]
            if before.shape == recorded.shape:
                print(f"\n{case}: golden drift {_drift(recorded, before).max():.6g}")
        np.savez_compressed(reference, frames=recorded)
        pytest.skip(f"regenerated {reference.name} -- inspect the drift above before committing")

    assert reference.exists(), (
        f"{reference} is missing. Record it with ROQSIM_REGEN_GOLDEN=1 pytest {Path(__file__).name}"
    )
    expected = np.load(reference)["frames"]
    assert recorded.shape == expected.shape, (
        f"{case}: recorded {recorded.shape}, reference {expected.shape} -- the tracked body set or "
        "the sampling changed, so the reference no longer describes the same measurement"
    )

    drift = _drift(recorded, expected)
    worst = int(drift.max(axis=1).argmax())
    assert drift.max() <= TOLERANCE, (
        f"{case}: the walk changed -- max drift {drift.max():.6g} at sample {worst} of "
        f"{len(expected)} ({worst * SAMPLE_EVERY} steps in). If that was intended, regenerate with "
        "ROQSIM_REGEN_GOLDEN=1 and say why in the commit message."
    )


def test_the_divider_case_actually_forces_a_detour():
    """Guards the guard: if this stops holding, ``divider`` has silently become another ``open``.

    The divider spans y in [-2.5, 2.5] at x = 0 and the goals sit either side of it on y = 0, so a
    straight line is blocked and any real path leaves the corridor by a wide margin.
    """
    frames = np.load(DATA / "walker_golden_divider.npz")["frames"]
    pelvis_y = frames[:, 1]  # first tracked body, y of its mocap_pos
    assert np.abs(pelvis_y).max() > 2.5, (
        "the recorded divider path never leaves y = 0, so A* was not routing around anything"
    )
