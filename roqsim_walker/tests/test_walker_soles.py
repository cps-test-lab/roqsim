"""A bundled walker's skin on its bones and its soles on the floor, as MuJoCo draws it.

The skin is read back from MuJoCo itself (``mjv_updateScene`` skins the mesh into
``MjvScene.skinvert``; no GL context needed), posed as the walker plugin poses it: forward kinematics
of a clip frame, then the foot grounding (``animation._foot_ground``).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim_walker import animation, skin
from roqsim_walker.blueprint import resolve_walker
from roqsim_walker.cli import import_actor
from roqsim_walker.humanoid import JOINT_NAMES, build_humanoid, forward_kinematics

#: How far (m) the lowest point of the skin may be from the floor, standing and walking.
SOLE_TOLERANCE = 0.001
#: How far (m) a collision capsule's radius may be from the skin's own spread round its bone (the
#: 75th percentile of the distance of the vertices that bone carries from its axis).
CAPSULE_TOLERANCE = 0.001

#: The blueprints whose soles are measured to the bar above.
WALKERS = ["WorkerHiVisWalk"]


class _Skinned:
    """A walker built into a model with its skin, posed by the animation's own grounding."""

    def __init__(self, walker: str):
        self.blueprint = bp = resolve_walker(walker)
        spec = mujoco.MjSpec()
        build_humanoid(
            spec,
            name="ped",
            mesh=bp["mesh"],
            materials=bp["materials"],
            tpose=bp["tpose"],
            skeleton=bp["skeleton"],
            collision=bp["collision"],
            flip=bp["flip"],
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.st = animation.make_anim_state(self.model, {"name": "ped", **bp})
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.args = (mujoco.MjvOption(), mujoco.MjvPerturb(), mujoco.MjvCamera())

    def lowest(self, joint_rot: dict, root_z: float = 0.0) -> float:
        """The lowest point of the skin (m above the floor) with the joints turned ``joint_rot``."""
        skel = self.st.skeleton
        poses = forward_kinematics([0.0, 0.0, skel.root_height + root_z], 0.0, joint_rot, skel)
        self.st.ground = None
        animation.write_pose(self.data, self.st, animation._foot_ground(poses, self.st))
        mujoco.mj_kinematics(self.model, self.data)
        opt, pert, cam = self.args
        mujoco.mjv_updateScene(
            self.model, self.data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        return float(np.asarray(self.scene.skinvert).reshape(-1, 3)[:, 2].min())


@pytest.fixture(scope="module", params=WALKERS)
def walker(request):
    return _Skinned(request.param)


def test_the_soles_touch_the_floor_standing(walker):
    rest = {n: np.array([1.0, 0.0, 0.0, 0.0]) for n in JOINT_NAMES}
    assert abs(walker.lowest(rest)) <= SOLE_TOLERANCE


def test_the_soles_touch_the_floor_through_the_walk_clip(walker):
    clip = walker.st.walk
    assert walker.blueprint["motion"].get("walk"), "the walk clip did not resolve"
    lows = np.array(
        [
            walker.lowest(
                {n: clip.joint_rot[i, j] for j, n in enumerate(JOINT_NAMES)}, clip.root_z[i]
            )
            for i in range(clip.num_frames)
        ]
    )
    worst = int(np.abs(lows).argmax())
    assert abs(lows[worst]) <= SOLE_TOLERANCE, (
        f"frame {worst} of {clip.num_frames}: the skin's lowest point is "
        f"{1000 * lows[worst]:+.1f} mm off the floor"
    )


def test_each_capsule_is_the_skin_round_its_bone(walker):
    """The skin stays where the collision capsules say it is round the bones it is bound to."""
    bp = walker.blueprint
    parts = skin._load_parts(bp["mesh"])
    raw = np.vstack([v for _, v, _, _ in parts])
    weights = skin._load_weights(bp["mesh"])(raw)
    verts = skin._aligner(raw, flip=bp["flip"])(raw)
    # one vertex per position: a texture seam splits a vertex, but not the surface
    _, first = np.unique(np.round(verts, 6), axis=0, return_index=True)
    verts, weights = verts[first], weights[first]
    bind = {n: pq[0] for n, pq in skin._bind_bones(bp["tpose"], walker.st.skeleton).items()}
    spread = import_actor.measure_collision(verts, weights, bind)
    assert set(spread) == set(bp["collision"])
    off = {j: round(1000 * (bp["collision"][j] - r), 1) for j, r in spread.items()}
    assert max(abs(v) for v in off.values()) <= 1000 * CAPSULE_TOLERANCE, f"mm off: {off}"
