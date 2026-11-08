"""A walker's hands on: the motion core driving its mocap skeleton.

With ``hands: true`` a walker gets a :class:`WalkerBody`, which the animation calls on every update
after the gait's blendspace (:func:`roqsim_walker.animation.animate`): it solves the hands' goals,
the posture and the gaze with :func:`roqsim_motion.ik.solve` over the gait's pose, blends each
hand's solution in by its own weight (the trunk by the stronger hand's), and writes the thirty
finger bones from each hand's pose. It is published on the blackboard as ``humanoid_body:<name>``,
the interface the moves of :mod:`roqsim_motion.moves` drive any humanoid body through; a robot's
body (:mod:`roqsim_motion.robot`) offers the same.

A walker with ``hands: false`` has none, and its animation is the one it always had.
"""

from __future__ import annotations


import mujoco
import numpy as np

from roqsim.pose import yaw_to_quat
from roqsim_motion import hands as H
from roqsim_motion import quat as Q
from roqsim_motion.body import HumanoidBody
from roqsim_motion.description import Description
from roqsim_motion.ik import HandGoal, Posture, solve
from roqsim_walker.humanoid import JOINT_NAMES, PRESS_RADIUS
from roqsim_walker.motion import blend_quats

#: How fast a hand's goal weight eases in and out (1/s), and the trunk joints the solver bends.
WEIGHT_RATE = 2.5
#: How fast the posture may change: rad/s for turn, lean and bend, m/s for the crouch.
POSTURE_RATE = {"turn": 1.5, "lean": 1.0, "crouch": 0.4, "bend": 1.5}
_TRUNK = ("pelvis", "spine", "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r")


def layouts_for(skeleton, block: dict | None = None, mesh: str | None = None, tpose: bool = False,
                flip=None) -> dict:
    """Each hand's layout for a walker's skeleton: measured on its character ``mesh`` where that has
    finger weights (:func:`roqsim_walker.skin.fit_hands`), so the bones turn the skin about its own
    knuckles; else the default hand scaled to its forearm. A side the blueprint's ``hands`` block
    states is the default hand with its overrides."""
    fitted = _fitted(mesh, tpose, skeleton, flip) if mesh else None
    if mesh and fitted is None:
        raise ValueError(f"{mesh}: its finger weights do not show a thumb and four fingers to measure the hands on")
    out = {}
    for side in ("l", "r"):
        stated = (block or {}).get(side)
        if fitted is not None and not stated:
            f = fitted[side]
            out[side] = H.make_layout(side, f["bases"], f["lengths"], f["directions"], f["palm"], f["grip"])
            continue
        forearm = float(np.linalg.norm(skeleton.offset(f"wrist_{side}")))
        out[side] = H.layout(side, scale=forearm / 0.24, block=stated)
    return out


_FITS: dict = {}


def _fitted(mesh, tpose, skeleton, flip):
    """:func:`roqsim_walker.skin.fit_hands`, once per mesh and skeleton in a process: a world of
    several workers of one character measures its hands once."""
    import os

    from roqsim_walker.skin import fit_hands

    key = (os.path.abspath(mesh), os.path.getmtime(mesh), bool(tpose), flip,
           tuple(sorted((skeleton.offsets or {}).items())), skeleton.root_height)
    if key not in _FITS:
        _FITS[key] = fit_hands(mesh, tpose=tpose, skeleton=skeleton, flip=flip)
    return _FITS[key]


def description_for(skeleton) -> Description:
    from roqsim_walker.humanoid import SKELETON

    return Description({j.name: tuple(skeleton.offset(j.name)) for j in SKELETON}, skeleton.root_height)


class WalkerBody(HumanoidBody):
    """The walker's side of the motion core (see the module docstring)."""

    has_proxies = True

    def __init__(self, name: str, st, model, layouts: dict):
        self.st = st
        self.layouts = layouts
        self.desc = description_for(st.skeleton)
        self._fingers = {
            side: np.array([int(model.body_mocapid[_body(model, f"{name}/{b}")]) for b in H.bone_names(side)])
            for side in ("l", "r")
        }
        self._wrist = {side: st.mocap[f"wrist_{side}"] for side in ("l", "r")}
        self._grip = {side: _site(model, f"{name}/grip_{side}") for side in ("l", "r")}
        self._frames = {side: H.FingerFrames(layouts[side]) for side in ("l", "r")}
        self._data = None  # the run's data, set by the plugin at configure
        self._proxy = {side: int(model.body_mocapid[_body(model, f"{name}/press_{side}")]) for side in ("l", "r")}
        super().__init__(name)

    # -- HumanoidBody ---------------------------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        self.weight = {"l": 0.0, "r": 0.0}
        self.last = {"l": None, "r": None}
        self.residual = {"l": 0.0, "r": 0.0}
        self.pressing = {"l": False, "r": False}
        self._last_posture = None
        self.st.face = None
        self.st.turn_rate = None

    def grip(self, data, side):
        sid = self._grip[side]
        return data.site_xpos[sid].copy(), _mat_quat(data.site_xmat[sid])

    def grip_site(self, side: str) -> str:
        return f"{self.name}/grip_{side}"

    def shoulder(self, data, side):
        return data.mocap_pos[self.st.mocap[f"shoulder_{side}"]].copy()

    def arm_reach(self, side: str) -> float:
        return 0.97 * self.desc.arm_length(side) + float(np.linalg.norm(self.layouts[side].grip_pos))

    def root(self):
        return float(self.st.pos[0]), float(self.st.pos[1]), float(self.st.yaw)

    def carry_height(self) -> float:
        return self.desc.root_height - 0.1

    def limit_turn(self, rate):
        self.st.turn_rate = None if rate is None else float(rate)

    def face(self, yaw):
        self.st.face = None if yaw is None else float(yaw)

    def tip(self, data, side: str):
        wid = self._wrist[side]
        return self._frames[side].tip(data.mocap_pos[wid], data.mocap_quat[wid], self.hand[side].curls)

    def fingertips(self, data, side: str):
        wid = self._wrist[side]
        return np.array([
            self._frames[side].tip(data.mocap_pos[wid], data.mocap_quat[wid], self.hand[side].curls, finger=f)
            for f in range(5)
        ])

    def press_aim(self, side: str, pose: str):
        lay = self.layouts[side]
        pos, rot = H.finger_fk(lay, np.zeros(3), Q.IDENTITY, np.array(H.POSES[pose], float))
        tip = H.fingertips(lay, np.zeros(3), Q.IDENTITY, np.array(H.POSES[pose], float))[1]
        along = tip - pos[5]  # the index's last bone
        return Q.rotate(Q.inv(lay.grip_quat), along / np.linalg.norm(along))

    def press_proxy(self, data, side: str, on: bool) -> None:
        """Pose ``side``'s press proxy on its index fingertip from now on (``on``), or park it."""
        self.pressing[side] = bool(on)
        self._place_proxy(data, side)

    def _place_proxy(self, data, side: str) -> None:
        """The proxy's front on the drawn index fingertip: its centre a radius back along the
        finger's last bone, so what pushes a button is the fingertip a viewer sees."""
        pid = self._proxy[side]
        if self.pressing[side]:
            wid = self._wrist[side]
            wp, wq = data.mocap_pos[wid], data.mocap_quat[wid]
            curls = self.hand[side].curls
            tip = self._frames[side].tip(wp, wq, curls)
            last = self._frames[side].world(wp, wq, curls)[0][5]  # the index's last bone
            along = (tip - last) / max(float(np.linalg.norm(tip - last)), 1e-9)
            data.mocap_pos[pid] = tip - PRESS_RADIUS * along
        else:
            data.mocap_pos[pid] = (0.0, 0.0, -50.0)

    def settled(self, side: str) -> bool:
        return self.goal[side] is None or self.weight[side] > 0.999

    def wrist_poses(self, data) -> dict:
        return {s: (data.mocap_pos[self._wrist[s]].copy(), data.mocap_quat[self._wrist[s]].copy()) for s in "lr"}

    # -- the animation's hooks ------------------------------------------------------------------
    def pose(self, st, q, body, root_z, dt) -> float:
        """Blend the solved hands, posture and gaze over the gait's pose ``q`` in place; the pelvis
        drop (m) a crouch asks for."""
        for side in ("l", "r"):
            self.hand[side].advance(dt)
            if self.closing[side] is not None:
                self.hand[side].close_on(self.closing[side])
            if dt > 0.0:
                aim = 1.0 if self.goal[side] is not None else 0.0
                step = WEIGHT_RATE * dt
                self.weight[side] = min(max(self.weight[side] + max(-step, min(step, aim - self.weight[side])), 0.0), 1.0)
            if self.goal[side] is None and self.weight[side] < 1e-3:
                self.last[side] = None
        weight = max(self.weight.values())
        gaze_w = 1.0 if self.gaze is not None else 0.0
        if weight < 1e-3 and gaze_w == 0.0 and self.posture is None:
            self._last_posture = None
            return 0.0
        shift = st.ground if st.ground is not None else 0.0
        root = np.array([body[0], body[1], st.skeleton.root_height + root_z - shift])
        local = {n: q[i].copy() for i, n in enumerate(JOINT_NAMES)}
        goals = {}
        for s in ("l", "r"):
            if self.goal[s] is None:
                goals[s] = None
                continue
            pos, quat = self.goal[s]
            if quat is None and self.goal_frame[s] == "grip":
                # The grip frame's position, the hand turned as it is: the wrist goes where the grip
                # is then on the point.
                wq = self._data.mocap_quat[self._wrist[s]]
                from roqsim_motion.quat import rotate

                pos = pos - rotate(wq, self.layouts[s].grip_pos)
            goals[s] = HandGoal(pos, quat)
        heading = yaw_to_quat(st.yaw)
        sol = solve(self.desc, root, heading, local, goals, self.layouts, self.posture, self.gaze)
        # The posture the solver chose eases from the last one at a bounded rate: a goal that crosses
        # from bending to crouching would otherwise move the body, and what it holds, at once.
        eased = self._ease(sol.posture, dt)
        if eased is not None:
            sol = solve(self.desc, root, heading, local, goals, self.layouts, eased, self.gaze)
        self._last_posture = sol.posture
        self.residual.update(sol.residual)
        for side in ("l", "r"):
            if self.goal[side] is not None:
                self.last[side] = {f"{j}_{side}": sol.rotations[f"{j}_{side}"] for j in ("shoulder", "elbow", "wrist")}
        over = []
        trunk_w = max(weight, 1.0 if self.posture is not None else 0.0)
        for name in _TRUNK:
            if name in sol.rotations:
                over.append((name, sol.rotations[name], trunk_w))
        if "head" in sol.rotations:
            over.append(("head", sol.rotations["head"], gaze_w))
        for side in ("l", "r"):
            if self.last[side] is not None:
                over += [(n, r, self.weight[side]) for n, r in self.last[side].items()]
        for name, rot, w in over:
            j = JOINT_NAMES.index(name)
            q[j] = blend_quats(q[j : j + 1], np.asarray(rot)[None, :], w)[0]
        return sol.root_drop * trunk_w

    def _ease(self, posture: Posture, dt: float):
        """``posture`` limited to :data:`POSTURE_RATE` from the last update's, or ``None`` when it
        is within it already."""
        last = self._last_posture
        if last is None or dt <= 0.0:
            return None
        out, changed = {}, False
        for key in ("turn", "lean", "crouch", "bend"):
            new, old = float(getattr(posture, key)), float(getattr(last, key))
            step = POSTURE_RATE[key] * dt
            value = old + max(-step, min(step, new - old))
            changed |= abs(value - new) > 1e-9
            out[key] = value
        return Posture(**out) if changed else None

    def write_fingers(self, data) -> None:
        """Pose the thirty finger bones on the wrists as the animation left them."""
        for side in ("l", "r"):
            wid = self._wrist[side]
            pos, rot = self._frames[side].world(data.mocap_pos[wid], data.mocap_quat[wid], self.hand[side].curls)
            ids = self._fingers[side]
            data.mocap_pos[ids] = pos
            data.mocap_quat[ids] = rot
            if self.pressing[side]:
                self._place_proxy(data, side)


def _body(model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"hand bone {name} not in the model (was the walker built with hands: true?)")
    return bid


def _site(model, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise ValueError(f"grip site {name} not in the model (was the walker built with hands: true?)")
    return sid


def _mat_quat(mat) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, float).reshape(9))
    return q


