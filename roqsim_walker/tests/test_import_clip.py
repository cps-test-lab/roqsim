"""``roqsim walker import-clip``: a BVH capture onto the walker, each bone pointing as its source's."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roqsim_walker.cli import import_clip as ic
from roqsim_walker.humanoid import DEFAULT_SKELETON, JOINT_NAMES, forward_kinematics
from roqsim_walker.motion import Clip

#: A Mixamo-named rig in a T-pose, Y up, facing +Z, in centimetres: (name, parent, offset, end).
RIG = [
    ("mixamorig:Hips", None, (0, 100, 0), None),
    ("mixamorig:Spine", "mixamorig:Hips", (0, 10, 0), None),
    ("mixamorig:Spine1", "mixamorig:Spine", (0, 10, 0), None),
    ("mixamorig:Spine2", "mixamorig:Spine1", (0, 10, 0), None),
    ("mixamorig:Neck", "mixamorig:Spine2", (0, 15, 0), None),
    ("mixamorig:Head", "mixamorig:Neck", (0, 10, 0), (0, 15, 0)),
    ("mixamorig:LeftShoulder", "mixamorig:Spine2", (6, 12, 0), None),
    ("mixamorig:LeftArm", "mixamorig:LeftShoulder", (12, 0, 0), None),
    ("mixamorig:LeftForeArm", "mixamorig:LeftArm", (26, 0, 0), None),
    ("mixamorig:LeftHand", "mixamorig:LeftForeArm", (24, 0, 0), (10, 0, 0)),
    ("mixamorig:RightShoulder", "mixamorig:Spine2", (-6, 12, 0), None),
    ("mixamorig:RightArm", "mixamorig:RightShoulder", (-12, 0, 0), None),
    ("mixamorig:RightForeArm", "mixamorig:RightArm", (-26, 0, 0), None),
    ("mixamorig:RightHand", "mixamorig:RightForeArm", (-24, 0, 0), (-10, 0, 0)),
    ("mixamorig:LeftUpLeg", "mixamorig:Hips", (10, -5, 0), None),
    ("mixamorig:LeftLeg", "mixamorig:LeftUpLeg", (0, -45, 0), None),
    ("mixamorig:LeftFoot", "mixamorig:LeftLeg", (0, -45, 0), None),
    ("mixamorig:LeftToeBase", "mixamorig:LeftFoot", (0, -6, 12), (0, 0, 6)),
    ("mixamorig:RightUpLeg", "mixamorig:Hips", (-10, -5, 0), None),
    ("mixamorig:RightLeg", "mixamorig:RightUpLeg", (0, -45, 0), None),
    ("mixamorig:RightFoot", "mixamorig:RightLeg", (0, -45, 0), None),
    ("mixamorig:RightToeBase", "mixamorig:RightFoot", (0, -6, 12), (0, 0, 6)),
]

#: The walker's bones and the source bones they stand for (start, end).
PAIRS = {
    ("shoulder_l", "elbow_l"): ("mixamorig:LeftArm", "mixamorig:LeftForeArm"),
    ("elbow_l", "wrist_l"): ("mixamorig:LeftForeArm", "mixamorig:LeftHand"),
    ("shoulder_r", "elbow_r"): ("mixamorig:RightArm", "mixamorig:RightForeArm"),
    ("elbow_r", "wrist_r"): ("mixamorig:RightForeArm", "mixamorig:RightHand"),
    ("hip_l", "knee_l"): ("mixamorig:LeftUpLeg", "mixamorig:LeftLeg"),
    ("knee_l", "ankle_l"): ("mixamorig:LeftLeg", "mixamorig:LeftFoot"),
    ("hip_r", "knee_r"): ("mixamorig:RightUpLeg", "mixamorig:RightLeg"),
    ("knee_r", "ankle_r"): ("mixamorig:RightLeg", "mixamorig:RightFoot"),
}
# The trunk is not among them: the walker's spine and head stand for several source bones (Spine to
# Spine2, Neck), so the bone between them follows the source's Spine2 and not its neck.


def _bvh(frames, rig=RIG, fps=30.0) -> str:
    """A BVH of ``rig`` with ``frames``: per frame, the root's position and every bone's (z, x, y)
    rotation in degrees."""
    children: dict = {}
    for name, parent, _off, _end in rig:
        children.setdefault(parent, []).append(name)
    info = {name: (off, end) for name, _p, off, end in rig}
    lines = ["HIERARCHY"]

    def write(name, depth, root=False):
        pad = "  " * depth
        off, end = info[name]
        lines.append(f"{pad}{'ROOT' if root else 'JOINT'} {name}")
        lines.append(pad + "{")
        lines.append(f"{pad}  OFFSET {off[0]} {off[1]} {off[2]}")
        ch = "Xposition Yposition Zposition " if root else ""
        lines.append(f"{pad}  CHANNELS {6 if root else 3} {ch}Zrotation Xrotation Yrotation")
        for child in children.get(name, []):
            write(child, depth + 1)
        if end is not None:
            lines.extend(
                [
                    f"{pad}  End Site",
                    pad + "  {",
                    f"{pad}    OFFSET {end[0]} {end[1]} {end[2]}",
                    pad + "  }",
                ]
            )
        lines.append(pad + "}")

    write(rig[0][0], 0, root=True)
    lines += ["MOTION", f"Frames: {len(frames)}", f"Frame Time: {1.0 / fps}"]
    lines += [" ".join(f"{v:.6f}" for v in f) for f in frames]
    return "\n".join(lines)


def _frame(rotations=None, root=(0.0, 100.0, 0.0)):
    rotations = rotations or {}
    row = list(root)
    for name, *_ in RIG:
        row += list(rotations.get(name, (0.0, 0.0, 0.0)))
    return row


def _walker_poses(clip, f):
    q = {name: clip.joint_rot[f, k] for k, name in enumerate(JOINT_NAMES)}
    return forward_kinematics([0.0, 0.0, 0.93], 0.0, q, skeleton=DEFAULT_SKELETON)


def _source_dirs(text, f):
    bones, frames, _ = ic.parse_bvh(text)
    _, pos = ic.pose_frames(bones, frames[f : f + 1])
    index = {b.name: k for k, b in enumerate(bones)}
    c = ic.basis("y", "+z")
    return {pair: c @ (pos[0, index[b]] - pos[0, index[a]]) for pair, (a, b) in PAIRS.items()}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_the_t_pose_at_rest_is_a_t_pose_on_the_walker():
    clip = ic.convert(_bvh([_frame()]))
    poses = _walker_poses(clip, 0)
    for side, left in (("l", 1.0), ("r", -1.0)):
        arm = _unit(poses[f"wrist_{side}"][0] - poses[f"shoulder_{side}"][0])
        assert np.allclose(arm, [0.0, left, 0.0], atol=1e-6)  # out to its side
        leg = _unit(poses[f"ankle_{side}"][0] - poses[f"hip_{side}"][0])
        assert np.allclose(leg, [0.0, 0.0, -1.0], atol=1e-6)
    assert np.allclose(_unit(poses["head"][0] - poses["spine"][0]), [0, 0, 1], atol=1e-6)


def test_every_bone_points_where_its_source_bone_does_frame_by_frame():
    rng = np.random.default_rng(3)
    frames = []
    for _ in range(5):
        rot = {name: rng.uniform(-50, 50, 3) for name, *_ in RIG[1:]}
        frames.append(_frame(rot))
    text = _bvh(frames)
    clip = ic.convert(text)
    for f in range(len(frames)):
        poses = _walker_poses(clip, f)
        for (a, b), theirs in _source_dirs(text, f).items():
            ours = _unit(poses[b][0] - poses[a][0])
            assert np.allclose(ours, _unit(theirs), atol=1e-6), (f, a, b)


def test_the_first_frames_heading_and_travel_are_dropped_and_a_cycle_keeps_its_stride():
    turned = [
        _frame({"mixamorig:Hips": (0.0, 0.0, 90.0)}, root=(0.0, 100.0, 50.0 * k)) for k in range(3)
    ]
    clip = ic.convert(_bvh(turned), loop=True)
    poses = _walker_poses(clip, 0)
    fwd = poses["pelvis"][1]
    from roqsim_walker.humanoid import quat_rotate

    assert np.allclose(quat_rotate(fwd, [1, 0, 0]), [1, 0, 0], atol=1e-6)  # facing the walker's +x
    legs = 90.0  # the rig's hip to ankle, cm
    assert clip.loop and clip.stride_len == pytest.approx(100.0 * 0.84 / legs, rel=1e-6)
    assert clip.fps == pytest.approx(30.0)


def test_a_joint_not_found_is_named_with_the_captures_bones_and_mapped_by_hand():
    rig = [
        (n.replace("LeftToeBase", "LToeTip"), p and p.replace("LeftToeBase", "LToeTip"), o, e)
        for n, p, o, e in RIG
    ]
    text = _bvh([[0.0, 100.0, 0.0] + [0.0] * 3 * len(rig)], rig=rig)
    with pytest.raises(
        ic.BvhError, match=r"no bone found for toe_l; the capture's bones are .*LToeTip"
    ):
        ic.convert(text)
    clip = ic.convert(text, overrides={"toe_l": "mixamorig:LToeTip"})
    assert clip.num_frames == 1
    with pytest.raises(ic.BvhError, match="the walker has no joint 'tail'"):
        ic.convert(text, overrides={"tail": "mixamorig:Hips"})


def test_the_command_writes_a_clip_a_walker_reads(tmp_path, capsys):
    src = tmp_path / "wave.bvh"
    src.write_text(_bvh([_frame(), _frame({"mixamorig:LeftArm": (60.0, 0.0, 0.0)})]))
    out = tmp_path / "anims" / "wave.npz"
    ic.main([str(src), "--out", str(out)])
    clip = Clip.load(out)
    assert clip.num_frames == 2 and not clip.loop
    assert "wrote" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="not a BVH file"):
        (tmp_path / "bad.bvh").write_text("nonsense")
        ic.main([str(tmp_path / "bad.bvh"), "--out", str(tmp_path / "x.npz")])


def test_an_angle_about_the_source_up_axis_turns_the_walker_about_its_own():
    # The left arm swung forward 90 degrees about the source's up axis (Y): it points the way the
    # character faces, the walker's +x.
    clip = ic.convert(_bvh([_frame({"mixamorig:LeftArm": (0.0, 0.0, -90.0)})]))
    poses = _walker_poses(clip, 0)
    arm = _unit(poses["elbow_l"][0] - poses["shoulder_l"][0])
    assert np.allclose(arm, [1.0, 0.0, 0.0], atol=1e-6), arm
    assert math.isclose(float(np.linalg.norm(arm)), 1.0)
