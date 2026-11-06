# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""``roqsim splat frame`` finds a capture's floor, up axis and walls, and proposes its pose."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from roqsim.config import load_config_from_dict
from roqsim.pose import rpy_to_quat
from roqsim_render_splat import frame, synthetic
from roqsim_render_splat.renderer import splat_placement


def _capture(tmp_path, rpy, position, scale, tilt_deg=0.0):
    """The example room captured with a pose; *tilt_deg* tips the capture's up off its axis."""
    roll, pitch, yaw = rpy
    quat = rpy_to_quat(roll + math.radians(tilt_deg), pitch, yaw)
    ply, _ = synthetic.write_example(tmp_path, quat, position, scale)
    return ply, quat


def _placed_box_error(result, quat, position, scale) -> float:
    """How far (m) the proposed placement puts the room's corners from where they are."""
    true = splat_placement(
        {
            "pose": {
                "position": dict(zip("xyz", position, strict=True)),
                "orientation": dict(zip("wxyz", quat, strict=True)),
            },
            "scale": scale,
        }
    )
    proposed = splat_placement({"pose": result["pose"], "scale": result["scale"]})
    corners = np.array([[x, y, z] for x in (-2, 2) for y in (-1.5, 1.5) for z in (0, 2.4)])
    # The capture point of each corner, placed back by the proposal.
    capture = (true[0].T @ (corners - true[1]).T).T / true[2]
    placed = (proposed[0] @ (capture * proposed[2]).T).T + proposed[1]
    return float(np.abs(placed - corners).max())


def test_a_captures_pose_and_scale_are_recovered(tmp_path):
    rpy, position, scale = (-1.5708, 0.0, 0.35), (0.0, 0.0, 1.0), 0.5
    ply, quat = _capture(tmp_path, rpy, position, scale)
    result = frame.propose(ply, known_height=2.4)
    assert result["up"]["capture_axis"] == "-y" and result["up"]["tilt_deg"] < 0.5
    assert result["scale"] == pytest.approx(scale, rel=0.01)
    assert result["footprint_m"] == pytest.approx([4.0, 3.0], abs=0.02)
    assert _placed_box_error(result, quat, position, scale) < 0.01


def test_a_tilted_capture_is_levelled_and_its_tilt_reported(tmp_path):
    ply, quat = _capture(tmp_path, (-1.5708, 0.0, -0.6), (0.4, -0.3, 0.7), 0.25, tilt_deg=3.0)
    result = frame.propose(ply, scale=0.25)
    assert result["up"]["tilt_deg"] == pytest.approx(3.0, abs=0.5)
    assert _placed_box_error(result, quat, (0.4, -0.3, 0.7), 0.25) < 0.01


def test_the_scale_is_not_guessed(tmp_path, capsys):
    ply, _ = _capture(tmp_path, (-1.5708, 0.0, 0.35), (0.0, 0.0, 1.0), 0.5)
    assert frame.main([str(ply)]) == 0
    out = capsys.readouterr().out
    assert "scale: null  # UNKNOWN" in out


def test_the_printed_block_is_a_splat_block_a_world_loads(tmp_path, capsys):
    ply, _ = _capture(tmp_path, (-1.5708, 0.0, 0.35), (0.0, 0.0, 1.0), 0.5)
    assert frame.main([str(ply), "--known-height", "2.4", "--file-ref", "room.ply"]) == 0
    out = capsys.readouterr().out
    block = yaml.safe_load(out[out.index("splat:") :])
    world = {"sim": {"renderer": "splat", "renderers": {"splat": block}}, "components": []}
    load_config_from_dict(world, base_dir=tmp_path)


def test_a_file_that_is_no_splat_is_said_and_exits_nonzero(tmp_path, capsys):
    (tmp_path / "mesh.ply").write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    assert frame.main([str(tmp_path / "mesh.ply")]) == 2
    assert "binary_little_endian" in capsys.readouterr().err
