# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A 3DGS .ply is read by property name, whatever order its exporter wrote."""

from __future__ import annotations

import numpy as np
import pytest

from roqsim_render_splat.ply import Splat, SplatFileError, read_header, read_splat, write_splat


def _splat(n: int = 5, degree: int = 3) -> Splat:
    rng = np.random.default_rng(1)
    k = (degree + 1) ** 2
    return Splat(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        quats=rng.normal(size=(n, 4)).astype(np.float32),
        log_scales=rng.normal(size=(n, 3)).astype(np.float32),
        opacity_logits=rng.normal(size=n).astype(np.float32),
        sh=rng.normal(size=(n, k, 3)).astype(np.float32),
        sh_degree=degree,
    )


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_a_written_splat_reads_back_exactly(tmp_path, degree):
    splat = _splat(degree=degree)
    back = read_splat(write_splat(tmp_path / "s.ply", splat))
    assert back.sh_degree == degree
    for field in ("means", "quats", "log_scales", "opacity_logits", "sh"):
        np.testing.assert_array_equal(getattr(back, field), getattr(splat, field))


def test_the_rest_coefficients_are_channel_major(tmp_path):
    """``f_rest_0..14`` are red's degree 1..3 coefficients, ``f_rest_15..29`` green's."""
    splat = _splat(n=1)
    path = write_splat(tmp_path / "s.ply", splat)
    header = read_header(path)
    rec = np.fromfile(path, dtype=header.dtype, offset=header.header_bytes)
    assert rec["f_rest_0"][0] == splat.sh[0, 1, 0]  # red, first coefficient of degree 1
    assert rec["f_rest_15"][0] == splat.sh[0, 1, 1]  # green, the same coefficient
    assert rec["f_rest_14"][0] == splat.sh[0, 15, 0]  # red, the last of degree 3


def test_properties_in_another_order_and_without_normals_read_the_same(tmp_path):
    """The order an exporter wrote, and whether it wrote the unused normals, do not matter."""
    splat = _splat(degree=1)
    path = write_splat(tmp_path / "s.ply", splat)
    header = read_header(path)
    records = np.fromfile(path, dtype=header.dtype, offset=header.header_bytes)
    names = [n for n, _ in header.properties if n not in ("nx", "ny", "nz")][::-1]
    reordered = np.zeros(len(records), dtype=[(n, "<f4") for n in names])
    for n in names:
        reordered[n] = records[n]
    other = tmp_path / "other.ply"
    text = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(records)}\n"
    text += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    other.write_bytes(text.encode() + reordered.tobytes())
    back = read_splat(other)
    np.testing.assert_array_equal(back.sh, splat.sh)
    np.testing.assert_array_equal(back.quats, splat.quats)


def test_a_file_cut_short_is_refused_by_its_header(tmp_path):
    splat = _splat(8, 0)
    write_splat(tmp_path / "room.ply", splat)
    data = (tmp_path / "room.ply").read_bytes()
    (tmp_path / "cut.ply").write_bytes(data[: len(data) // 2])
    with pytest.raises(SplatFileError, match="cut short: its header promises 8 Gaussians"):
        read_header(tmp_path / "cut.ply")
    with pytest.raises(SplatFileError, match="cut short"):
        read_splat(tmp_path / "cut.ply")


def test_a_file_that_is_no_ply_says_so(tmp_path):
    (tmp_path / "room.ply").write_bytes(b"glTF....")
    with pytest.raises(SplatFileError, match="is not a PLY file"):
        read_header(tmp_path / "room.ply")


def test_an_odd_count_of_rest_coefficients_is_named(tmp_path):
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "f_rest_0", "opacity"]
    names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    text = "ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
    text += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    (tmp_path / "s.ply").write_text(text)
    with pytest.raises(SplatFileError, match="1 f_rest_\\* properties; .* have 9, 24 or 45"):
        read_header(tmp_path / "s.ply")
