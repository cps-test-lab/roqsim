"""`roqsim assets reduce-mesh` fails with a status a caller can branch on, not Blender's 0."""

from __future__ import annotations

import shutil

import pytest

from roqsim_assets.cli import reduce_mesh


def test_a_missing_input_is_refused_before_blender_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        reduce_mesh, "blender_exe", lambda _path: pytest.fail("Blender was started")
    )
    with pytest.raises(FileNotFoundError, match="no such mesh"):
        reduce_mesh.main([str(tmp_path / "nope.glb"), str(tmp_path / "out.obj")])


@pytest.mark.skipif(shutil.which("blender") is None, reason="needs Blender on PATH")
def test_a_script_error_inside_blender_is_not_exit_0(tmp_path):
    broken = tmp_path / "broken.glb"
    broken.write_text("not glTF\n")
    with pytest.raises(SystemExit) as exc:
        reduce_mesh.main([str(broken), str(tmp_path / "out.obj")])
    assert exc.value.code != 0
