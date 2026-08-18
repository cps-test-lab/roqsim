"""The shared textures resolve by 'roqsim_assets:<Name>' and carry their manifest properties."""

from __future__ import annotations

import pytest

from roqsim.textures import TextureError, UVScaler, resolve_texture, texture_manifest


def test_uvscaler_scales_obj_uvs(tmp_path):
    obj = tmp_path / "m.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nvt 2 4\nvt -1 3\nf 1 2 1\n")
    scaler = UVScaler()
    out = scaler.scaled(obj, 0.5, "m")
    assert out != str(obj)  # a scaled copy was written
    vts = [ln.split()[1:] for ln in open(out) if ln.startswith("vt ")]
    assert vts == [["1.000000", "2.000000"], ["-0.500000", "1.500000"]]  # UVs halved
    # verts/faces preserved
    assert any(ln.startswith("f ") for ln in open(out))


def test_uvscaler_noop_cases(tmp_path):
    scaler = UVScaler()
    # scale 1.0 -> original path
    obj = tmp_path / "m.obj"
    obj.write_text("v 0 0 0\nvt 1 1\n")
    assert scaler.scaled(obj, 1.0, "m") == str(obj)
    # UV-less OBJ -> original path (auto-projected; texrepeat handles it)
    nouv = tmp_path / "n.obj"
    nouv.write_text("v 0 0 0\nv 1 0 0\nf 1 2 1\n")
    assert scaler.scaled(nouv, 0.5, "n") == str(nouv)
    # non-OBJ (e.g. STL) -> original path
    stl = tmp_path / "s.stl"
    stl.write_text("solid\nendsolid\n")
    assert scaler.scaled(stl, 0.5, "s") == str(stl)


def test_resolve_package_qualified():
    png = resolve_texture("roqsim_assets:Concrete030")
    assert png.is_file() and png.suffix.lower() == ".png"
    manifest = texture_manifest(png)
    assert manifest.get("physical_size") and manifest.get("reflectance") is not None


def test_bare_name_is_rejected_with_hint():
    # No cross-package name search: a bare name must be qualified (avoids duplicate-name collisions).
    with pytest.raises(TextureError, match="package-qualified"):
        resolve_texture("Concrete030")


def test_unknown_provider_and_name():
    with pytest.raises(TextureError, match="not an importable package"):
        resolve_texture("no_such_pkg:Concrete030")
    with pytest.raises(TextureError, match="not found in provider"):
        resolve_texture("roqsim_assets:DoesNotExist")


def test_path_form(tmp_path):
    p = tmp_path / "custom.png"
    p.write_bytes(b"\x89PNG\r\n")
    assert resolve_texture(str(p)) == p
    with pytest.raises(TextureError, match="does not exist"):
        resolve_texture(str(tmp_path / "missing.png"))
