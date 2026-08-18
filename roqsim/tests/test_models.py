"""Model discovery: resolve_model finds bundled models across packages, by short name, by
package-qualified ``<package>:<model>`` ref, and by filesystem path -- each carrying its own
asset dirs. The cross-package registry is what lets a spawn plugin place a model bundled in a
different package (e.g. spawning a model shipped by one package via another package's spawn plugin)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from roqsim.models import ModelAsset, ModelError, apply_assets, providers, resolve_model


def test_providers_registered():
    names = {name for name, *_ in providers()}
    # Every models package that ships a provider entry point should appear.
    assert {
        "roqsim_manipulation_assets",
        "roqsim_sensors",
        "roqsim_mobile",
        "roqsim_assets",
    } <= names


def test_short_name_resolves_across_packages():
    a = resolve_model("ur10e")
    assert a.path.name == "ur10e.xml"
    # Asset dirs follow the model to its own package, never the caller's. That package is laid out one
    # folder per model, so the dirs are its models root and the model's own folder; the `meshes/`
    # subdir inside that folder comes from the MJCF's own `meshdir` and is prepended by apply_assets.
    assert all("roqsim_manipulation_assets" in str(d) for d in a.meshdirs)
    assert a.path.parent in a.meshdirs
    assert isinstance(a, ModelAsset)


def test_package_qualified_ref():
    a = resolve_model("roqsim_manipulation_assets:ur10e")
    assert a.path.name == "ur10e.xml"
    assert "roqsim_manipulation_assets" in str(a.path)


def test_filename_form_resolves():
    assert resolve_model("d435.xml").path.name == "d435.xml"


def test_filesystem_path(tmp_path):
    model = tmp_path / "custom.xml"
    model.write_text("<mujoco/>")
    a = resolve_model(str(model))
    assert a.path == model
    # No sibling meshes/ dir -> meshdir falls back to the model's own directory.
    assert a.meshdir == tmp_path


def test_filesystem_path_prefers_sibling_meshes(tmp_path):
    (tmp_path / "meshes").mkdir()
    model = tmp_path / "custom.xml"
    model.write_text("<mujoco/>")
    assert resolve_model(str(model)).meshdir == tmp_path / "meshes"


def test_unknown_short_name_raises():
    with pytest.raises(ModelError, match="not found in any"):
        resolve_model("definitely_not_a_model")


def test_qualified_not_found_raises():
    with pytest.raises(ModelError, match="not found in provider"):
        resolve_model("roqsim_manipulation_assets:definitely_not_a_model")


def test_unknown_provider_raises():
    with pytest.raises(ModelError):
        resolve_model("no_such_package:ur10e")


def test_manifest_assets_borrows_provider(tmp_path):
    model = tmp_path / "myarm.xml"
    model.write_text("<mujoco/>")
    (tmp_path / "myarm.manifest.yaml").write_text("assets: roqsim_manipulation_assets\nplugins: []\n")
    a = resolve_model(str(model))
    # Meshes come from the borrowed provider, not the model's own (empty) directory.
    assert "roqsim_manipulation_assets" in str(a.meshdir)


def test_manifest_assets_unknown_provider(tmp_path):
    model = tmp_path / "myarm.xml"
    model.write_text("<mujoco/>")
    (tmp_path / "myarm.manifest.yaml").write_text("assets: no_such_provider\n")
    with pytest.raises(ModelError, match="unknown provider"):
        resolve_model(str(model))


def test_manifest_assets_multiple_providers(tmp_path):
    # A list borrows several packages' dirs (e.g. arm meshes + a camera mesh), searched in order,
    # then the model's own dir last.
    model = tmp_path / "myarm.xml"
    model.write_text("<mujoco/>")
    (tmp_path / "myarm.manifest.yaml").write_text(
        "assets: [roqsim_manipulation_assets, roqsim_sensors]\nplugins: []\n"
    )
    a = resolve_model(str(model))
    joined = " ".join(str(d) for d in a.meshdirs)
    assert "roqsim_manipulation_assets" in joined and "roqsim_sensors" in joined
    assert a.meshdirs[-1] == tmp_path  # own dir is the final fallback


def test_apply_assets_own_meshdir_beats_provider(tmp_path):
    # Regression: the Oli's ``left_hip_yaw_link.STL`` (own ``meshes/oli/``) was silently shadowed
    # by the G1's same-named Menagerie mesh in the provider's flat ``meshes/`` dir -- the spawned
    # robot wore the wrong robot's limb shells. The spec's own declared meshdir must win.
    own = tmp_path / "meshes" / "bot"
    own.mkdir(parents=True)
    (own / "part.STL").write_bytes(b"")
    provider = tmp_path / "provider_meshes"
    provider.mkdir()
    (provider / "part.STL").write_bytes(b"")  # same-named impostor
    (provider / "borrowed.STL").write_bytes(b"")  # only the provider has this one
    model = tmp_path / "bot.xml"
    model.write_text(
        '<mujoco><compiler meshdir="meshes/bot/"/><asset>'
        '<mesh name="part" file="part.STL"/><mesh name="borrowed" file="borrowed.STL"/>'
        "</asset></mujoco>"
    )
    spec = mujoco.MjSpec.from_file(str(model))
    apply_assets(spec, ModelAsset(model, (provider,), (provider,)))
    resolved = {m.name: Path(m.file) for m in spec.meshes}
    assert resolved["part"] == (own / "part.STL").resolve()  # own mesh, not the impostor
    assert resolved["borrowed"] == (provider / "borrowed.STL").resolve()  # borrowing still works
