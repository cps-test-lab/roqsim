"""The one-folder-per-model layout holds, and every shipped model's meshes actually resolve.

Three failures this catches, none of which the per-model drive tests would:

* a model whose MJCF moved into ``models/<name>/`` but whose ``<compiler meshdir=...>`` still points at
  the old shared ``models/meshes/`` -- MuJoCo does not error on an unresolvable mesh path, it compiles
  the model with the reference as given, so the robot silently loses its geometry;
* a mesh (or licence) that no ``[tool.setuptools.package-data]`` glob matches. That installs cleanly
  from an editable checkout and fails at RUN time inside a campaign container, where the checkout is
  not there to paper over it -- the reason those globs are patterns rather than an enumeration;
* a vendor mesh redistributed without the licence sidecar that permits it.

Named ``test_model_layout`` rather than the siblings' ``test_layout`` because pytest cannot import two
same-named test modules in one run: ``roqsim_sensors`` and ``roqsim_manipulation_assets`` both ship a
``tests/test_model_layout.py`` and already collide when the whole tree is collected at once.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import mujoco
import pytest

from roqsim.models import apply_assets, resolve_model
from roqsim_mobile.models import MODELS_DIR

MODEL_NAMES = sorted(
    p.name for p in MODELS_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.xml").is_file()
)
# `floor` is a bare ground plane, not a platform: no vendor geometry to licence, no controller to
# declare in a manifest, no port to log. Every other folder here is a robot and must carry all three.
ROBOTS = [n for n in MODEL_NAMES if n != "floor"]


def test_every_model_is_a_folder():
    # A flat `models/<name>.xml` still resolves, so a stray one would work and quietly split the
    # layout in two -- with its meshes in whichever dir the author happened to pick.
    assert not list(MODELS_DIR.glob("*.xml"))
    assert not (MODELS_DIR / "meshes").exists(), "the shared meshes/ dir is per-model now"
    assert set(ROBOTS) == {
        "clearpath_jackal", "husky_a200", "panther", "raspimouse", "ridgeback", "rosbot",
        "oomwoo_one", "turtlebot3_waffle", "turtlebot4", "warthog",
    }


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_resolves_with_all_meshes(name):
    asset = resolve_model(f"roqsim_mobile:{name}")
    assert asset.path == MODELS_DIR / name / f"{name}.xml"
    spec = mujoco.MjSpec.from_file(str(asset.path))
    apply_assets(spec, asset)
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        # apply_assets rewrites a resolved ref to an absolute path; anything still relative was not
        # found in any searched dir.
        assert Path(mesh.file).is_absolute() and Path(mesh.file).exists(), (
            f"{name}: unresolved mesh {mesh.file!r}"
        )


@pytest.mark.parametrize("name", ROBOTS)
def test_robot_folder_ships_manifest_and_licence(name):
    folder = MODELS_DIR / name
    assert (folder / f"{name}.manifest.yaml").is_file()
    assert [p for p in folder.iterdir() if p.is_file() and "LICENSE" in p.name.upper()], (
        f"{name}: vendor geometry without a licence sidecar is not redistributable"
    )




def test_package_data_globs_cover_every_shipped_file():
    tomllib = pytest.importorskip("tomllib")  # stdlib from 3.11; package supports 3.10
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    globs = data["tool"]["setuptools"]["package-data"]["roqsim_mobile"]
    pkg = root / "src" / "roqsim_mobile"
    missing = [
        str(p.relative_to(pkg))
        for p in MODELS_DIR.rglob("*")
        if p.is_file()
        # __init__.py is a module (and .pyc build litter); thumbnails are docs-only.
        and p.suffix not in (".py", ".pyc", ".png")
        and not any(fnmatch.fnmatch(str(p.relative_to(pkg)), g) for g in globs)
    ]
    assert not missing, f"not covered by package-data (would be absent from the wheel): {missing}"
