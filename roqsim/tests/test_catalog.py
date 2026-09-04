# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The catalog must agree with the loader, and with what each package chose to offer.

A catalog is only worth having if a name it prints resolves, so every ref is handed back to the
resolver that would load it -- against the real registries, not a fixture.

The other direction is about *choice* rather than completeness. A ``roqsim.worlds`` entry point is a
package saying "these worlds are an interface"; several packages also ship worlds that are debugging
aids and are deliberately not registered. So what is pinned is that a REGISTERED provider's worlds
directory and the catalog agree exactly, and that an unregistered package contributes nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from importlib import import_module, metadata
from pathlib import Path

import pytest

from roqsim.catalog import get_model_details, list_models, list_worlds, main
from roqsim.manifest import manifest_license
from roqsim.models import ModelError, resolve_model
from roqsim.world import _resolve_world_ref, available_worlds, resolve_world_yaml_ref


def _rows(catalog: dict, kind: str | None = None) -> list[dict]:
    items = [i for i in catalog["items"] if "error" not in i]
    return [i for i in items if kind is None or i.get("kind") == kind]


# -- forward: everything listed resolves -----------------------------------------------------


def test_every_listed_model_resolves():
    rows = _rows(list_models())
    assert rows, "no models at all -- the provider registry is not being read"
    for row in rows:
        assert Path(row["path"]).is_file()
        # The ref is the product: a catalog whose names the resolver rejects is worse than none.
        assert Path(resolve_model(row["ref"]).path) == Path(row["path"])


def test_every_listed_world_resolves_the_way_its_row_says_it_is_used():
    rows = _rows(list_worlds())
    kinds = {row["kind"] for row in rows}
    assert {"builtin", "world"} <= kinds
    for row in rows:
        if row["kind"] == "builtin":
            assert row["name"] in available_worlds()
        elif row["kind"] == "world":
            # `roqsim sim <ref>` and `extends: <ref>` both go through this one.
            assert resolve_world_yaml_ref(row["ref"]) == row["path"]
        else:  # scene: a sim.world target, not runnable on its own
            assert _resolve_world_ref(row["ref"]) == row["path"]


def test_a_worlds_row_carries_the_line_that_runs_it():
    world = next(r for r in _rows(list_worlds(), "world"))
    assert world["use"] == f"roqsim sim {world['ref']}"
    builtin = next(r for r in _rows(list_worlds(), "builtin"))
    assert builtin["use"] == "sim: {world: empty_room}"


# -- what registration means -------------------------------------------------------------------


def _installed_roqsim_packages() -> list[str]:
    names = set()
    for dist in metadata.distributions():
        name = (dist.metadata["Name"] or "").replace("-", "_")
        if name.startswith("roqsim"):
            names.add(name)
    return sorted(names)


def test_a_world_is_listed_exactly_when_its_package_offers_it_as_an_interface():
    """Registration is a CHOICE, and the catalog reports the choice rather than the filesystem.

    A ``roqsim.worlds`` entry point is a package saying "these worlds are an interface": nameable by
    `roqsim sim <ref>` and by `extends:`, and answerable for. Several packages also ship worlds that
    are debugging aids -- a rig that exercises one sensor, a scene that reproduces one bug -- and
    those are deliberately NOT registered: they are run by path, and a catalog that advertised them
    would be inviting a world to `extends:` something nobody maintains as an interface.

    So the rule this pins is the one that is true in both directions for a REGISTERED provider: its
    worlds directory and the catalog agree exactly. An unregistered package contributes nothing,
    which is what its author intended.
    """
    from roqsim.world import _world_entry_points

    listed: dict[str, set[str]] = {}
    for row in _rows(list_worlds(), "world"):
        listed.setdefault(row["provider"], set()).add(Path(row["path"]).name)

    providers = {
        ep.name: Path(import_module(ep.load().__name__).__file__).parent
        for ep in _world_entry_points()
    }
    assert providers, "no world providers installed -- this test would be vacuous"
    for provider, package_dir in providers.items():
        worlds = package_dir / "worlds"
        if not worlds.is_dir():
            continue
        on_disk = {p.name for p in worlds.glob("*.yaml")}
        assert listed.get(provider, set()) == on_disk, (
            f"provider {provider!r} registered its worlds, so every YAML it ships must be in the "
            f"catalog and nothing else may be"
        )


def test_an_unregistered_packages_worlds_are_absent_by_design():
    """The debug worlds: shipped, runnable by path, and not part of any package's interface."""
    listed = {Path(row["path"]).resolve() for row in _rows(list_worlds()) if row.get("path")}
    unregistered = 0
    for package in _installed_roqsim_packages():
        try:
            module = import_module(package)
        except ImportError:
            continue
        if hasattr(module, "WORLDS_DIR"):
            continue  # a provider; covered by the test above
        for path in sorted((Path(module.__file__).parent / "worlds").glob("*.yaml")):
            assert path.resolve() not in listed
            unregistered += 1
    if not unregistered:
        pytest.skip("no unregistered world YAMLs installed here")


# -- what a row says -------------------------------------------------------------------------


def test_a_model_row_names_the_components_its_manifest_brings():
    """ "What do I get when I spawn this" is not visible from the MJCF -- it is the manifest."""
    pytest.importorskip("roqsim_mobile", reason="the turtlebot4 manifest lives in roqsim_mobile")
    row = next(r for r in _rows(list_models()) if r["name"] == "turtlebot4")
    assert {"diff_drive", "lidar"} <= set(row["components"])
    assert row["provenance"], "a vendored model ships its licence beside it"


def test_a_model_sharing_its_directory_with_several_licences_names_its_own():
    """Attribution cannot be left to "the file next to it" where several vendors' terms sit there.

    A flat provider keeps every model and every vendored licence in one directory, so a model there
    says which one covers it (``license:`` in its manifest). Without that the catalog reports one
    vendor's terms for another vendor's robot, which is a wrong answer nobody would notice.
    """
    undeclared = []
    for row in _rows(list_models()):
        model_file = Path(row["path"])
        if manifest_license(model_file):
            continue
        sidecars = [
            p for p in model_file.parent.glob("*") if p.is_file() and "LICENSE" in p.name.upper()
        ]
        if len(sidecars) > 1 and not any(
            model_file.stem.lower() in p.name.lower() for p in sidecars
        ):
            undeclared.append(row["ref"])
    assert not undeclared, (
        "these models share a directory with several licences and name none, so nothing can say "
        f"which one covers them: {undeclared}"
    )


def test_model_details_add_the_config_a_spawn_actually_injects():
    pytest.importorskip("roqsim_mobile", reason="the turtlebot4 manifest lives in roqsim_mobile")
    detail = get_model_details("roqsim_mobile:turtlebot4")
    assert "error" not in detail
    lidar = next(c for c in detail["components"] if "lidar" in c)
    # The manifest's own config, not just the plugin's name: this is where a TurtleBot 4's scan gets
    # its 360 rays and the frame name the robot's URDF uses, and a spawn injects it verbatim.
    assert lidar["lidar"]["frame_id"] == "rplidar_link"
    # A component whose entry is bare is listed too (`diff_drive: {}` -- its geometry is in the
    # plugin's defaults), because what a spawn injects is the entry, empty or not.
    assert any("diff_drive" in c for c in detail["components"])
    # Where meshes are searched, in order: the answer to a model that loads without its geometry.
    assert detail["mesh_dirs"] and all(Path(d).is_dir() for d in detail["mesh_dirs"][:1])


def test_details_for_an_unknown_model_is_an_error_not_an_exception():
    assert "error" in get_model_details("not_a_model_xyz")
    with pytest.raises(ModelError):
        resolve_model("not_a_model_xyz")  # the difference this wrapper exists to make


# -- the CLI ---------------------------------------------------------------------------------


def test_the_cli_prints_json_and_refs(capsys):
    assert main(["models"]) == 0
    assert json.loads(capsys.readouterr().out)["items"]
    assert main(["worlds", "--refs"]) == 0
    refs = capsys.readouterr().out.split()
    assert "empty_room" in refs and all(":" in r or r == "empty_room" for r in refs)


def test_the_cli_exits_nonzero_for_an_unknown_model(capsys):
    assert main(["model", "not_a_model_xyz"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_it_runs_as_a_module_inside_a_container(tmp_path):
    """`python -m roqsim.catalog` is the form a caller on the host runs over `docker exec`."""
    out = subprocess.run(
        [sys.executable, "-m", "roqsim.catalog", "models"],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    assert json.loads(out.stdout)["items"]
