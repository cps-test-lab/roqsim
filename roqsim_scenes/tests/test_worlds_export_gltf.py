"""Every bundled world exports as a glTF file the Khronos glTF Validator finds nothing wrong with.

Run by `make test-gltf`, which installs the pinned validator (tools/gltf_validator) and requires it.
Under `make test` this skips and says so: exporting the building-sized worlds is slow, and the
validator is a Node.js program `make test` does not require.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import roqsim_scenes
from roqsim import exit_status, export_gltf

_VALIDATOR = Path(__file__).resolve().parents[2] / "tools" / "gltf_validator"
_WORLDS = sorted((Path(roqsim_scenes.__file__).parent / "worlds").glob("*.yaml"))


def _khronos(paths) -> dict:
    if os.environ.get("ROQSIM_GLTF_VALIDATOR") != "required":
        pytest.skip("every bundled world through the Khronos glTF Validator: `make test-gltf`")
    if shutil.which("node") is None or not (_VALIDATOR / "node_modules").is_dir():
        pytest.fail(
            "the Khronos glTF Validator is not installed: `make test-gltf` installs it "
            f"({_VALIDATOR / 'package.json'}), and needs Node.js"
        )
    run = subprocess.run(
        ["node", str(_VALIDATOR / "validate.mjs"), *map(str, paths)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.stdout, run.stderr
    return json.loads(run.stdout)


def test_there_are_worlds_to_export():
    assert {w.stem for w in _WORLDS} >= {"depot", "tb3_world"}


@pytest.mark.parametrize("world", _WORLDS, ids=lambda w: w.stem)
def test_khronos_validator_finds_nothing_wrong_with_a_bundled_world(tmp_path, world):
    if os.environ.get("ROQSIM_GLTF_VALIDATOR") != "required":
        pytest.skip("every bundled world through the Khronos glTF Validator: `make test-gltf`")
    out = tmp_path / f"{world.stem}.glb"
    assert export_gltf.main(["--world", str(world), "--out", str(out)]) == exit_status.OK
    # One file a phone loads, at the defaults; a ceiling with headroom, not a measurement.
    assert out.stat().st_size < 20 * 2**20
    for path, report in _khronos([out]).items():
        assert report["errors"] == 0 and report["warnings"] == 0, (path, report["messages"])
