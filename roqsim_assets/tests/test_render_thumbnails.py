"""A robot's thumbnail shows the devices its manifest mounts.

A device a manifest mounts (a ``spawn_sensor`` lidar or camera) is not in the robot's own MJCF, so a
thumbnail rendered from that MJCF alone leaves it out. These tests compile the scene a thumbnail is
rendered from -- no GL needed -- and look for each mounted device's bodies in it.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from roqsim import models as M
from roqsim.config import parse_plugin_entry
from roqsim.manifest import load_manifest
from roqsim.registry import resolve_plugin
from roqsim_assets.cli.render_thumbnails import model_scene, mounts_devices


def _model_refs():
    for name, models_dir, _mesh, _tex in M.providers():
        models_dir = Path(models_dir)
        stems = [p.stem for p in models_dir.glob("*.xml")]
        stems += [p.name for p in models_dir.iterdir() if (p / f"{p.name}.xml").is_file()]
        for stem in sorted(set(stems)):
            yield f"{name}:{stem}"


def _mounted_devices(model_file: Path) -> list[tuple[str, str]]:
    """``(label, device model)`` for each entry of the manifest that mounts a device."""
    out = []
    for entry in load_manifest(model_file):
        spec = parse_plugin_entry(entry, "manifest plugin")
        if resolve_plugin(spec.ref).provides_entity:
            out.append((spec.label, spec.config["model"]))
    return out


ROBOTS = sorted({ref for ref in _model_refs() if _mounted_devices(M.resolve_model(ref).path)})


def test_a_prop_is_rendered_from_its_own_mjcf():
    assert not mounts_devices(M.resolve_model("roqsim_assets:office_table").path)


@pytest.mark.skipif(not ROBOTS, reason="no installed model mounts a device")
@pytest.mark.parametrize("ref", ROBOTS)
def test_the_thumbnail_scene_carries_every_mounted_device(ref):
    model_file = M.resolve_model(ref).path
    assert mounts_devices(model_file)
    devices = _mounted_devices(model_file)
    with model_scene(ref) as (model, _data):
        bodies = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
    for label, device in devices:
        device_spec = mujoco.MjSpec.from_file(str(M.resolve_model(device).path))
        expected = {f"{label}_{b.name}" for b in device_spec.bodies if b.name != "world"}
        assert expected, f"{device} has no bodies to look for"
        missing = expected - bodies
        assert not missing, f"{ref}: mounted {device} ({label}) missing {sorted(missing)}"
