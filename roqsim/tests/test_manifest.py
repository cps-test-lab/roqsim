"""Manifest: expand_manifest injects a model's default plugins, wired to the entity and deduped
against explicit world declarations on the producing plugin's entity key.

The model is referenced by an absolute file path (resolve_model's filesystem form), so these tests
exercise the manifest logic in isolation without registering a roqsim.models provider -- the
``<model>.manifest.yaml`` is read from beside the resolved model file.
"""

from __future__ import annotations

import textwrap

from roqsim.config import PluginSpec
from roqsim.manifest import expand_manifest, load_manifest


def _write_model(models_dir, model, manifest_body=None):
    """Create a stub model file (+ an optional manifest beside it); return its path as the model ref."""
    model_file = models_dir / f"{model}.xml"
    model_file.write_text("<mujoco/>")
    if manifest_body is not None:
        (models_dir / f"{model}.manifest.yaml").write_text(textwrap.dedent(manifest_body))
    return str(model_file)


def _spawn(model=None, name=None, **cfg):
    if model is not None:
        cfg["model"] = model
    if name is not None:
        cfg["name"] = name
    return PluginSpec(ref="spawn_arm", name=None, config=cfg)


def _expand(spec, world, target_key="arm"):
    return expand_manifest(spec, world, target_key=target_key, default_name="arm")


ARM_MANIFEST = """
    plugins:
      - arm_controller: {}
"""


def test_injects_and_wires_entity(tmp_path):
    model = _write_model(tmp_path, "ur10e", ARM_MANIFEST)
    spec = _spawn(model, "ur10e")
    out = _expand(spec, [spec])
    assert [(s.ref, s.config) for s in out] == [("arm_controller", {"arm": "ur10e", "prefix": ""})]


def test_injects_spawn_prefix(tmp_path):
    # Injected plugins inherit the spawn's prefix so a build-time plugin can form prefixed body
    # names; an explicit prefix in the manifest entry is preserved (setdefault).
    model = _write_model(
        tmp_path,
        "ur10e",
        """
        plugins:
          - fiducial_marker:
              attach_to: wrist_3_link
          - keep_own_prefix:
              prefix: custom_
    """,
    )
    spec = _spawn(model, "ur10e", prefix="ur10e_")
    by_ref = {s.ref: s.config for s in _expand(spec, [spec])}
    assert by_ref["fiducial_marker"]["prefix"] == "ur10e_"
    assert by_ref["keep_own_prefix"]["prefix"] == "custom_"


def test_world_entry_overrides_manifest(tmp_path):
    model = _write_model(tmp_path, "ur10e", ARM_MANIFEST)
    spec = _spawn(model, "ur10e")
    world_ctrl = PluginSpec("arm_controller", None, {"arm": "ur10e", "test_target": [0.0]})
    out = _expand(spec, [spec, world_ctrl])
    assert out == []  # world already declares arm_controller for ur10e -> injected default skipped


def test_world_entry_merges_manifest_defaults(tmp_path):
    """A PARTIAL override keeps the rest of the model's defaults.

    The bug this pins: the world declaring a plugin used to make expand_manifest skip the manifest
    entry entirely, so `diff_drive: {robot, test_cmd}` silently dropped the model's wheel geometry and
    actuator names and fell back to the plugin's own (TurtleBot) defaults -- which then failed to
    resolve against the husky's MJCF. roqsim_mobile's own husky_ros2.yaml crashed on exactly this,
    while docs/plugins.rst documented the merge ("add test_cmd, or change lidar rays").
    """
    model = _write_model(
        tmp_path,
        "husky",
        """
        plugins:
          - diff_drive:
              wheel_radius: 0.17775
              slip_factor: 3.0
              left_actuators: [front_left_wheel_motor, rear_left_wheel_motor]
    """,
    )
    spec = _spawn(model, "husky")
    world_dd = PluginSpec("diff_drive", None, {"arm": "husky", "test_cmd": [0.5, 0.4]})
    out = _expand(spec, [spec, world_dd])

    assert out == []  # still not injected: the world's entry is the one that runs
    # ...but it now carries the model's description instead of the plugin's generic defaults.
    assert world_dd.config["wheel_radius"] == 0.17775
    assert world_dd.config["slip_factor"] == 3.0
    assert world_dd.config["left_actuators"] == ["front_left_wheel_motor", "rear_left_wheel_motor"]
    assert world_dd.config["test_cmd"] == [0.5, 0.4]  # the world's own value survives


def test_world_value_wins_over_manifest(tmp_path):
    """Merge fills gaps; it never overwrites what the world actually said (docs: "your entry wins")."""
    model = _write_model(
        tmp_path,
        "turtlebot4",
        """
        plugins:
          - lidar:
              site: lidar
              rays: 640
    """,
    )
    spec = PluginSpec("spawn_robot", None, {"model": model, "name": "robot"})
    world_lidar = PluginSpec("lidar", None, {"robot": "robot", "rays": 1440})
    out = expand_manifest(spec, [spec, world_lidar], target_key="robot", default_name="robot")

    assert out == []
    assert world_lidar.config["rays"] == 1440  # the world changed it; the manifest must not win
    assert world_lidar.config["site"] == "lidar"  # untouched key still filled from the manifest


def test_two_entities_do_not_collide(tmp_path):
    # The bug this fixes: two arms must each get their own controller, not have one deduped away.
    ur_model = _write_model(tmp_path, "ur10e", ARM_MANIFEST)
    pa_model = _write_model(tmp_path, "panda", ARM_MANIFEST)
    ur = _spawn(ur_model, "ur10e")
    pa = _spawn(pa_model, "panda")
    world = [ur, pa]
    ur_out = _expand(ur, world)
    pa_out = _expand(pa, world)
    assert [s.config["arm"] for s in ur_out] == ["ur10e"]
    assert [s.config["arm"] for s in pa_out] == ["panda"]


def test_default_plugins_false_and_missing_manifest(tmp_path):
    model = _write_model(tmp_path, "ur10e", ARM_MANIFEST)
    off = _spawn(model, "ur10e", default_plugins=False)
    assert _expand(off, [off]) == []
    missing = _spawn(_write_model(tmp_path, "nomanifest"), "x")  # model file, no manifest beside it
    assert _expand(missing, [missing]) == []
    no_model = _spawn(name="x")
    assert _expand(no_model, [no_model]) == []


def test_target_key_selects_wiring(tmp_path):
    model = _write_model(
        tmp_path,
        "turtlebot4",
        """
        plugins:
          - lidar:
              site: lidar
    """,
    )
    spec = PluginSpec("spawn_robot", None, {"model": model, "name": "robot"})
    out = expand_manifest(spec, [spec], target_key="robot", default_name="robot")
    assert out[0].config == {"site": "lidar", "robot": "robot", "prefix": ""}


def test_load_manifest_missing_returns_empty(tmp_path):
    assert load_manifest(tmp_path / "nope.xml") == []
