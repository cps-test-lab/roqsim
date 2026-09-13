"""spawn_sensor mounted by its carrier: identity inherited, placed at a vendor frame, chain published.

Synthetic carrier and device models are written to tmp_path, and the carrier is a stand-in plugin
defined here, so nothing depends on a shipped robot or on roqsim_mobile.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.context import Entity, SimContext
from roqsim.engine import Engine
from roqsim.frames import add_frame_sites, parse_frames
from roqsim.manifest import expand_manifest, manifest_frames
from roqsim.plugin import Plugin, PluginError

DEVICE_XML = """
<mujoco>
  <worldbody>
    <body name="mount">
      <geom type="box" size="0.03 0.03 0.03"/>
      <site name="scan" pos="0 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""

# The scan site sits INSIDE the housing geom: every ray starts in the mount, so only excluding this
# device's own `mount` lets it see anything.
DEVICE_MANIFEST = """
components:
  - lidar: {site: scan, frame_id: "{frame_id}", exclude_body: mount, emit_static_tf: false,
            rays: 4, max_range: 4.0, range_min: 0.0}
frames:
  - {name: "{frame_id}", parent: mount}
  - {name: "{frame_id}_optical", parent: "{frame_id}", pos: [0, 0, 0.01], rpy: [0, 0, 1.5707963267948966]}
"""

CARRIER_XML = """
<mujoco>
  <worldbody>
    <body name="base_link">
      <geom type="box" size="0.2 0.2 0.05"/>
      <body name="cover_link" pos="0.1 0 0.1">
        <geom type="box" size="0.05 0.05 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

CARRIER_FRAMES = """
frames:
  - {name: shell_link, parent: base_link, pos: [-0.1, 0, 0.2], rpy: [0, 0, 1.5707963267948966]}
"""


class _Carrier(Plugin):
    """A robot as far as a mount can tell: a prefixed model with declared frames, and an entity."""

    provides_entity = True

    @classmethod
    def expand(cls, spec, world, base_dir):
        return expand_manifest(spec, world, base_dir=base_dir)

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        path = Path(self.config["model"])
        child = mujoco.MjSpec.from_file(str(path))
        add_frame_sites(child, parse_frames(manifest_frames(path), "carrier"), "carrier")
        spec.attach(child, prefix=self.config.get("prefix", ""), frame=spec.worldbody.add_frame())

    def configure(self, ctx: SimContext) -> None:
        prefix = self.config.get("prefix", "")
        ctx.entities.add(
            Entity(
                name=self.address,
                kind="robot",
                body=prefix + "base_link",
                meta={"prefix": prefix, "namespace": self.config.get("namespace", "")},
            )
        )


class _Wall(Plugin):
    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, pos=[2, 0, 0.5], size=[0.05, 5, 0.5]
        )


CARRIER = f"{__name__}:_Carrier"
WALL = f"{__name__}:_Wall"


def _device(tmp_path, frame_id: str | None = "scanner_link") -> str:
    """The device model; *frame_id* is its manifest's vendor default, ``None`` for a vendor with none."""
    default = f"frame_id: {frame_id}\n" if frame_id else ""
    (tmp_path / "scanner.xml").write_text(DEVICE_XML)
    (tmp_path / "scanner.manifest.yaml").write_text(default + textwrap.dedent(DEVICE_MANIFEST))
    return str(tmp_path / "scanner.xml")


def _carrier(tmp_path, mounts: str) -> str:
    (tmp_path / "carrier.xml").write_text(CARRIER_XML)
    (tmp_path / "carrier.manifest.yaml").write_text(
        textwrap.dedent(CARRIER_FRAMES)
        + "components:\n"
        + textwrap.indent(textwrap.dedent(mounts), "  ")
    )
    return str(tmp_path / "carrier.xml")


def _front(device, extra=""):
    return f"""
    - spawn_sensor: {{model: {device}, parent_frame: cover_link, pos: [0, 0, 0.05]}}
      name: scan_front{extra}
    """


def _world(carrier, overrides=None, **carrier_cfg):
    cfg = load_config_from_dict(
        {
            "components": [
                {WALL: {}},
                {
                    CARRIER: {"model": carrier, "prefix": "r_", "namespace": "tb", **carrier_cfg},
                    "name": "robot",
                },
            ]
        },
        overrides=overrides,
    )
    return cfg


def _engine(cfg) -> Engine:
    engine = Engine(cfg)
    engine.setup()
    return engine


def _plugin(engine, address):
    return next(p for p in engine.plugins if p.address == address)


def _endpoint(engine, name, owner):
    return next(e for e in engine.ctx.interface.all() if e.name == name and e.owner == owner)


def _body_pose(engine, name):
    m, d = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, name
    return d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()


def test_a_nested_device_inherits_its_carriers_prefix_and_namespace(tmp_path):
    cfg = _world(_carrier(tmp_path, _front(_device(tmp_path))))
    mount = next(s for s in cfg.plugins if s.address == "robot.scan_front")
    assert mount.config["attach_prefix"] == "r_"
    assert mount.config["prefix"] == "r_scan_front_"
    assert mount.config["frame_id"] == "scanner_link"  # the device manifest's vendor default
    engine = _engine(cfg)
    entity = engine.ctx.entities.get("robot.scan_front")
    assert entity.meta["prefix"] == "r_scan_front_" and entity.meta["namespace"] == "tb"
    scan = _endpoint(engine, "scan", "robot.scan_front")
    assert scan.namespace == "tb"
    assert scan.backend["ros2"]["topic"] == "scan"  # relative, so the namespace scopes it
    assert scan.backend["ros2"]["frame_id"] == "scanner_link"
    assert "static_tf" not in scan.backend["ros2"]  # the mount owns the chain


def test_exclude_body_mount_is_this_devices_housing_and_the_scan_sees_out(tmp_path):
    engine = _engine(_world(_carrier(tmp_path, _front(_device(tmp_path)))))
    lidar = _plugin(engine, "robot.scan_front.lidar")
    housing = mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, "r_scan_front_mount")
    assert lidar._bodyexclude == housing
    engine.reset()
    engine.step()
    # Mounted 0.1 in front of the base: the wall face at x=1.95 reads 1.85, not the housing.
    assert np.isclose(lidar.latest.ranges[0], 1.85, atol=1e-3)


def test_the_device_chain_is_published_from_its_parent_frame(tmp_path):
    engine = _engine(_world(_carrier(tmp_path, _front(_device(tmp_path)))))
    ep = _endpoint(engine, "frames", "robot.scan_front")
    assert ep.namespace == "tb"
    root, optical = ep.backend["ros2"]["static_tf"]
    assert (root["parent"], root["child"]) == ("cover_link", "scanner_link")
    assert np.allclose(root["translation"], [0, 0, 0.05])
    assert (optical["parent"], optical["child"]) == ("scanner_link", "scanner_link_optical")
    assert np.allclose(optical["translation"], [0, 0, 0.01])
    assert np.allclose(optical["rotation"], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])


def test_parent_frame_may_be_a_frame_the_carrier_declares(tmp_path):
    mounts = f"""
    - spawn_sensor: {{model: {_device(tmp_path)}, parent_frame: shell_link, pos: [0.1, 0, 0]}}
      name: scan_front
    """
    engine = _engine(_world(_carrier(tmp_path, mounts)))
    pos, mat = _body_pose(engine, "r_scan_front_mount")
    # shell_link is (-0.1, 0, 0.2) yawed 90 deg; 0.1 along ITS x is +y in the carrier's frame.
    assert np.allclose(pos, [-0.1, 0.1, 0.2], atol=1e-9)
    assert np.allclose(mat[:, 0], [0, 1, 0], atol=1e-9)
    root = _endpoint(engine, "frames", "robot.scan_front").backend["ros2"]["static_tf"][0]
    assert root["parent"] == "shell_link" and np.allclose(root["translation"], [0.1, 0, 0])


def test_parent_frame_as_a_body_places_the_mount_at_the_composed_pose(tmp_path):
    engine = _engine(_world(_carrier(tmp_path, _front(_device(tmp_path)))))
    pos, _ = _body_pose(engine, "r_scan_front_mount")
    assert np.allclose(pos, [0.1, 0, 0.15], atol=1e-9)  # cover_link (0.1, 0, 0.1) + 0.05 up


def _rear(device, frame_id=None):
    explicit = f", frame_id: {frame_id}" if frame_id else ""
    return f"""
    - spawn_sensor: {{model: {device}, parent_frame: base_link, pos: [-0.25, 0, 0.1], rpy: [0, 0, 3.141592653589793]{explicit}}}
      name: scan_rear
    """


def test_two_identical_devices_on_one_carrier_resolve_distinct_names(tmp_path):
    device = _device(tmp_path)
    mounts = _front(device) + _rear(device, frame_id="rear_link")
    engine = _engine(_world(_carrier(tmp_path, mounts)))
    front = _plugin(engine, "robot.scan_front.lidar")
    rear = _plugin(engine, "robot.scan_rear.lidar")
    assert front._bodyexclude != rear._bodyexclude and front._site_id != rear._site_id
    frames = {
        e.owner: e.backend["ros2"]["static_tf"][0]["child"]
        for e in engine.ctx.interface.all()
        if e.name == "frames"
    }
    # The rear mount's explicit frame_id wins over the default the front one keeps.
    assert frames == {"robot.scan_front": "scanner_link", "robot.scan_rear": "rear_link"}
    assert rear.config["frame_id"] == "rear_link"


def test_two_mounts_on_one_carrier_left_on_the_same_default_are_refused(tmp_path):
    """They share the carrier's namespace, so one frame name would be published from two poses."""
    device = _device(tmp_path)
    with pytest.raises(PluginError, match="both use frame_id 'scanner_link'"):
        _world(_carrier(tmp_path, _front(device) + _rear(device)))


def test_a_device_whose_vendor_names_no_default_needs_a_frame_id_on_the_mount(tmp_path):
    device = _device(tmp_path, frame_id=None)
    with pytest.raises(PluginError, match="declares no default 'frame_id'"):
        _world(_carrier(tmp_path, _front(device)))
    # Named on the mount, the same device mounts.
    cfg = _world(_carrier(tmp_path, _rear(device, frame_id="rear_link")))
    lidar = next(s for s in cfg.plugins if s.address == "robot.scan_rear.lidar")
    assert lidar.config["frame_id"] == "rear_link"


def test_a_carrier_manifest_override_and_topics_win_over_the_device(tmp_path):
    mounts = _front(
        _device(tmp_path),
        extra="""
      components:
        - lidar: {rays: 8, topics: {scan: /front/scan}}""",
    )
    engine = _engine(_world(_carrier(tmp_path, mounts)))
    lidar = _plugin(engine, "robot.scan_front.lidar")
    assert lidar.num_rays == 8
    assert lidar.config["max_range"] == 4.0  # still the device's
    assert _endpoint(engine, "scan", "robot.scan_front").backend["ros2"]["topic"] == "/front/scan"


def test_a_world_declared_mount_sets_frame_id_before_the_device_expands(tmp_path):
    carrier = _carrier(tmp_path, _front(_device(tmp_path)))
    cfg = load_config_from_dict(
        {
            "components": [
                {
                    CARRIER: {"model": carrier, "prefix": "r_", "namespace": "tb"},
                    "name": "robot",
                    "components": [{"spawn_sensor": {"frame_id": "laser"}, "name": "scan_front"}],
                }
            ]
        },
        overrides={"components": {"robot": {"scan_front": {"namespace": "other"}}}},
    )
    engine = _engine(cfg)
    scan = _endpoint(engine, "scan", "robot.scan_front")
    # frame_id reached the device's templated components; namespace is read at configure, so an
    # override of it on the mount is fine either way.
    assert scan.namespace == "other" and scan.backend["ros2"]["frame_id"] == "laser"
    root = _endpoint(engine, "frames", "robot.scan_front").backend["ros2"]["static_tf"][0]
    assert root["child"] == "laser"


def test_overriding_an_expansion_input_of_an_injected_mount_is_refused(tmp_path):
    """Too late to reach the device's components: the scan would keep the old frame name while the
    mount published the new one."""
    with pytest.raises(PluginError, match="Declare it in the world instead"):
        _world(
            _carrier(tmp_path, _front(_device(tmp_path))),
            overrides={"components": {"robot": {"scan_front": {"frame_id": "laser"}}}},
        )


@pytest.mark.parametrize(
    "mount, match",
    [
        ("{model: DEVICE, pos: [0, 0, 0.05]}", "says nowhere to hang from"),
        ("{model: DEVICE, parent_frame: cover_link, motion: driven}", "welded"),
        ("{model: DEVICE, parent_frame: cover_link, attach_to: cover_link}", "both say where"),
    ],
)
def test_a_nested_mount_that_cannot_ride_its_carrier_is_refused(tmp_path, mount, match):
    mounts = f"""
    - spawn_sensor: {mount.replace("DEVICE", _device(tmp_path))}
      name: scan_front
    """
    with pytest.raises(PluginError, match=match):
        Engine(_world(_carrier(tmp_path, mounts)))


def test_a_parent_frame_the_carrier_does_not_have_is_named(tmp_path):
    mounts = f"""
    - spawn_sensor: {{model: {_device(tmp_path)}, parent_frame: mast_link}}
      name: scan_front
    """
    with pytest.raises(Exception, match="r_mast_link"):
        _engine(_world(_carrier(tmp_path, mounts)))


def test_a_world_level_device_hangs_its_chain_off_the_world(tmp_path):
    cfg = load_config_from_dict(
        {
            "components": [
                {
                    "spawn_sensor": {
                        "model": _device(tmp_path),
                        "pos": [0, 0, 1.0],
                        "frame_id": "laser",
                    },
                    "name": "tripod",
                }
            ]
        }
    )
    engine = _engine(cfg)
    root = _endpoint(engine, "frames", "tripod").backend["ros2"]["static_tf"][0]
    assert (root["parent"], root["child"]) == ("world", "laser")
    assert np.allclose(root["translation"], [0, 0, 1.0])
