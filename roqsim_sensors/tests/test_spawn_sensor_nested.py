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
from roqsim.frames import add_frame_sites, parse_frames, tf_anchors
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
  - name: "{frame_id}_optical"
    parent: "{frame_id}"
    pose: {position: {z: 0.01}, orientation: {yaw: 1.5707963267948966}}
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
  - name: shell_link
    parent: base_link
    pose: {position: {x: -0.1, z: 0.2}, orientation: {yaw: 1.5707963267948966}}
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
        self.frames = parse_frames(manifest_frames(path), "carrier")
        add_frame_sites(child, self.frames, "carrier")
        spec.attach(child, prefix=self.config.get("prefix", ""), frame=spec.worldbody.add_frame())

    def configure(self, ctx: SimContext) -> None:
        prefix = self.config.get("prefix", "")
        ctx.entities.add(
            Entity(
                name=self.address,
                kind="robot",
                body=prefix + "base_link",
                meta={
                    "prefix": prefix,
                    "namespace": self.config.get("namespace", ""),
                    "frame_anchors": {
                        f.name: a
                        for f in self.frames
                        if not f.tf
                        for a in [tf_anchors(self.frames)[f.name]]
                    },
                },
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


def _carrier(tmp_path, mounts: str, frames: str = "") -> str:
    """The carrier model; *mounts* are its devices, *frames* more entries of its ``frames:``."""
    (tmp_path / "carrier.xml").write_text(CARRIER_XML)
    (tmp_path / "carrier.manifest.yaml").write_text(
        textwrap.dedent(CARRIER_FRAMES)
        + textwrap.indent(textwrap.dedent(frames), "  ")
        + "components:\n"
        + textwrap.indent(textwrap.dedent(mounts), "  ")
    )
    return str(tmp_path / "carrier.xml")


def _front(device, extra=""):
    return f"""
    - spawn_sensor: {{model: {device}, parent_frame: cover_link, pose: {{position: {{z: 0.05}}}}}}
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
    - spawn_sensor: {{model: {_device(tmp_path)}, parent_frame: shell_link, pose: {{position: {{x: 0.1}}}}}}
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
    - spawn_sensor: {{model: {device}, parent_frame: base_link, pose: {{position: {{x: -0.25, z: 0.1}}, orientation: {{yaw: 3.141592653589793}}}}{explicit}}}
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
    with pytest.raises(PluginError, match=r"would both publish frame\(s\) \['scanner_link'"):
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
        ("{model: DEVICE, pose: {position: {z: 0.05}}}", "says nowhere to hang from"),
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
    with pytest.raises(
        PluginError,
        match=r"parent_frame 'mast_link' is neither a body nor a frame of carrier\. .*"
        r"Its frames: shell_link",
    ):
        _world(_carrier(tmp_path, mounts))


def test_a_world_level_device_hangs_its_chain_off_the_world(tmp_path):
    cfg = load_config_from_dict(
        {
            "components": [
                {
                    "spawn_sensor": {
                        "model": _device(tmp_path),
                        "pose": {"position": {"z": 1.0}},
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


# A device whose vendor macro prefixes every link with its `name` parameter, as a camera
# description does: the chain is `<name>_link -> <name>_optical_frame`.
NAMED_DEVICE_MANIFEST = """
device_name: cam
frame_id: "{device_name}_optical_frame"
components:
  - lidar: {site: scan, frame_id: "{frame_id}", exclude_body: mount, emit_static_tf: false,
            rays: 4, max_range: 4.0, range_min: 0.0}
frames:
  - {name: "{device_name}_link", parent: mount}
  - {name: "{frame_id}", parent: "{device_name}_link", pose: {position: {z: 0.01}}}
"""


def _named_device(tmp_path, manifest=NAMED_DEVICE_MANIFEST) -> str:
    (tmp_path / "camdev.xml").write_text(DEVICE_XML)
    (tmp_path / "camdev.manifest.yaml").write_text(textwrap.dedent(manifest))
    return str(tmp_path / "camdev.xml")


def _mount(device, name, parent="cover_link", extra=""):
    return f"""
    - spawn_sensor: {{model: {device}, parent_frame: {parent}{extra}}}
      name: {name}
    """


def _chain(engine, owner):
    tfs = _endpoint(engine, "frames", owner).backend["ros2"]["static_tf"]
    return [(t["parent"], t["child"]) for t in tfs]


def test_device_name_defaults_to_the_vendor_macro_name_and_fills_the_chain(tmp_path):
    engine = _engine(_world(_carrier(tmp_path, _mount(_named_device(tmp_path), "head"))))
    mount = _plugin(engine, "robot.head")
    assert mount.config["device_name"] == "cam"
    assert mount.config["frame_id"] == "cam_optical_frame"  # the default, its name filled in
    assert _chain(engine, "robot.head") == [
        ("cover_link", "cam_link"),
        ("cam_link", "cam_optical_frame"),
    ]
    scan = _endpoint(engine, "scan", "robot.head")
    assert scan.backend["ros2"]["frame_id"] == "cam_optical_frame"


def test_two_mounts_of_one_device_each_name_their_own_instance(tmp_path):
    device = _named_device(tmp_path)
    mounts = _mount(device, "head", extra=", device_name: head_camera") + _mount(
        device, "chest", parent="base_link", extra=", device_name: chest_camera"
    )
    engine = _engine(_world(_carrier(tmp_path, mounts)))
    assert _chain(engine, "robot.head") == [
        ("cover_link", "head_camera_link"),
        ("head_camera_link", "head_camera_optical_frame"),
    ]
    assert _chain(engine, "robot.chest") == [
        ("base_link", "chest_camera_link"),
        ("chest_camera_link", "chest_camera_optical_frame"),
    ]


def test_two_mounts_that_share_an_intermediate_frame_are_refused(tmp_path):
    """Distinct scan frames are not enough: the chain's `<device_name>_link` would still be one name
    published from two poses."""
    device = _named_device(tmp_path)
    mounts = _mount(device, "head", extra=", frame_id: head_optical") + _mount(
        device, "chest", parent="base_link", extra=", frame_id: chest_optical"
    )
    with pytest.raises(PluginError, match=r"would both publish frame\(s\) \['cam_link'\]"):
        _world(_carrier(tmp_path, mounts))


def test_a_device_that_names_frames_after_device_name_needs_one(tmp_path):
    manifest = NAMED_DEVICE_MANIFEST.replace("device_name: cam\n", "")
    device = _named_device(tmp_path, manifest)
    with pytest.raises(PluginError, match="declares no default 'device_name'"):
        _world(_carrier(tmp_path, _mount(device, "head")))
    cfg = _world(_carrier(tmp_path, _mount(device, "head", extra=", device_name: eye")))
    mount = next(s for s in cfg.plugins if s.address == "robot.head")
    assert mount.config["frame_id"] == "eye_optical_frame"


def test_a_carrier_mount_of_a_device_without_a_frame_chain_is_refused(tmp_path):
    """No vendor link to hang from, and nothing to connect its data's frame to the carrier's tree."""
    (tmp_path / "bare.xml").write_text(DEVICE_XML)
    (tmp_path / "bare.manifest.yaml").write_text("components: []\n")
    with pytest.raises(PluginError, match=r"device '.*bare\.xml' declares no 'frames:' chain"):
        _world(_carrier(tmp_path, _mount(str(tmp_path / "bare.xml"), "cam")))
    # The same device still mounts on its own in a world.
    cfg = load_config_from_dict(
        {"components": [{"spawn_sensor": {"model": str(tmp_path / "bare.xml")}, "name": "cam"}]}
    )
    _engine(cfg)


# -- frames a device hangs from ------------------------------------------------------------------

CARRIER_PLACES = """
  - {name: front, parent: shell_link, pose: {position: {x: 0.1}}, tf: false}
  - name: rear
    parent: base_link
    pose: {position: {x: -0.25, z: 0.1}, orientation: {yaw: 3.141592653589793}}
    tf: false
"""


def _on(device, frame, label="scan_front", extra=""):
    return f"""
    - spawn_sensor: {{model: {device}, parent_frame: {frame}{extra}}}
      name: {label}
    """


def _world_with(carrier, *children, overrides=None):
    return load_config_from_dict(
        {
            "components": [
                {
                    CARRIER: {"model": carrier, "prefix": "r_", "namespace": "tb"},
                    "name": "robot",
                    "components": list(children),
                },
            ]
        },
        overrides=overrides,
    )


def _published(engine):
    """Every static transform sent: a robot's frames are the Transforms it returns, a mounted
    device's its static_tf hint."""
    out = set()
    for e in engine.ctx.interface.all():
        hint = (e.backend.get("ros2") or {}).get("static_tf")
        if hint:
            out |= {(t["parent"], t["child"]) for t in hint}
        elif e.name == "frames":
            out |= {(t.parent, t.child) for t in e.read().transforms}
    return out


def test_a_device_on_an_unpublished_frame_sits_where_its_parent_and_pose_would(tmp_path):
    device = _device(tmp_path)
    on_frame = _engine(_world(_carrier(tmp_path, _on(device, "front"), CARRIER_PLACES)))
    explicit = f"""
    - spawn_sensor: {{model: {device}, parent_frame: shell_link, pose: {{position: {{x: 0.1}}}}}}
      name: scan_front
    """
    by_pose = _engine(_world(_carrier(tmp_path, explicit, CARRIER_PLACES)))
    for a, b in zip(
        _body_pose(on_frame, "r_scan_front_mount"),
        _body_pose(by_pose, "r_scan_front_mount"),
        strict=True,
    ):
        assert np.allclose(a, b, atol=1e-12)
    spec = next(s for s in on_frame.config.plugins if s.address == "robot.scan_front")
    assert spec.config["parent_frame"] == "front" and "pose" not in spec.config
    root = _endpoint(on_frame, "frames", "robot.scan_front").backend["ros2"]["static_tf"][0]
    assert root["parent"] == "shell_link" and np.allclose(root["translation"], [0.1, 0, 0])


def test_an_unpublished_frame_is_a_site_and_never_in_tf(tmp_path):
    engine = _engine(_world(_carrier(tmp_path, _on(_device(tmp_path), "front"), CARRIER_PLACES)))
    assert mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_SITE, "r_front") >= 0
    published = _published(engine)
    assert ("shell_link", "scanner_link") in published
    assert not any("front" in pair for pair in published)


def test_a_pose_beside_parent_frame_is_an_offset_from_that_frame(tmp_path):
    device = _device(tmp_path)
    at = _engine(_world(_carrier(tmp_path, _on(device, "rear"), CARRIER_PLACES)))
    offset = _on(device, "rear", extra=", pose: {position: {x: 0.05}}")
    moved = _engine(_world(_carrier(tmp_path, offset, CARRIER_PLACES)))
    pos, mat = _body_pose(at, "r_scan_front_mount")
    assert np.allclose(pos, [-0.25, 0, 0.1], atol=1e-9)
    # `rear` faces -x, so 0.05 along its x is 0.05 further back.
    assert np.allclose(_body_pose(moved, "r_scan_front_mount")[0], pos + 0.05 * mat[:, 0])


def test_an_override_moves_a_device_by_its_pose_and_the_record_states_it(tmp_path):
    from roqsim.config import overrides_from_dotlist

    carrier = _carrier(tmp_path, _on(_device(tmp_path), "front"), CARRIER_PLACES)
    cfg = _world_with(
        carrier,
        overrides=overrides_from_dotlist(["components.robot.scan_front.pose.position.z=0.2"]),
    )
    spec = next(s for s in cfg.plugins if s.address == "robot.scan_front")
    assert spec.config["parent_frame"] == "front"
    assert spec.config["pose"] == {"position": {"z": 0.2}}
    pos, _ = _body_pose(_engine(cfg), "r_scan_front_mount")
    assert np.allclose(pos, [-0.1, 0.1, 0.4], atol=1e-9)


def test_a_world_puts_a_device_on_a_robots_frame_by_nesting_it_under_the_robot(tmp_path):
    """The carrier's own device is on `front`; a world adds a second one on `rear` with no offset."""
    device = _device(tmp_path)
    carrier = _carrier(tmp_path, _on(device, "front"), CARRIER_PLACES)
    extra = {
        "spawn_sensor": {"model": device, "parent_frame": "rear", "frame_id": "rear_link"},
        "name": "added",
    }
    pos, _ = _body_pose(_engine(_world_with(carrier, extra)), "r_added_mount")
    assert np.allclose(pos, [-0.25, 0, 0.1], atol=1e-9)


def test_a_frame_the_carrier_does_not_declare_is_refused_listing_its_frames(tmp_path):
    with pytest.raises(
        PluginError,
        match=r"parent_frame 'fron' .*Did you mean 'front'\?.*"
        r"shell_link, front \(tf: false\), rear \(tf: false\)",
    ):
        _world(_carrier(tmp_path, _on(_device(tmp_path), "fron"), CARRIER_PLACES))


def test_a_frame_whose_parent_the_carrier_lacks_is_refused(tmp_path):
    bad = """
      - {name: front, parent: nowhere_link, tf: false}
    """
    with pytest.raises(
        PluginError, match=r"frame 'front' hangs from 'nowhere_link', which is neither"
    ):
        _world(_carrier(tmp_path, _on(_device(tmp_path), "front"), bad))


def test_two_frames_of_one_name_are_refused(tmp_path):
    dup = """
      - {name: shell_link, parent: cover_link, tf: false}
    """
    with pytest.raises(PluginError, match=r"frames\[1\]: frame 'shell_link' is declared twice"):
        _world(_carrier(tmp_path, _on(_device(tmp_path), "shell_link"), dup))


@pytest.mark.parametrize(
    "extra, spelling",
    [
        (", pos: [0, 0, 1]", r"pose: \{position: \{z: 1\.0\}\}"),
        (", rpy: [0, 0, 1]", r"pose: \{orientation: \{yaw: 1\.0\}\}"),
    ],
)
def test_pos_and_rpy_are_refused_with_the_pose_they_mean(tmp_path, extra, spelling):
    with pytest.raises(PluginError, match=spelling):
        Engine(
            _world(_carrier(tmp_path, _on(_device(tmp_path), "front", extra=extra), CARRIER_PLACES))
        )


def test_tf_false_in_a_device_manifest_is_refused(tmp_path):
    manifest = DEVICE_MANIFEST.replace("parent: mount}", "parent: mount, tf: false}")
    (tmp_path / "scanner.xml").write_text(DEVICE_XML)
    (tmp_path / "scanner.manifest.yaml").write_text("frame_id: scanner_link\n" + manifest)
    with pytest.raises(Exception, match=r"a device's frames are its vendor links"):
        _engine(_world(_carrier(tmp_path, _front(str(tmp_path / "scanner.xml")))))
