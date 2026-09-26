"""roqsim.frames: vendor fixed links declared beside a model, built as sites, read back as transforms."""

from __future__ import annotations

import textwrap

import mujoco
import numpy as np
import pytest

from roqsim.config import PluginSpec
from roqsim.frames import (
    FRAME_SITE_GROUP,
    add_frame_sites,
    parse_frames,
    static_tf_endpoint,
    static_transforms,
    substitute,
)
from roqsim.manifest import (
    expand_manifest,
    manifest_device_name,
    manifest_frame_id,
    manifest_frames,
)
from roqsim.plugin import PluginError


def _model():
    spec = mujoco.MjSpec()
    base = spec.worldbody.add_body(name="base_link", pos=[1.0, 0.0, 0.0])
    base.add_geom(size=[0.1, 0, 0])
    return spec


CHAIN = [
    {"name": "shell_link", "parent": "base_link", "pos": [0, 0, 0.1]},
    {
        "name": "rplidar_link",
        "parent": "shell_link",
        "pos": [-0.04, 0, 0.1],
        "rpy": [0, 0, np.pi / 2],
    },
]


def test_a_chain_is_built_on_its_body_and_read_back_per_link():
    spec = _model()
    add_frame_sites(spec, parse_frames(CHAIN, "t"), "t")
    # Both frames cost a site on base_link and no body: a flattened link stays flattened.
    assert [s.name for s in spec.body("base_link").sites] == ["shell_link", "rplidar_link"]
    assert all(s.group == FRAME_SITE_GROUP for s in spec.sites)
    model = spec.compile()
    shell, lidar, direct = static_transforms(
        model,
        [
            ("base_link", "base_link", "shell_link", "shell_link"),
            ("shell_link", "shell_link", "rplidar_link", "rplidar_link"),
            ("base_link", "base_link", "rplidar_link", "rplidar_link"),
        ],
        "t",
    )
    assert shell["parent"] == "base_link" and shell["child"] == "shell_link"
    assert np.allclose(shell["translation"], [0, 0, 0.1])
    assert np.allclose(lidar["translation"], [-0.04, 0, 0.1])
    assert np.allclose(lidar["rotation"], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    # Composed along the chain, not merely stored per link.
    assert np.allclose(direct["translation"], [-0.04, 0, 0.2])


def test_a_rotated_parent_frame_rotates_what_hangs_from_it():
    spec = _model()
    frames = [
        {"name": "flipped", "parent": "base_link", "rpy": [np.pi, 0, 0]},
        {"name": "below", "parent": "flipped", "pos": [0, 0, 0.1]},
    ]
    add_frame_sites(spec, parse_frames(frames, "t"), "t")
    (link,) = static_transforms(spec.compile(), [("base_link", "base_link", "below", "below")], "t")
    assert np.allclose(link["translation"], [0, 0, -0.1])  # upside down: +z of the frame is down


@pytest.mark.parametrize(
    "frames, match",
    [
        ([{"name": "a", "parent": "nowhere"}], "neither a body"),
        ([{"name": "b", "parent": "a"}, {"name": "a", "parent": "base_link"}], "declared before"),
        ([{"name": "base_link", "parent": "base_link"}], "own parent"),
        ([{"name": "a", "parent": "base_link"}, {"name": "a", "parent": "base_link"}], "twice"),
        ([{"name": "a", "parent": "base_link", "xyz": [0, 0, 0]}], "unknown key"),
        ([{"name": "a", "parent": "base_link", "pos": [0, 0]}], "'pos'"),
        ([{"parent": "base_link"}], "'name' is required"),
    ],
)
def test_a_frame_that_cannot_be_placed_is_refused(frames, match):
    with pytest.raises(PluginError, match=match):
        add_frame_sites(_model(), parse_frames(frames, "t"), "t")


def test_a_frame_named_like_a_body_or_site_is_refused():
    spec = _model()
    spec.body("base_link").add_body(name="wheel")
    spec.body("base_link").add_site(name="lidar")
    for name in ("wheel", "lidar"):
        with pytest.raises(PluginError, match="already a"):
            add_frame_sites(spec, parse_frames([{"name": name, "parent": "base_link"}], "t"), "t")


def test_substitute_fills_known_placeholders_and_refuses_the_rest():
    values = {"frame_id": "laser", "parent_frame": "base_link"}
    assert substitute({"a": ["{frame_id}_mount", "{parent_frame}"], "n": 3}, values, "w") == {
        "a": ["laser_mount", "base_link"],
        "n": 3,
    }
    assert substitute("{{literal}}", values, "w") == "{literal}"
    for bad, match in (
        ("{frame}", "not one"),
        ("{frame_id!r}", "no conversion"),
        ("{", "template"),
    ):
        with pytest.raises(PluginError, match=match):
            substitute(bad, values, "w")


def test_a_static_tf_endpoint_carries_only_its_list():
    ep = static_tf_endpoint("frames", "robot", "ns", [{"parent": "p", "child": "c"}])
    assert ep.read() is None and ep.namespace == "ns" and ep.owner == "robot"
    assert ep.backend["ros2"]["static_tf"] == [{"parent": "p", "child": "c"}]


def _write(tmp_path, manifest):
    (tmp_path / "dev.xml").write_text("<mujoco/>")
    (tmp_path / "dev.manifest.yaml").write_text(textwrap.dedent(manifest))
    return tmp_path / "dev.xml"


def test_manifest_frames_reads_the_block_raw(tmp_path):
    model = _write(tmp_path, "frames:\n  - {name: '{frame_id}', parent: mount}\n")
    assert manifest_frames(model) == [{"name": "{frame_id}", "parent": "mount"}]
    assert manifest_frames(_write(tmp_path / ".." / tmp_path.name, "components: []\n")) == []


def test_manifest_frame_id_is_the_vendor_default_and_refuses_a_template(tmp_path):
    assert manifest_frame_id(_write(tmp_path, "frame_id: laser\n")) == "laser"
    assert manifest_frame_id(_write(tmp_path, "components: []\n")) is None
    assert manifest_frame_id(tmp_path / "absent.xml") is None
    for bad in ("frame_id: '{frame_id}'\n", "frame_id: ''\n", "frame_id: [a]\n"):
        with pytest.raises(PluginError, match="only placeholder may be"):
            manifest_frame_id(_write(tmp_path, bad))
    # The vendor macro's `name` is the one thing a default frame name may leave to the mount.
    templated = "frame_id: '{device_name}_color_optical_frame'\n"
    assert manifest_frame_id(_write(tmp_path, templated)) == "{device_name}_color_optical_frame"


def test_manifest_device_name_is_the_vendor_macro_default_and_refuses_a_template(tmp_path):
    assert manifest_device_name(_write(tmp_path, "device_name: camera\n")) == "camera"
    assert manifest_device_name(_write(tmp_path, "components: []\n")) is None
    assert manifest_device_name(tmp_path / "absent.xml") is None
    for bad in ("device_name: '{device_name}'\n", "device_name: ''\n", "device_name: [a]\n"):
        with pytest.raises(PluginError, match="no placeholder"):
            manifest_device_name(_write(tmp_path, bad))


def test_expand_manifest_substitutes_into_nested_configs_and_refuses_unknown(tmp_path):
    model = _write(
        tmp_path,
        """
        components:
          - capture: {frame_id: "{frame_id}", tf: {parent: "{parent_frame}"}}
        """,
    )
    # Stand-in refs: core tests name no sibling package's plugin, even one that would resolve here.
    spec = PluginSpec("mount", "scan", {"model": str(model)})
    (lidar,) = expand_manifest(
        spec, [spec], substitutions={"frame_id": "laser", "parent_frame": "x"}
    )
    assert lidar.config["frame_id"] == "laser" and lidar.config["tf"] == {"parent": "x"}
    # Without substitutions a manifest's strings are what it wrote.
    (raw,) = expand_manifest(spec, [spec])
    assert raw.config["frame_id"] == "{frame_id}"
    with pytest.raises(PluginError, match=r"\{frame_id\}"):
        expand_manifest(spec, [spec], substitutions={"parent_frame": "x"})
