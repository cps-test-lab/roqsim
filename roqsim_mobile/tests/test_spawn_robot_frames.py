"""spawn_robot's ``frames:``: a vendor's flattened fixed links, as sites and as published transforms.

A synthetic robot written to tmp_path, so what is pinned is the mechanism rather than any one
shipped model's numbers.
"""

from __future__ import annotations

import textwrap

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

ROBOT = """
<mujoco>
  <worldbody>
    <body name="base_link">
      <freejoint name="base_free"/>
      <geom type="box" size="0.2 0.2 0.05"/>
      <body name="body_link" pos="0 0 0.1">
        <geom type="box" size="0.1 0.1 0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

MANIFEST = """
frames:
  - {name: cover_link, parent: body_link, pos: [0.02, 0, 0.05]}
  - {name: laser, parent: cover_link, pos: [0, 0, 0.01], rpy: [3.141592653589793, 0, 0]}
"""


def _robot(tmp_path, manifest=MANIFEST):
    (tmp_path / "bot.xml").write_text(ROBOT)
    (tmp_path / "bot.manifest.yaml").write_text(textwrap.dedent(manifest))
    return str(tmp_path / "bot.xml")


def _engine(model, **spawn):
    cfg = load_config_from_dict(
        {
            "components": [
                {"spawn_robot": {"model": model, "prefix": "r_", **spawn}, "name": "robot"}
            ]
        }
    )
    engine = Engine(cfg)
    engine.setup()
    return engine


def _frames(engine):
    return next(e for e in engine.ctx.interface.all() if e.name == "frames")


def test_frames_become_prefixed_sites_at_their_composed_pose(tmp_path):
    engine = _engine(_robot(tmp_path))
    m, d = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "r_laser")
    assert sid >= 0
    body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[sid]))
    assert body == "r_body_link"  # a frame costs a site on its chain's body, not a body
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_base_link")
    assert np.allclose(d.site_xpos[sid] - d.xpos[base], [0.02, 0.0, 0.16])


def test_frames_are_published_as_one_static_chain_in_the_robots_namespace(tmp_path):
    ep = _frames(_engine(_robot(tmp_path), namespace="rb"))
    assert ep.namespace == "rb" and ep.owner == "robot" and ep.read() is None
    body, cover, laser = ep.backend["ros2"]["static_tf"]
    # Bare names: the bridge applies the namespace. The chain starts at a body that is not the root,
    # so the root's link to it comes first, or the chain would be a tree of its own.
    assert (body["parent"], body["child"]) == ("base_link", "body_link")
    assert np.allclose(body["translation"], [0, 0, 0.1])
    assert (cover["parent"], cover["child"]) == ("body_link", "cover_link")
    assert np.allclose(cover["translation"], [0.02, 0, 0.05])
    assert (laser["parent"], laser["child"]) == ("cover_link", "laser")
    assert np.allclose(np.abs(laser["rotation"]), [0, 1, 0, 0], atol=1e-9)  # roll pi


def test_config_frames_extend_the_manifests(tmp_path):
    engine = _engine(
        _robot(tmp_path), frames=[{"name": "mast_link", "parent": "laser", "pos": [0, 0, 0.3]}]
    )
    links = _frames(engine).backend["ros2"]["static_tf"]
    assert [(link["parent"], link["child"]) for link in links][-1] == ("laser", "mast_link")
    # Hung under the upside-down laser, +0.3 in its frame is 0.3 down in the robot's.
    assert np.allclose(links[-1]["translation"], [0, 0, 0.3])


def test_a_chain_from_the_root_publishes_no_extra_link(tmp_path):
    manifest = "frames:\n  - {name: mast_link, parent: base_link, pos: [0, 0, 0.3]}\n"
    links = _frames(_engine(_robot(tmp_path, manifest=manifest))).backend["ros2"]["static_tf"]
    assert [(link["parent"], link["child"]) for link in links] == [("base_link", "mast_link")]


def test_a_frame_on_a_jointed_body_is_refused(tmp_path):
    (tmp_path / "bot.xml").write_text(
        ROBOT.replace(
            '<body name="body_link" pos="0 0 0.1">',
            '<body name="body_link" pos="0 0 0.1"><joint name="pan" type="hinge"/>',
        )
    )
    (tmp_path / "bot.manifest.yaml").write_text(textwrap.dedent(MANIFEST))
    with pytest.raises(Exception, match="not static"):
        _engine(str(tmp_path / "bot.xml"))


def test_a_robot_without_frames_publishes_no_frames_endpoint(tmp_path):
    engine = _engine(_robot(tmp_path, manifest="components: []\n"))
    assert all(e.name != "frames" for e in engine.ctx.interface.all())


@pytest.mark.parametrize(
    "frames, match",
    [
        ([{"name": "cover_link", "parent": "body_link"}], "twice"),
        ([{"name": "x", "parent": "no_such_link"}], "neither a body"),
        ([{"name": "body_link", "parent": "base_link"}], "already a body"),
    ],
)
def test_a_bad_frame_is_refused_by_name(tmp_path, frames, match):
    with pytest.raises(Exception, match=match):
        _engine(_robot(tmp_path), frames=frames)
