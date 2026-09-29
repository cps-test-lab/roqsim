"""A tool mounted by default is in the wrench a force_torque on the arm reads.

A site force sensor reads the subtree of the body its site is on. A tool welded beside that body
rather than below it is outside the reading -- no weight, no push, no contact -- and nothing says
so. Measured here per arm, on the running model: a tool of known mass, mounted where ``spawn_arm``
puts it by default, must show up at the arm's force/torque site at rest, and so must a known push
on it. An arm model that carries a sensor stack names its site ``fts_site``; on any other the
sensor sits at the mount site itself, which is the flange a real sensor would be bolted to.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from roqsim_manipulation.plugins.spawn_arm import manifest_end_effector_site
from roqsim_manipulation_assets.models import MODELS_DIR
from roqsim_sensors.plugins.force_torque import ForceTorquePlugin

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine

MASS_KG = 0.5
PUSH_N = 10.0
HOLD_S = 0.5
#: Relative. Measured: every arm within 0.1 % of both the weight and the push after the hold.
TOLERANCE = 0.02

TOOL = f"""<mujoco><worldbody><body name="probe">
  <geom type="box" size=".01 .01 .01" pos="0 0 .02" mass="{MASS_KG}" contype="0" conaffinity="0"/>
</body></worldbody></mujoco>"""

#: Every bundled model whose manifest says where a tool goes -- found, not listed, so an arm added
#: with a declaration is measured here without anyone editing this file.
MOUNTABLE = sorted(
    d.name
    for d in MODELS_DIR.iterdir()
    if (d / f"{d.name}.xml").is_file() and manifest_end_effector_site(d / f"{d.name}.xml")
)

#: Arms that ship a hand of their own and so no free flange: a tool needs an explicit `site:`.
PRE_ASSEMBLED = ["gen3", "panda", "vx300s", "wx250s"]


def _sites(arm: str) -> set[str]:
    spec = mujoco.MjSpec.from_file(str(MODELS_DIR / arm / f"{arm}.xml"))
    return {s.name for s in spec.sites}


def _ft_site(arm: str) -> str:
    return (
        "fts_site"
        if "fts_site" in _sites(arm)
        else manifest_end_effector_site(MODELS_DIR / arm / f"{arm}.xml")
    )


def _engine(tmp_path: Path, arm: str, *, tool=TOOL, ee=None, ft=None) -> Engine:
    spawn: dict = {"model": arm, "prefix": "a_"}
    if tool is not None:
        (tmp_path / "tool.xml").write_text(tool, encoding="utf-8")
        spawn["end_effector"] = {"model": "tool.xml", **(ee or {})}
    entry = {"spawn_arm": spawn, "name": "arm"}
    if ft is not False:
        ft = {"site": _ft_site(arm), "frame": "world", **(ft or {})}
        entry["components"] = [{"force_torque": ft, "name": "ft"}]
    cfg = load_config_from_dict(
        {"sim": {}, "components": [entry]},
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    engine.reset()
    return engine


def _force_at_rest(engine: Engine, push: float = 0.0) -> np.ndarray:
    """World-frame force at the sensor after the arm has held its home pose for ``HOLD_S``."""
    model, data = engine.ctx.model, engine.ctx.data
    probe = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "a_probe")
    for _ in range(int(HOLD_S / model.opt.timestep)):
        if push:
            data.xfrc_applied[probe, 2] = -push
        engine.step()
    ft = next(p for p in engine.plugins if isinstance(p, ForceTorquePlugin))
    return ft.read()[0]


def test_every_arm_with_a_flange_declares_where_a_tool_goes():
    assert set(MOUNTABLE) >= {"m1013", "open_manipulator_x", "ur10e", "ur5e", "xarm7"}
    for arm in MOUNTABLE:
        site = manifest_end_effector_site(MODELS_DIR / arm / f"{arm}.xml")
        assert site in _sites(arm), f"{arm}'s manifest names site {site!r}, which its MJCF lacks"


@pytest.mark.parametrize("arm", MOUNTABLE)
def test_a_tool_mounted_by_default_is_in_the_arms_wrench(tmp_path, arm):
    bare = _force_at_rest(_engine(tmp_path, arm, tool=None))
    engine = _engine(tmp_path, arm)
    loaded = _force_at_rest(engine)
    weight = MASS_KG * float(np.linalg.norm(engine.ctx.model.opt.gravity))
    assert abs(bare[2] - loaded[2]) == pytest.approx(weight, rel=TOLERANCE), (
        f"{arm}: the tool's {weight:.3f} N is not in the reading at {_ft_site(arm)!r}"
    )
    pushed = _force_at_rest(_engine(tmp_path, arm), push=PUSH_N)
    assert abs(pushed[2] - loaded[2]) == pytest.approx(PUSH_N, rel=TOLERANCE)


def test_the_ur5e_mounts_past_its_sensor_stack(tmp_path):
    engine = _engine(tmp_path, "ur5e")
    tool = engine.ctx.entities.get("arm").meta["end_effector"]
    assert tool == {"site": "a_tool_site", "bodies": ["a_probe"]}
    model = engine.ctx.model
    assert model.body(int(model.body("a_probe").parentid[0])).name == "a_tool0"


def test_a_tool_on_the_ur5e_flange_is_refused_under_its_stack(tmp_path):
    """The bare flange hangs the tool beside tool0: measured, the sensor reads 0.000 N of it."""
    with pytest.raises(RuntimeError, match=r"reads the subtree of body 'a_tool0'.*not in it"):
        _engine(tmp_path, "ur5e", ee={"site": "attachment_site"})


def test_a_sensor_on_the_flange_reads_a_tool_on_the_flange(tmp_path):
    """The rule is the subtree, not a site name: the flange body carries a tool mounted on it."""
    engine = _engine(
        tmp_path, "ur5e", ee={"site": "attachment_site"}, ft={"site": "attachment_site"}
    )
    assert engine.ctx.entities.get("arm").meta["end_effector"]["site"] == "a_attachment_site"


def test_a_sensor_inside_the_tool_sees_the_tool(tmp_path):
    """A fingertip sensor is in the tool rather than above it, and is not refused."""
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [
                {
                    "spawn_arm": {
                        "model": "ur5e",
                        "prefix": "a_",
                        "end_effector": {"model": "robotiq_2f85"},
                    },
                    "name": "arm",
                    "components": [{"force_torque": {"site": "pinch"}, "name": "ft"}],
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 0
    engine.setup()
    assert engine.ctx.entities.get("arm").meta["end_effector"]["bodies"] == ["a_base_mount"]


def test_an_arm_without_a_tool_records_none(tmp_path):
    engine = _engine(tmp_path, "ur5e", tool=None)
    assert "end_effector" not in engine.ctx.entities.get("arm").meta


@pytest.mark.parametrize("arm", PRE_ASSEMBLED)
def test_an_arm_with_its_own_hand_has_no_default_mount(tmp_path, arm):
    with pytest.raises(RuntimeError, match=r"has no site 'attachment_site'"):
        _engine(tmp_path, arm, ft=False)


def test_a_malformed_manifest_declaration_is_refused(tmp_path):
    (tmp_path / "arm.xml").write_text("<mujoco/>", encoding="utf-8")
    (tmp_path / "arm.manifest.yaml").write_text("end_effector: tool_site\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'end_effector' takes one key, 'site'"):
        manifest_end_effector_site(tmp_path / "arm.xml")
