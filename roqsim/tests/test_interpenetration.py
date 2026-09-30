# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A start state that puts one body inside another is named, with both sides and the depth.

The overlap does not stop a world from loading; the contact solver resolves it on the first steps
with a force that grows with the depth. These tests pin what is reported (a box sunk into a table,
a flex pushed through a geom), what is not (a box resting on it, a pair MuJoCo does not collide,
a contact soft enough to be sunk into), and that the engine's reset says so in the run's log.
"""

from __future__ import annotations

import logging
import textwrap

import mujoco
import pytest

from roqsim.config import load_config
from roqsim.context import Entity, EntityRegistry
from roqsim.engine import Engine
from roqsim.interpenetration import (
    DEFAULT_TOLERANCE,
    HINT,
    contact_tolerance,
    interpenetrations,
    summary,
)

TABLE_TOP = 0.40  # z of the table's top surface


def _scene(crate_z: float, *, crate_attrs: str = "", extra: str = "", world_extra: str = "") -> str:
    """A static table and a free crate (10 cm cube) whose centre is at *crate_z*."""
    return f"""
    <mujoco>
      {extra}
      <worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <body name="table">
          <geom name="table_top" type="box" size=".4 .4 .02" pos="0 0 {TABLE_TOP - 0.02}"/>
        </body>
        <body name="crate" pos="0 0 {crate_z}">
          <freejoint/>
          <geom name="crate" type="box" size=".05 .05 .05" {crate_attrs}/>
        </body>
        {world_extra}
      </worldbody>
    </mujoco>
    """


def _forward(xml: str):
    model = mujoco.MjModel.from_xml_string(textwrap.dedent(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _registry(*pairs: tuple[str, str]) -> EntityRegistry:
    registry = EntityRegistry()
    for name, body in pairs:
        registry.add(Entity(name=name, kind="object", body=body))
    return registry


def test_a_box_sunk_into_a_table_is_named_with_both_sides_and_the_depth():
    model, data = _forward(_scene(TABLE_TOP + 0.05 - 0.03))
    (found,) = interpenetrations(model, data, _registry(("table", "table"), ("crate", "crate")))
    assert {found.first.name, found.second.name} == {"table_top", "crate"}
    assert {found.first.entity, found.second.entity} == {"table", "crate"}
    assert found.depth == pytest.approx(0.03, abs=1e-6)
    assert found.contacts > 1, "the corner contacts fold into one finding"
    text = found.describe()
    assert "'table_top' (entity 'table')" in text and "'crate' (entity 'crate')" in text
    assert "30.0 mm" in text


@pytest.mark.parametrize("sink", [0.0, 0.002])
def test_a_box_resting_on_a_table_is_not_reported(sink):
    """Touching, or sunk by less than the placement slack of a pose written by hand."""
    model, data = _forward(_scene(TABLE_TOP + 0.05 - sink))
    assert data.ncon > 0, "the crate does touch the table"
    assert interpenetrations(model, data) == []


def test_a_pair_mujoco_does_not_collide_is_not_reported():
    """An overlap without a constraint exerts no force: it is what the author asked for."""
    sunk = TABLE_TOP + 0.05 - 0.03
    for model, data in (
        _forward(_scene(sunk, crate_attrs='contype="0" conaffinity="0"')),
        _forward(_scene(sunk, extra='<contact><exclude body1="table" body2="crate"/></contact>')),
    ):
        assert interpenetrations(model, data) == []


def test_the_tolerance_follows_a_contact_softer_than_the_default():
    """A soft contact rests deeper; the depth at which its spring asks for one g grows with it."""
    sunk = TABLE_TOP + 0.05 - 0.02
    assert interpenetrations(*_forward(_scene(sunk)))
    soft = _scene(sunk, crate_attrs='solref="0.2 1"')
    assert interpenetrations(*_forward(soft)) == []

    model, _ = _forward(_scene(sunk))
    default = contact_tolerance(model, [[0.02, 1.0]], [[0.9, 0.95, 0.001, 0.5, 2.0]])
    assert default[0] == DEFAULT_TOLERANCE, "at MuJoCo's defaults the floor is the tolerance"
    wide = contact_tolerance(model, [[0.02, 1.0]], [[0.9, 0.95, 0.05, 0.5, 2.0]])
    assert wide[0] == pytest.approx(0.05), "a contact's own solimp width is honoured"


@pytest.mark.parametrize(("dim", "count"), [(2, "4 4 1"), (3, "3 3 3")])
def test_a_flex_through_a_geom_is_named_as_the_flex(dim, count):
    """``contact.geom`` is -1 on a flex side; indexing with it would name the model's last geom."""
    flex = f"""
        <body name="rig">
          <flexcomp name="sheet" type="grid" count="{count}" spacing=".05 .05 .05"
                    pos="0 0 {TABLE_TOP - 0.01}" radius="0.005" dim="{dim}" mass="0.1">
            <edge equality="true"/>
          </flexcomp>
        </body>
        <geom name="last_geom_in_the_model" type="sphere" size=".01" pos="3 3 3"/>
    """
    model, data = _forward(_scene(1.0, world_extra=flex))
    found = interpenetrations(model, data, _registry(("rig", "rig"), ("table", "table")))
    assert found, "the flex is pushed through the table top"
    top = found[0]
    sides = {(top.first.kind, top.first.name), (top.second.kind, top.second.name)}
    assert sides == {("flex", "sheet"), ("geom", "table_top")}
    flex_side = top.first if top.first.kind == "flex" else top.second
    assert flex_side.entity == "rig"
    assert "flex 'sheet' (entity 'rig')" in top.describe()
    assert all("last_geom_in_the_model" not in f.describe() for f in found)


def test_the_summary_names_the_deepest_few_and_says_what_to_change():
    two_crates = _scene(TABLE_TOP + 0.05 - 0.03).replace(
        "</worldbody>",
        f"""<body name="crate2" pos="0.2 0 {TABLE_TOP + 0.05 - 0.04}"><freejoint/>
            <geom name="crate2" type="box" size=".05 .05 .05"/></body></worldbody>""",
    )
    found = interpenetrations(*_forward(two_crates))
    assert [f.depth for f in found] == sorted((f.depth for f in found), reverse=True)
    line = summary(found, limit=1)
    assert "crate2" in line and "40.0 mm" in line
    assert "and 1 more pair(s)" in line
    assert "`home`" in HINT and HINT in line


# -- the engine: a run's own log says it --------------------------------------------------------


def _engine(tmp_path, crate_z: float) -> Engine:
    (tmp_path / "scene.xml").write_text(textwrap.dedent(_scene(crate_z)), encoding="utf-8")
    world = tmp_path / "world.yaml"
    world.write_text("sim: {world: scene.xml}\ncomponents: []\n", encoding="utf-8")
    engine = Engine(load_config(world), preview=True)
    engine.setup()
    return engine


def test_reset_logs_one_warning_naming_the_pair(tmp_path, caplog):
    engine = _engine(tmp_path, TABLE_TOP + 0.05 - 0.03)
    with caplog.at_level(logging.WARNING, logger="roqsim.engine"):
        engine.reset()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "table_top" in warnings[0].getMessage() and "crate" in warnings[0].getMessage()
    assert len(engine.interpenetrations) == 1
    engine.shutdown()


def test_reset_of_a_clean_start_state_logs_nothing(tmp_path, caplog):
    engine = _engine(tmp_path, TABLE_TOP + 0.05)
    with caplog.at_level(logging.WARNING, logger="roqsim.engine"):
        engine.reset()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert engine.interpenetrations == []
    engine.shutdown()
