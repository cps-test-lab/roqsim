# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0
"""What a bake sees of a world (``roqsim.bake``): the static/moving split, where the lights come from,
and a fingerprint that changes with what the bake depends on and nothing else. No Blender."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.bake import fingerprint as fp
from roqsim.bake.analysis import analyse
from roqsim.bake.settings import PRESETS, settings_for

PANEL = '<geom name="light_panel" type="box" size="0.3 0.3 0.01" pos="0 0 2.48" material="glow"/>'
LAMP = '<light name="lamp" pos="0 0 2.4" dir="0 0 -1" diffuse="0.5 0.5 0.5" cutoff="90"/>'


def room(*, wall_x=2.0, door_x=1.0, ceiling=True, panel=True, lamp=True, noise=0.0, extra=""):
    """A box room: floor, four walls, a ceiling, a static prop, a hinged door, a free box, a mocap body,
    an emissive ceiling panel and a lamp."""
    w = wall_x + noise
    return f"""
    <mujoco>
      <asset>
        <material name="glow" rgba="1 1 1 1" emission="1"/>
        <material name="grey" rgba="0.6 0.6 0.6 1"/>
      </asset>
      <worldbody>
        {LAMP if lamp else ""}
        <geom name="floor_visual" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
        <geom name="wall_e" type="box" size="0.05 2 1.25" pos="{w} 0 1.25" material="grey"/>
        <geom name="wall_w" type="box" size="0.05 2 1.25" pos="-2 0 1.25" material="grey"/>
        <geom name="wall_n" type="box" size="2 0.05 1.25" pos="0 2 1.25" material="grey"/>
        <geom name="wall_s" type="box" size="2 0.05 1.25" pos="0 -2 1.25" material="grey"/>
        {'<geom name="ceiling" type="box" size="2 2 0.02" pos="0 0 2.52" material="grey"/>' if ceiling else ""}
        {PANEL if panel else ""}
        <geom name="collider" type="box" size="1 1 1" pos="0 0 -1" group="3"/>
        <body name="P/table" pos="0.5 0.5 0">
          <geom name="P/top" type="box" size="0.4 0.3 0.02" pos="0 0 0.7" rgba="0.5 0.3 0.1 1"/>
          <geom name="P/leg" type="cylinder" size="0.03 0.35" pos="0 0 0.35" rgba="0.2 0.2 0.2 1"/>
        </body>
        <body name="door" pos="{door_x} -1.9 1">
          <joint name="hinge" type="hinge" axis="0 0 1"/>
          <geom name="leaf" type="box" size="0.4 0.02 1" rgba="0.7 0.5 0.3 1"/>
        </body>
        <body name="crate" pos="-1 -1 0.2">
          <freejoint/>
          <geom name="crate" type="box" size="0.2 0.2 0.2" rgba="0.6 0.4 0.2 1"/>
        </body>
        <body name="target" mocap="true" pos="1 1 1">
          <geom name="target" type="sphere" size="0.05" rgba="1 0 0 1"/>
        </body>
        {extra}
      </worldbody>
    </mujoco>"""


def analysed(xml, **kwargs):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return analyse(m, d, kwargs.pop("settings", PRESETS["default"]), **kwargs)


def kinds(a):
    return {g.key: g.kind for g in a.export.geoms}


def test_static_drawn_geoms_are_baked_and_what_moves_occludes():
    a = analysed(room())
    k = kinds(a)
    for key in ("floor_visual", "wall_e", "wall_w", "ceiling", "P/top", "P/leg"):
        assert k[key] == "baked", key
    for key in ("leaf", "crate", "target"):
        assert k[key] == "moving", key
    assert k["light_panel"] == "emitter"
    assert "collider" not in k  # group 3 is not drawn
    # What hangs on a joint to the world throws its shadow into the bake, at reset: the door's leaf
    # (a box, 12 triangles). The free crate and the mocap target do not: they are elsewhere every run.
    assert len(a.export.occluders) == 12
    assert a.export.occluders[..., 1].min() < -1.9  # the door stands at y = -1.9


def test_the_occluders_can_be_named():
    assert len(analysed(room(), occluders=()).export.occluders) == 0
    assert len(analysed(room(), occluders=("crate",)).export.occluders) == 12


def test_a_robot_added_to_the_world_does_not_make_its_bake_stale():
    robot = """<body name="robot" pos="0 1 0.1"><freejoint/>
        <geom name="robot/chassis" type="box" size="0.3 0.2 0.1"/>
        <body name="robot/arm" pos="0 0 0.2"><joint type="hinge"/>
        <geom name="robot/link" type="capsule" size="0.03 0.2"/></body></body>"""
    assert verdict(analysed(room()), analysed(room(extra=robot)))["status"] == "fresh"


def test_entities_are_the_prefix_the_floor_and_the_shell():
    a = analysed(room())
    ents = a.export.entities
    assert set(ents) == {"floor", "shell", "P"}
    assert (
        ents["floor"].cls == "floor" and ents["shell"].cls == "shell" and ents["P"].cls == "props"
    )
    assert ents["P"].two_sided and not ents["shell"].two_sided
    # The floor plane is the floor entity; the walls' tops and the ceiling are the shell.
    floor_geoms = {a.export.geoms[int(g)].key for g in np.unique(ents["floor"].geom)}
    assert floor_geoms == {"floor_visual"}
    # Every light map is a multiple of 64 and within its class's cap.
    for ent in ents.values():
        assert ent.size % 64 == 0 and 64 <= ent.size <= 4096
    assert ents["P"].size <= PRESETS["default"].prop_cap


def test_a_static_geom_can_be_excluded_from_the_bake():
    a = analysed(room(), exclude=("P/*",))
    assert kinds(a)["P/top"] == "moving" and "P" not in a.export.entities


def test_a_shell_wound_inside_out_is_turned_to_face_the_room():
    # A box wound inside out, as imported walls can be: its mesh normals point into the solid.
    verts = "-1 -1 0  1 -1 0  1 1 0  -1 1 0  -1 -1 1  1 -1 1  1 1 1  -1 1 1"
    faces = "0 1 2  0 2 3  4 6 5  4 7 6  0 5 1  0 4 5  1 6 2  1 5 6  2 7 3  2 6 7  3 4 0  3 7 4"
    xml = f"""<mujoco><asset><mesh name="inv" vertex="{verts}" face="{faces}"/></asset>
      <worldbody><geom name="block" type="mesh" mesh="inv"/></worldbody></mujoco>"""
    a = analysed(xml)
    assert any("turned" in n for n in a.export.notes)


# -- lights -----------------------------------------------------------------------------------------


def test_emitting_fixtures_come_first_and_the_lamp_is_their_stand_in():
    a = analysed(room())
    assert a.light["rule"] == "fixtures"
    assert [e["geom"] for e in a.light["emitters"]] == ["light_panel"]
    assert a.light["stand_ins"] == ["lamp"] and a.light["lights"] == []


def test_without_fixtures_the_mjcf_lights():
    a = analysed(room(panel=False))
    assert a.light["rule"] == "mjcf_lights"
    (lamp,) = a.light["lights"]
    assert lamp["type"] == "point" and lamp["name"] == "lamp"  # a cutoff of 90 degrees is a point


def test_a_narrow_spot_and_a_directional_light():
    extra = (
        '<light name="spot" pos="1 0 2" dir="0 0 -1" cutoff="30" diffuse="1 1 1"/>'
        '<light name="sun" directional="true" dir="0 0 -1" diffuse="0.4 0.4 0.4"/>'
    )
    a = analysed(room(panel=False, lamp=False, extra=extra))
    by_name = {x["name"]: x for x in a.light["lights"]}
    assert by_name["spot"]["type"] == "spot" and by_name["spot"]["spot_deg"] == pytest.approx(60.0)
    assert by_name["sun"]["type"] == "sun"


def test_with_neither_the_documented_default():
    a = analysed(room(panel=False, lamp=False))
    assert a.light["rule"] == "default"
    (light,) = a.light["lights"]
    assert light["type"] == "area" and light["pos"][2] == pytest.approx(2.54 - 0.1, abs=1e-6)
    assert a.light["sky"] > 0


def test_emitters_name_lamps_no_material_declares():
    a = analysed(room(panel=False), emitters=[{"match": "P/top", "strength_w_m2": 20.0}])
    assert kinds(a)["P/top"] == "emitter"
    assert a.light["emitters"][0]["strength_w_m2"] == 20.0
    with pytest.raises(ValueError, match="match no drawn static geom"):
        analysed(room(), emitters=[{"match": "nothing*"}])


def test_an_explicit_lights_list_replaces_the_declared_ones():
    lights = [{"type": "sun", "dir": [0, 0, -1], "strength": 2.0}]
    a = analysed(room(panel=False), lights=lights)
    assert a.light["rule"] == "explicit" and a.light["lights"] == lights


# -- the fingerprint --------------------------------------------------------------------------------


def verdict(before, after):
    return fp.compare(before.fingerprint, after.fingerprint)


def test_the_same_world_is_fresh():
    assert verdict(analysed(room()), analysed(room()))["status"] == "fresh"


def test_float_noise_of_a_micrometre_changes_nothing():
    assert verdict(analysed(room()), analysed(room(noise=1e-6)))["status"] == "fresh"


def test_a_moved_wall_is_that_geom_only():
    v = verdict(analysed(room()), analysed(room(wall_x=2.2)))
    assert v["status"] == "stale" and v["changed"] == ["wall_e"]
    assert "wall_e" in v["reasons"][0]


def test_a_new_static_geom_is_stale():
    extra = '<geom name="shelf" type="box" size="0.2 0.2 0.5" pos="-1.5 1.5 0.5"/>'
    v = verdict(analysed(room()), analysed(room(extra=extra)))
    assert v["status"] == "stale" and v["added"] == ["shelf"]


def test_a_door_standing_elsewhere_at_reset_changes_the_occluders_only():
    v = verdict(analysed(room()), analysed(room(door_x=0.0)))
    assert v["status"] == "stale" and v["changed"] == [] and v["added"] == []
    assert any("what moves" in r for r in v["reasons"])


def test_a_removed_ceiling_is_partial_not_stale():
    v = verdict(analysed(room()), analysed(room(ceiling=False)))
    assert v["status"] == "partial" and v["missing"] == ["ceiling"]


def test_other_lights_are_stale():
    v = verdict(
        analysed(room()),
        analysed(room(), emitters=[{"match": "light_panel", "strength_w_m2": 10.0}]),
    )
    assert v["status"] == "stale" and any("lights" in r for r in v["reasons"])


def test_other_settings_are_stale_and_say_which():
    v = verdict(analysed(room()), analysed(room(), settings=settings_for("default", samples=128)))
    assert v["status"] == "stale" and any("samples" in r for r in v["reasons"])
    v = verdict(analysed(room()), analysed(room(), settings=PRESETS["high"]))
    assert any("texels_per_m" in r for r in v["reasons"])


def test_a_texture_edit_is_a_material_change():
    tex = '<texture name="t" type="2d" builtin="checker" width="8" height="8" rgb1="{}" rgb2="0 0 0"/>'
    mat = '<material name="tm" texture="t"/>'

    def world(rgb):
        return (
            room(extra="")
            .replace("<asset>", "<asset>" + tex.format(rgb) + mat)
            .replace(
                'name="wall_w" type="box" size="0.05 2 1.25" pos="-2 0 1.25" material="grey"',
                'name="wall_w" type="box" size="0.05 2 1.25" pos="-2 0 1.25" material="tm"',
            )
        )

    v = verdict(analysed(world("1 1 1")), analysed(world("0.5 0.5 0.5")))
    assert v["changed"] == ["wall_w"]


def test_the_floor_is_the_largest_surface_facing_up_not_the_lowest():
    # A foundation slab under the floor, as an imported room can have.
    slab = '<geom name="slab" type="box" size="1 1 0.05" pos="0 0 -0.3"/>'
    ents = analysed(room(extra=slab)).export.entities
    floor_geoms = {
        analysed(room(extra=slab)).export.geoms[int(g)].key for g in np.unique(ents["floor"].geom)
    }
    assert floor_geoms == {"floor_visual"}


def test_two_unnamed_geoms_of_unnamed_bodies_are_refused_by_name():
    """A geom without a name is named after its body; two in bodies without names would be one geom
    to the bake, so the world says what to name instead of baking or drawing them as one."""
    xml = room(
        extra='<body pos="1 1 0.2"><geom type="box" size=".1 .1 .1"/></body>'
        '<body pos="-1 1 0.2"><geom type="box" size=".1 .1 .1"/></body>'
    )
    with pytest.raises(ValueError, match="name the geoms or their bodies"):
        analysed(xml)


def test_a_robot_with_a_lamp_keeps_the_bake_and_its_lamp():
    """A light on what moves is lit live, as what carries it is: not the world's, not baked."""
    robot = """<body name="robot" pos="0 1 0.1"><freejoint/>
        <geom name="robot/chassis" type="box" size="0.3 0.2 0.1"/>
        <light name="robot/headlamp" pos="0.3 0 0.1" dir="1 0 0" diffuse="0.5 0.5 0.5"/></body>"""
    without = room(panel=False)  # the MJCF lights are the bake's (rule 2)
    v = verdict(analysed(without), analysed(room(panel=False, extra=robot)))
    assert v["status"] == "fresh", v
