"""The merged-mesh export: the frame contract, the unit scale, and what each format actually carries.

Two tests here matter more than the rest. **The frame contract** (``test_export_ignores_where_the_body
_sits``) is the property the whole export exists for: a pose estimated against the mesh is the pose of
the chosen body, which holds only if the geom transforms are composed back through the body tree
instead of read from the world. And **the unit scale** (``test_millimetres_are_exactly_a_thousand
_metres``), because an STL carries no unit: a factor-1000 slip produces a file that looks perfectly
fine everywhere except in the CAD tool that imports it 1000x too small.

The primitive volumes are asserted against the INSCRIBED polyhedron, not against the analytic solid.
A tessellated cylinder is smaller than the cylinder by a known amount, so the exact form pins the
tessellation itself -- an analytic comparison with a loose tolerance would pass just as happily on a
generator that had silently changed.
"""

from __future__ import annotations

import json
import re
import struct
import zipfile

import mujoco
import numpy as np
import pytest

from roqsim.export_mesh import (
    MeshExporter,
    MeshExportError,
    _edge_stats,
    _signed_volume,
    main,
)

# One box, one cylinder, one sphere and a per-face-duplicated tetrahedron, spread over a parent and a
# child body so the body-tree composition is exercised. The tetrahedron's 12 vertices are deliberately
# unshared: that is the shipped-mesh failure mode --weld exists for. The faces must be spelled out --
# given vertices alone MuJoCo computes the convex hull, which silently welds them for us.
_TET = " ".join(
    " ".join(f"{c}" for c in v)
    for tri in (
        ((0, 0, 0), (0, 1, 0), (1, 0, 0)),
        ((0, 0, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    for v in tri
)

MJCF = """
<mujoco model="fixture">
  <asset>
    <mesh name="tet" vertex="{tet}" face="0 1 2  3 4 5  6 7 8  9 10 11"/>
    <hfield name="terrain" nrow="2" ncol="2" size="1 1 1 0.1"/>
    <material name="red" rgba="1 0 0 1"/>
    <material name="blue" rgba="0 0 1 1"/>
  </asset>
  <worldbody>
    <geom name="ground" type="plane" size="0 0 1"/>
    <body name="base" pos="{base_pos}" quat="{base_quat}">
      <geom name="box" type="box" size="0.1 0.2 0.3" material="red"/>
      <geom name="tetra" type="mesh" mesh="tet" pos="0 0 1" material="blue"/>
      <geom name="hull" type="cylinder" size="0.164 0.03" group="3"/>
      <body name="wheel" pos="0 0.5 0" quat="0.70710678 0.70710678 0 0">
        <joint name="spin" type="hinge" axis="0 0 1"/>
        <geom name="tyre" type="cylinder" size="0.05 0.01" group="0" material="red"/>
      </body>
      <body name="dome" pos="0.3 0 0">
        <geom name="ball" type="sphere" size="0.07" material="blue"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def model(base_pos: str = "0 0 0.4", base_quat: str = "1 0 0 0", extra: str = ""):
    xml = MJCF.format(tet=_TET, base_pos=base_pos, base_quat=base_quat)
    if extra:
        xml = xml.replace("</worldbody>", extra + "\n  </worldbody>")
    return mujoco.MjModel.from_xml_string(xml)


def merged(mdl, **kwargs):
    scale = kwargs.pop("scale", 1.0)
    exporter = MeshExporter(mdl, **kwargs)
    exporter.collect()
    return exporter, exporter.merge(scale)


def geom_of(exporter, name: str) -> dict:
    return next(g for g in exporter.geoms if g["name"] == name)


# -- the frame contract ----------------------------------------------------------------------------


def test_frame_defaults_to_the_root_of_the_selection():
    exporter, _ = merged(model())
    # base, wheel and dome all carry selected geoms; their common ancestor is base -- found from the
    # tree, not from a hardcoded link name.
    assert exporter.frame == "base"


def test_export_ignores_where_the_body_sits():
    """A robot compiled at a different pose must export to the same file. This is the whole contract."""
    _, here = merged(model(base_pos="0 0 0.4"))
    _, moved = merged(model(base_pos="7 -3 1.25", base_quat="0.7071 0 0 0.7071"))
    assert np.allclose(here["verts"], moved["verts"], atol=0)
    assert (here["faces"] == moved["faces"]).all()


def test_a_body_outside_the_frame_is_refused():
    mdl = model()
    with pytest.raises(MeshExportError, match="not below the frame"):
        merged(mdl, frame="dome")


def test_unknown_frame_is_refused():
    with pytest.raises(MeshExportError, match="frame body 'nope' not found"):
        merged(model(), frame="nope")


# -- units and geometry ----------------------------------------------------------------------------


def test_millimetres_are_exactly_a_thousand_metres():
    _, metres = merged(model())
    _, millimetres = merged(model(), scale=1000.0)
    assert np.allclose(millimetres["verts"], metres["verts"] * 1000.0, atol=0)


def test_box_volume_is_exact():
    exporter, _ = merged(model())
    box = geom_of(exporter, "box")
    assert _signed_volume(box["verts"], box["faces"]) == pytest.approx(
        8 * 0.1 * 0.2 * 0.3, rel=1e-12
    )


@pytest.mark.parametrize("segments", [8, 16, 32])
def test_cylinder_matches_the_inscribed_prism(segments):
    """Volume of the emitted mesh == the inscribed n-gon prism, exactly. Pins the tessellation."""
    exporter, _ = merged(model(), segments=segments)
    tyre = geom_of(exporter, "tyre")
    r, half = 0.05, 0.01
    expected = 0.5 * segments * r * r * np.sin(2 * np.pi / segments) * 2 * half
    assert _signed_volume(tyre["verts"], tyre["faces"]) == pytest.approx(expected, rel=1e-9)


def test_sphere_approaches_the_analytic_volume():
    exporter, _ = merged(model(), segments=48)
    ball = geom_of(exporter, "ball")
    analytic = 4.0 / 3.0 * np.pi * 0.07**3
    assert _signed_volume(ball["verts"], ball["faces"]) == pytest.approx(analytic, rel=0.01)


def test_closed_geoms_wind_outward():
    """A negative volume is an inverted mesh: it renders as a hole and CAD reads it as a void."""
    exporter, _ = merged(model())
    for geom in exporter.geoms:
        if _edge_stats(geom["faces"])["closed"]:
            assert _signed_volume(geom["verts"], geom["faces"]) > 0, geom["name"]


# -- selection -------------------------------------------------------------------------------------


def test_collision_group_is_excluded_by_default():
    exporter, _ = merged(model())
    assert "hull" not in {g["name"] for g in exporter.geoms}
    assert {g["name"] for g in exporter.geoms} == {"box", "tetra", "tyre", "ball"}


def test_the_collision_group_can_be_asked_for_on_its_own():
    exporter, _ = merged(model(), groups=[3])
    assert [g["name"] for g in exporter.geoms] == ["hull"]


def test_exclude_drops_bodies_by_name():
    exporter, _ = merged(model(), exclude=["wheel"])
    assert "tyre" not in {g["name"] for g in exporter.geoms}
    assert any("wheel" in entry for entry in exporter.skipped)


def test_an_empty_selection_is_refused():
    with pytest.raises(MeshExportError, match="nothing to export"):
        merged(model(), groups=[2])


def test_a_plane_is_skipped_and_reported():
    exporter, _ = merged(model(), frame="world")
    assert "ground" not in {g["name"] for g in exporter.geoms}
    assert any("plane" in entry for entry in exporter.skipped)


def test_a_heightfield_is_refused_rather_than_stood_in_for():
    mdl = model(extra='<body name="terra"><geom name="hf" type="hfield" hfield="terrain"/></body>')
    with pytest.raises(MeshExportError, match="HFIELD"):
        merged(mdl, frame="world")


# -- welding ---------------------------------------------------------------------------------------


def test_weld_closes_a_mesh_whose_vertices_are_merely_duplicated():
    open_shell = geom_of(merged(model(), weld=0.0)[0], "tetra")
    welded = geom_of(merged(model(), weld=1e-6)[0], "tetra")
    assert len(open_shell["verts"]) == 12  # MuJoCo keeps the duplicates as authored
    assert _edge_stats(open_shell["faces"])["boundary"] == 12
    assert len(welded["verts"]) == 4
    assert _edge_stats(welded["faces"])["closed"]


# -- the CLI and the formats -----------------------------------------------------------------------


def run(tmp_path, capsys, *args) -> dict:
    mjcf = tmp_path / "fixture.xml"
    mjcf.write_text(MJCF.format(tet=_TET, base_pos="0 0 0.4", base_quat="1 0 0 0"))
    code = main(["--mjcf", str(mjcf), *[str(a) for a in args]])
    assert code == 0, capsys.readouterr().err
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_binary_stl_header_and_triangle_count(tmp_path, capsys):
    out = tmp_path / "robot.stl"
    summary = run(tmp_path, capsys, "--out", out)
    raw = out.read_bytes()
    header, count = raw[:80].decode().strip(), struct.unpack("<I", raw[80:84])[0]
    assert count == summary["triangles"]
    assert len(raw) == 84 + count * 50
    assert "base" in header and "(m)" in header
    assert "/" not in header  # no path, no command line -- see the module docstring


def test_ascii_stl_is_a_solid_with_one_facet_per_triangle(tmp_path, capsys):
    out = tmp_path / "robot.stl"
    summary = run(tmp_path, capsys, "--out", out, "--format", "stl-ascii")
    text = out.read_text()
    assert text.startswith("solid ") and text.rstrip().endswith("endsolid fixture")
    assert text.count("facet normal") == summary["triangles"]


def test_obj_carries_a_material_group_per_material(tmp_path, capsys):
    out = tmp_path / "robot.obj"
    summary = run(tmp_path, capsys, "--out", out)
    text, mtl = out.read_text(), (tmp_path / "robot.mtl").read_text()
    assert text.count("\nv ") + text.startswith("v ") == summary["vertices"]
    assert sum(line.startswith("f ") for line in text.splitlines()) == summary["triangles"]
    assert [m["name"] for m in summary["materials"]] == ["red", "blue"]
    for material in summary["materials"]:
        assert f"newmtl {material['name']}" in mtl
        r, g, b, _a = material["rgba"]
        assert f"Kd {r:.6f} {g:.6f} {b:.6f}" in mtl
    assert text.count("usemtl ") == len(summary["materials"])


def test_obj_without_colour_writes_no_mtl(tmp_path, capsys):
    out = tmp_path / "robot.obj"
    summary = run(tmp_path, capsys, "--out", out, "--no-color")
    assert not (tmp_path / "robot.mtl").exists()
    assert "usemtl" not in out.read_text()
    assert summary["materials"] == []


def test_ply_carries_a_colour_per_vertex(tmp_path, capsys):
    out = tmp_path / "robot.ply"
    summary = run(tmp_path, capsys, "--out", out)
    raw = out.read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    lines = head.decode().splitlines()
    assert lines[0] == "ply" and lines[1] == "format binary_little_endian 1.0"
    assert f"element vertex {summary['vertices']}" in lines
    assert f"element face {summary['triangles']}" in lines
    assert "property uchar red" in lines
    assert len(body) == summary["vertices"] * 15 + summary["triangles"] * 13
    # The first vertex belongs to the red box, and its colour must have travelled as 255,0,0.
    assert struct.unpack("<3fBBB", body[:15])[3:] == (255, 0, 0)


def test_the_summary_reports_watertightness_per_geom(tmp_path, capsys):
    summary = run(tmp_path, capsys, "--out", tmp_path / "robot.stl")
    assert summary["frame"] == "base"
    assert summary["watertight"]["closed"] is True  # every fixture geom closes after the weld
    assert {g["name"] for g in summary["geoms"]} == {"box", "tetra", "tyre", "ball"}
    assert all(g["watertight"]["closed"] for g in summary["geoms"])


def test_a_repeated_export_is_byte_identical(tmp_path, capsys):
    first, second = tmp_path / "a.stl", tmp_path / "b.stl"
    run(tmp_path, capsys, "--out", first)
    run(tmp_path, capsys, "--out", second)
    assert first.read_bytes() == second.read_bytes()


def test_the_sidecar_is_the_summary(tmp_path, capsys):
    sidecar = tmp_path / "robot.json"
    summary = run(tmp_path, capsys, "--out", tmp_path / "robot.ply", "--sidecar", sidecar)
    assert json.loads(sidecar.read_text()) == summary


def test_an_unknown_extension_is_a_usage_error(tmp_path, capsys):
    mjcf = tmp_path / "fixture.xml"
    mjcf.write_text(MJCF.format(tet=_TET, base_pos="0 0 0", base_quat="1 0 0 0"))
    out = tmp_path / "robot.step"
    assert main(["--mjcf", str(mjcf), "--out", str(out)]) == 2
    assert not out.exists()


def test_a_refused_export_writes_nothing(tmp_path, capsys):
    mjcf = tmp_path / "fixture.xml"
    mjcf.write_text(MJCF.format(tet=_TET, base_pos="0 0 0", base_quat="1 0 0 0"))
    out = tmp_path / "robot.stl"
    assert main(["--mjcf", str(mjcf), "--out", str(out), "--groups", "2"]) == 1
    assert not out.exists()


def test_the_manifest_names_the_source(tmp_path, capsys):
    manifest = tmp_path / "gen.json"
    mjcf = tmp_path / "fixture.xml"
    mjcf.write_text(MJCF.format(tet=_TET, base_pos="0 0 0", base_quat="1 0 0 0"))
    assert (
        main(["--mjcf", str(mjcf), "--out", str(tmp_path / "r.stl"), "--manifest", str(manifest)])
        == 0
    )
    assert json.loads(manifest.read_text()) == {"inputs": [str(mjcf.resolve())]}


# -- 3MF -------------------------------------------------------------------------------------------
#
# 3MF is the only format here that keeps the parts apart and states its unit, which is exactly what a
# CAD tool needs, so both properties are asserted rather than assumed. It is also a zip, and a zip
# stores a timestamp per member -- hence the determinism test below, which fails on a naive writer.


def model_xml(path) -> str:
    with zipfile.ZipFile(path) as package:
        return package.read("3D/3dmodel.model").decode()


def test_3mf_is_a_package_with_the_three_required_members(tmp_path, capsys):
    out = tmp_path / "robot.3mf"
    run(tmp_path, capsys, "--out", out)
    with zipfile.ZipFile(out) as package:
        assert set(package.namelist()) == {
            "[Content_Types].xml",
            "_rels/.rels",
            "3D/3dmodel.model",
        }
        rels = package.read("_rels/.rels").decode()
        assert "/3D/3dmodel.model" in rels
        assert "3dmanufacturing/2013/01/3dmodel" in rels
        assert "3dmanufacturing-3dmodel+xml" in package.read("[Content_Types].xml").decode()


@pytest.mark.parametrize(("units", "declared"), [("m", "meter"), ("mm", "millimeter")])
def test_3mf_states_its_unit(tmp_path, capsys, units, declared):
    """The one thing an STL cannot do, and the reason a 0.35 m robot imports 0.35 mm tall."""
    out = tmp_path / "robot.3mf"
    run(tmp_path, capsys, "--out", out, "--units", units)
    assert f'unit="{declared}"' in model_xml(out)


def test_3mf_is_one_named_object_per_geom_in_a_single_build_item(tmp_path, capsys):
    out = tmp_path / "robot.3mf"
    summary = run(tmp_path, capsys, "--out", out)
    xml = model_xml(out)
    objects = re.findall(r'<object id="(\d+)"[^>]*name="([^"]+)"', xml)
    parts = [name for _id, name in objects]
    assert summary["objects"] == len(summary["geoms"]) == 4
    assert parts[:-1] == ["box", "tetra", "tyre", "ball"]
    assert parts[-1] == "fixture"  # the assembly the build item points at
    root = objects[-1][0]
    assert f'<build><item objectid="{root}"/></build>' in xml
    assert xml.count("<component objectid=") == len(summary["geoms"])
    # every triangle and vertex of the merged mesh is present, just distributed over the parts
    assert xml.count("<vertex ") == summary["vertices"]
    assert xml.count("<triangle ") == summary["triangles"]


def test_3mf_carries_the_materials_as_display_colours(tmp_path, capsys):
    out = tmp_path / "robot.3mf"
    summary = run(tmp_path, capsys, "--out", out)
    xml = model_xml(out)
    for index, material in enumerate(summary["materials"]):
        r, g, b, a = (int(round(v * 255)) for v in material["rgba"])
        assert (
            f'<base name="{material["name"]}" displaycolor="#{r:02X}{g:02X}{b:02X}{a:02X}"/>' in xml
        )
        assert f'pindex="{index}"' in xml
    assert '<basematerials id="1">' in xml


def test_3mf_without_colour_declares_no_materials(tmp_path, capsys):
    out = tmp_path / "robot.3mf"
    run(tmp_path, capsys, "--out", out, "--no-color")
    xml = model_xml(out)
    assert "basematerials" not in xml
    assert "pindex" not in xml


def test_3mf_repeats_byte_identically(tmp_path, capsys):
    """A zip stores a timestamp per member, so this fails on any writer that lets the clock in."""
    first, second = tmp_path / "a.3mf", tmp_path / "b.3mf"
    run(tmp_path, capsys, "--out", first)
    run(tmp_path, capsys, "--out", second)
    assert first.read_bytes() == second.read_bytes()


def test_3mf_names_repeated_parts_apart(tmp_path, capsys):
    """Four identical standoffs must not arrive as four parts with one name."""
    from roqsim.export_mesh import _unique

    assert _unique(["a", "b", "a", "a", "c"]) == ["a", "b", "a_2", "a_3", "c"]
