"""What the floorplan plugin declares as its own dependencies.

The mesh and its json-ld colliders are one artifact split across two directories, and the
rule that ties them together lives here rather than in whoever is staging a world.
"""

def test_the_floorplan_declares_its_mesh_and_its_colliders(tmp_path):
    """One artifact split across two directories, so both travel or neither works.

    The colliders live at ``<env>/json-ld/`` *relative to the mesh*; a caller staging only the
    mesh gets a floorplan this plugin refuses to build (visual-only walls). Only the plugin
    knows that rule, which is why it declares it rather than a caller guessing.
    """
    from roqsim_mobile.plugins.floorplan import FloorplanPlugin

    mesh = tmp_path / "3d-mesh" / "hex.stl"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"solid\n")
    jdir = tmp_path / "json-ld"
    jdir.mkdir()
    (jdir / "floorplan.fpm.json").write_text('{"@graph": []}')

    sources = FloorplanPlugin({"mesh": str(mesh)}).sources()

    assert str(mesh) in sources
    assert str(jdir / "floorplan.fpm.json") in sources


def test_a_floorplan_without_a_mesh_declares_nothing():
    from roqsim_mobile.plugins.floorplan import FloorplanPlugin

    assert FloorplanPlugin({}).sources() == []
