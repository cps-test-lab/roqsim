"""Asset de-duplication: identical file-backed meshes/materials attached per instance collapse to
one, references retarget onto the survivor, and the compiled model is geometrically unchanged."""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.assets import deduplicate_assets


def _prop(tmp_path):
    """A one-body prop MJCF referencing a mesh + a textured material from files on disk."""
    obj = tmp_path / "box.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n")
    png = tmp_path / "tex.png"
    # a trivial valid PNG (1x1) so MuJoCo has a file to key on and load
    import struct
    import zlib

    raw = b"\x00\xff\x00\x00\xff"  # one scanline: filter byte 0x00 + one RGBA pixel
    idat = zlib.compress(raw)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    png.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )
    return (
        f"<mujoco><asset>"
        f'<mesh name="m" file="{obj}"/>'
        f'<texture name="t" type="2d" file="{png}"/>'
        f'<material name="mat" texture="t"/>'
        f'</asset><worldbody><body name="b">'
        f'<geom type="mesh" mesh="m" material="mat"/>'
        f"</body></worldbody></mujoco>"
    )


def _world_of(n, tmp_path):
    """A parent spec with ``n`` attached copies of the same prop under distinct prefixes."""
    parent = mujoco.MjSpec.from_string("<mujoco/>")
    for i in range(n):
        child = mujoco.MjSpec.from_string(_prop(tmp_path))
        parent.attach(child, prefix=f"p{i}_", frame=parent.worldbody.add_frame())
    return parent


def test_dedup_collapses_identical_attached_assets(tmp_path):
    spec = _world_of(5, tmp_path)
    assert spec.compile().nmesh == 5  # baseline: one copy per attach, no sharing
    spec = _world_of(5, tmp_path)

    removed = deduplicate_assets(spec)
    assert removed["meshes_removed"] == 4
    assert removed["materials_removed"] == 4
    assert removed["textures_removed"] == 4

    model = spec.compile()
    assert model.nmesh == 1
    assert model.ntex == 1
    assert model.nmat == 1
    assert model.ngeom == 5  # every instance's geom survives, now sharing the one mesh/material


def test_dedup_preserves_geometry(tmp_path):
    ref = _world_of(4, tmp_path).compile()
    deduped = _world_of(4, tmp_path)
    deduplicate_assets(deduped)
    got = deduped.compile()
    assert got.nbody == ref.nbody and got.ngeom == ref.ngeom
    # geom world positions unchanged by the merge.
    d_ref, d_got = mujoco.MjData(ref), mujoco.MjData(got)
    mujoco.mj_forward(ref, d_ref)
    mujoco.mj_forward(got, d_got)
    np.testing.assert_allclose(d_got.geom_xpos, d_ref.geom_xpos, atol=1e-9)


def test_dedup_noop_when_unique(tmp_path):
    # Two props with different geometry (different scale) must not merge.
    parent = mujoco.MjSpec.from_string("<mujoco/>")
    for i, scale in enumerate((1.0, 2.0)):
        child = mujoco.MjSpec.from_string(_prop(tmp_path))
        child.meshes[0].scale = [scale, scale, scale]
        parent.attach(child, prefix=f"p{i}_", frame=parent.worldbody.add_frame())
    removed = deduplicate_assets(parent)
    assert removed["meshes_removed"] == 0
    assert parent.compile().nmesh == 2
