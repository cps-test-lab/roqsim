"""``spawn_model``'s ``scale`` key: one asset serves every size a scene needs.

The case that matters is a prop built from more than meshes. Scaling only ``mesh.scale`` -- the
obvious implementation -- leaves primitive geoms and the offsets of parts inside a body at their
original size, so the prop comes apart instead of getting smaller. ``door_glass`` is exactly that
shape (a mesh pane plus a ``<geom type="box">`` chrome rail at a non-zero ``pos``), so it is the
model these tests scale.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from roqsim.plugins.spawn_model import SpawnModelPlugin

# A prop with a mesh geom AND an offset primitive geom; the mixed case a mesh-only scale breaks.
_MIXED_MODEL = "door_glass"


def _world_extent(model_ref: str, scale: float) -> np.ndarray:
    """World-space bounding box of everything a single spawn of ``model_ref`` puts in the scene."""
    spec = mujoco.MjSpec()
    SpawnModelPlugin({"model": model_ref, "scale": scale, "name": "t"}).build(spec, None)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        pos, r = data.geom_xpos[g], model.geom_rbound[g]
        lo = np.minimum(lo, pos - r)
        hi = np.maximum(hi, pos + r)
    return hi - lo


def test_scale_shrinks_every_part_uniformly():
    """Halving the scale halves the whole prop -- meshes and the offset primitive alike."""
    full = _world_extent(_MIXED_MODEL, 1.0)
    half = _world_extent(_MIXED_MODEL, 0.5)
    np.testing.assert_allclose(half, full * 0.5, rtol=1e-6)


def test_default_scale_leaves_the_prop_untouched():
    """Omitting the key must be byte-for-byte the old behaviour, not a 1.0 round-trip."""
    np.testing.assert_allclose(
        _world_extent(_MIXED_MODEL, 1.0), _world_extent(_MIXED_MODEL, 1.0), rtol=0
    )


def test_primitive_geom_scales_with_the_meshes():
    """Guard the specific regression: the box rail's size must shrink, not just the meshes."""
    sizes = {}
    for scale in (1.0, 0.25):
        spec = mujoco.MjSpec()
        SpawnModelPlugin({"model": _MIXED_MODEL, "scale": scale, "name": "t"}).build(spec, None)
        model = spec.compile()
        boxes = [
            model.geom_size[g]
            for g in range(model.ngeom)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
        ]
        assert boxes, "expected door_glass to contribute a box geom"
        sizes[scale] = np.array(boxes[0])
    np.testing.assert_allclose(sizes[0.25], sizes[1.0] * 0.25, rtol=1e-6)


@pytest.mark.parametrize(
    "bad, expected",
    [
        ([2.0, 1.0, 1.0], "uniform"),  # per-axis shears rotated children / breaks spheres
        (0.0, "> 0"),
        (-1.0, "> 0"),
        ("big", "number"),
    ],
)
def test_bad_scale_is_rejected_by_validate_config(bad, expected):
    errors = SpawnModelPlugin({}).validate_config({"model": _MIXED_MODEL, "scale": bad})
    assert any(expected in e for e in errors), errors


def test_valid_scale_passes_validation():
    assert SpawnModelPlugin({}).validate_config({"model": _MIXED_MODEL, "scale": 0.42}) == []
