# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""One picture drawn through two cameras whose frusta agree to float32, compared as a rasteriser
draws it.

The labels are equal to the bit. Colour and depth are interpolated across a triangle, and a
rasteriser may round that the other way at a few pixels when the frustum's last bits differ: a GPU
draws both pictures equal, a software rasteriser (OSMesa) differs by one colour level, or by a depth
a float32 apart, at a fraction of a percent of the pixels.
"""

from __future__ import annotations

import numpy as np

#: The share of pixels whose colour may be one level off, or whose depth more than RTOL.
FEW = 0.005
#: How far apart two depths of one surface may be, relative.
RTOL = 1e-5


def assert_same_picture(got, want, kind: str) -> None:
    """*got* is *want*, one image *kind* (``rgb``, ``depth``, ``segmentation``) of one picture."""
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, kind
    if kind == "segmentation":
        assert np.array_equal(got, want), kind
    elif kind == "rgb":
        off = np.abs(got.astype(int) - want.astype(int)).max(axis=-1)
        assert off.max() <= 1 and (off > 0).mean() < FEW, (
            kind,
            int(off.max()),
            int((off > 0).sum()),
        )
    else:
        assert np.array_equal(np.isfinite(got), np.isfinite(want)), kind
        both = np.isfinite(got)
        far = np.zeros(got.shape, dtype=bool)
        far[both] = np.abs(got[both] - want[both]) > RTOL * np.abs(want[both])
        assert far.mean() < FEW, (kind, int(far.sum()))
