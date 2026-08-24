# SPDX-License-Identifier: Apache-2.0
"""Skip markers for the tests that need a generated, uncommitted vendor mesh.

The mid360, zivid and robin_w1g meshes are DERIVED from vendor CAD whose redistribution terms are
unclear, so they are generated locally by `make external-resources` and git-ignored -- and two of the
three sources sit behind a product page that cannot be fetched at all. A clean checkout (CI included)
therefore has the models' MJCF but not their meshes, and the engine fails at compile with "Error
opening file 'meshes/mid360_body.obj'". Skipping is the honest outcome: the asset is absent by
design, not broken -- the same call `make smoke` makes when a world names a git-ignored asset.

Shared between test modules rather than defined in one of them: any test that COMPILES a world naming
these sensors needs the guard, and a module without it fails in CI only. Not conftest.py, because two
of those exist in this repo and `import conftest` from a test module does not say which one it means.

See external/external_assets.yaml and the package's *_MESH_LICENSE files.
"""

from __future__ import annotations

import pathlib

import pytest

_MODELS = pathlib.Path(__file__).resolve().parent.parent / "src" / "roqsim_sensors" / "models"


def needs_external_mesh(model: str, mesh: str):
    return pytest.mark.skipif(
        not (_MODELS / model / "meshes" / mesh).is_file(),
        reason=(
            f"{model}: {mesh} is a generated external asset and is not committed "
            f"(run `make external-resources RESOURCE=...` -- some sources are manual downloads)"
        ),
    )


needs_mid360 = needs_external_mesh("mid360", "mid360_body.obj")
needs_zivid = needs_external_mesh("zivid", "zivid_body.obj")
needs_robin = needs_external_mesh("robin_w1g", "robin_w1g_body.obj")
