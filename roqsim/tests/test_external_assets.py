"""Every fetched external source is pinned by content, not only by location.

A URL names where a file was, and a vendor that replaces the CAD behind one changes every mesh
regenerated from it while ``external_assets.yaml`` says nothing moved. The runner verifies a
``sha256`` when a source carries one, so the pin is what makes a regenerated asset the same asset.
A ``manual`` source is obtained by hand and has nothing to hash until it is.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MANIFEST = Path(__file__).resolve().parents[2] / "external" / "external_assets.yaml"


def test_every_fetched_source_carries_a_sha256():
    if not MANIFEST.is_file():
        pytest.skip("external/external_assets.yaml is a repository file, not part of the package")
    resources = yaml.safe_load(MANIFEST.read_text())["resources"]
    unpinned = [
        f"{res['name']}: {src['path']}"
        for res in resources
        for src in res["sources"]
        if not src.get("manual") and not src.get("sha256")
    ]
    assert not unpinned, f"fetched sources without a sha256 pin: {unpinned}"
