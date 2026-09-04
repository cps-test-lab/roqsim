# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The model catalog describes each model from the model, and names the licence it carries.

Both halves of a catalog row used to be file-shaped rather than informative: every row repeated its
package's summary, and the licence was the sidecar's *file name*. Both are now read out of what
ships -- the sentence that opens the MJCF, and the licence text itself -- which only holds while
every model file actually opens with such a sentence. That is what the first test pins: a model
whose header opens with a generator banner or an element note falls back to describing its package,
and nothing else would notice.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("docutils")  # the doc extensions parse RST; docs are a dev-time extra

_EXT = Path(__file__).resolve().parents[2] / "docs" / "_ext"


@pytest.fixture(scope="module")
def model_docs():
    """The ``roqsim-models`` directive module, loaded by path (docs/_ext is not a package)."""
    sys.path.insert(0, str(_EXT))  # it imports its sibling `_render` by bare name
    try:
        spec = importlib.util.spec_from_file_location("model_docs", _EXT / "model_docs.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_EXT))


def _installed_models() -> list[Path]:
    from roqsim.models import ENTRY_POINT_GROUP, _entry_points, _provider_dirs

    files: list[Path] = []
    for ep in sorted(_entry_points(ENTRY_POINT_GROUP), key=lambda e: e.name):
        module = import_module(ep.value.split(":")[0] if isinstance(ep.value, str) else ep.value)
        models_dir = Path(_provider_dirs(module)[0])
        files += sorted(models_dir.glob("*.xml"))
        files += [
            sub / f"{sub.name}.xml"
            for sub in sorted(p for p in models_dir.iterdir() if p.is_dir())
            if (sub / f"{sub.name}.xml").is_file()
        ]
    return files


def test_every_installed_model_opens_with_a_sentence_saying_what_it_is(model_docs):
    """A new model gets caught here rather than by a reader wondering what it is."""
    missing = [str(f) for f in _installed_models() if model_docs._headline(f) is None]
    assert not missing, (
        "these models do not open with a headline sentence, so the catalog can only repeat their "
        f"package's summary: {missing}"
    )


def test_a_licence_notice_gives_up_its_holder(model_docs):
    assert model_docs._license_holder("BSD 3-Clause License\n\nCopyright (c) 2023, Someone") == (
        "Someone"
    )
    assert model_docs._license_holder("Copyright:  Someone (someone@example.org)") == "Someone"
    assert model_docs._license_holder(
        "\\copyright Copyright (c) 2015, Someone, All rights reserved."
    ) == ("Someone")


def test_stock_licence_prose_is_not_read_as_a_holder(model_docs):
    """An unfilled Apache appendix and the body's own wording name nobody."""
    assert model_docs._license_holder("   Copyright [yyyy] [name of copyright owner]") is None
    assert model_docs._license_holder('"Licensor" shall mean the copyright owner or entity') is None
    assert model_docs._license_holder("must retain the above copyright notice, this list") is None


def test_licence_line_names_the_licence_and_offers_the_file(model_docs, tmp_path):
    sidecar = tmp_path / "robot_LICENSE"
    sidecar.write_text("BSD 3-Clause License\n\nCopyright (c) 2023, Someone\n")
    env = type("Env", (), {"srcdir": str(tmp_path)})()

    line = model_docs._license_line(env, [sidecar])

    assert line.startswith("License: BSD-3-Clause (© Someone)")
    assert ":download:`robot_LICENSE <robot_LICENSE>`" in line


def test_an_unrecognised_licence_text_is_named_by_file(model_docs, tmp_path):
    """A vendor's own permission note has no SPDX name; the catalog must still point at it."""
    sidecar = tmp_path / "mesh_LICENSE"
    sidecar.write_text("The mesh is derived from the vendor's CAD; refer to their terms.\n")
    env = type("Env", (), {"srcdir": str(tmp_path)})()

    assert model_docs._license_line(env, [sidecar]) == (
        "License: see :download:`mesh_LICENSE <mesh_LICENSE>`."
    )


def test_a_flat_provider_takes_the_licence_the_model_declares(model_docs, tmp_path):
    """Two vendors' terms in one directory: "the file beside it" would attribute the wrong one."""
    (tmp_path / "LICENSE.vendor_a").write_text("MIT License\n\nCopyright (c) 2024 A\n")
    (tmp_path / "LICENSE.vendor_b").write_text("MIT License\n\nCopyright (c) 2024 B\n")
    model = tmp_path / "robot.xml"
    model.write_text('<mujoco model="robot"/>')
    manifest = tmp_path / "robot.manifest.yaml"

    assert model_docs._license_files(model) == []  # ambiguous, and nothing is guessed

    manifest.write_text("license: LICENSE.vendor_b\n")
    assert model_docs._license_files(model) == [tmp_path / "LICENSE.vendor_b"]
