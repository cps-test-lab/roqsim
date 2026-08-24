# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A manifest is a document, so it may ``extends:`` another model's.

`unitree_g1_dex1` IS a `unitree_g1` plus hands. It said so by repeating the base's locomotion and
lidar blocks verbatim, which then had to be kept in step by hand. Inheritance says it once.

What is inherited is COMPONENTS, not geometry: a derived model keeps its own MJCF. `extends:` never
carried geometry -- `sim.world` did -- and `sim:` is refused in a manifest entirely, because a model
is a component of a world and must not reach up to set the run's seed or pacing.
"""

import textwrap

import pytest

from roqsim.config import PluginError
from roqsim.manifest import load_manifest


def _model(models_dir, name, body=None):
    f = models_dir / f"{name}.xml"
    f.write_text("<mujoco/>")
    if body is not None:
        (models_dir / f"{name}.manifest.yaml").write_text(textwrap.dedent(body))
    return f


def _refs(entries):
    return [next(k for k in e if k not in ("name", "components")) for e in entries]


def test_a_derived_manifest_inherits_the_bases_components(tmp_path):
    _model(tmp_path, "base", """
        components:
          - locomotion: {}
          - lidar: {rays: 360}
    """)
    derived = _model(tmp_path, "derived", """
        extends: base.xml
        components:
          - arm_controller: {}
    """)
    assert _refs(load_manifest(derived, base_dir=tmp_path)) == [
        "locomotion",
        "lidar",
        "arm_controller",
    ]


def test_the_base_comes_first_so_the_derived_model_can_override_it(tmp_path):
    """Order is the contract: a derived entry with the base's label is the one that runs."""
    _model(tmp_path, "base", "components:\n  - lidar: {rays: 360}\n")
    derived = _model(
        tmp_path, "derived", "extends: base.xml\ncomponents:\n  - lidar: {rays: 1440}\n"
    )
    entries = load_manifest(derived, base_dir=tmp_path)
    assert [e["lidar"]["rays"] for e in entries] == [360, 1440]


def test_a_cycle_raises_naming_the_chain(tmp_path):
    _model(tmp_path, "a", "extends: b.xml\ncomponents: []\n")
    b = _model(tmp_path, "b", "extends: a.xml\ncomponents: []\n")
    with pytest.raises(PluginError, match="cycle"):
        load_manifest(b, base_dir=tmp_path)


def test_a_manifest_may_not_carry_a_sim_block(tmp_path):
    """A model is a component of a world. Letting its manifest set `seed` or `pacing` would let a
    robot change the experiment it is part of, from a file the world never opened."""
    m = _model(tmp_path, "loud", "sim: {seed: 7}\ncomponents: []\n")
    with pytest.raises(PluginError) as exc:
        load_manifest(m, base_dir=tmp_path)
    assert "sim:" in str(exc.value) and "loud" in str(exc.value)


def test_the_refusal_reaches_a_base_manifest_too(tmp_path):
    """Otherwise `sim:` could be smuggled in through a base a reader never looks at."""
    _model(tmp_path, "sneaky_base", "sim: {pacing: asap}\ncomponents: []\n")
    derived = _model(tmp_path, "derived", "extends: sneaky_base.xml\ncomponents: []\n")
    with pytest.raises(PluginError, match="sim:"):
        load_manifest(derived, base_dir=tmp_path)
