"""``sim.density`` / ``sim.viscosity`` / ``sim.wind``: the medium, as world-scoped mjOption fields.

These join the ``<option>`` passthroughs the ``sim:`` block already carried (``integrator``,
``solver``, ``cone``, ``gravity``, ...). They exist because MuJoCo defaults ``density`` and
``viscosity`` to **0** -- a vacuum -- so before them a world had no way to ask for air, which left an
aerial model no honest option but to pin ``<option>`` itself and thereby reconfigure every world it
was spawned into.

``wind`` is only meaningful alongside a medium: MuJoCo feeds it into the drag terms as a relative
velocity, so with density and viscosity at 0 it does nothing at all. That is asserted here rather
than left as folklore.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine


def _opt(sim):
    engine = Engine(load_config_from_dict({"sim": sim, "components": []}, base_dir=Path(".")))
    engine.setup()
    engine.reset()
    try:
        opt = engine.ctx.model.opt
        return {
            "integrator": int(opt.integrator),
            "density": float(opt.density),
            "viscosity": float(opt.viscosity),
            "wind": [float(v) for v in opt.wind],
            "gravity": [float(v) for v in opt.gravity],
        }
    finally:
        engine.shutdown()


def test_default_world_is_a_vacuum():
    # Not a bug -- MuJoCo's default, and the whole reason this key exists. Pinned so that if the
    # default ever changes, the aerial docs and warnings get revisited rather than quietly rotting.
    opt = _opt({})
    assert opt["density"] == 0.0
    assert opt["viscosity"] == 0.0


def test_medium_and_integrator_are_applied():
    opt = _opt({"integrator": "rk4", "density": 1.225, "viscosity": 1.8e-5})
    assert opt["integrator"] == int(mujoco.mjtIntegrator.mjINT_RK4)
    assert opt["density"] == pytest.approx(1.225)
    assert opt["viscosity"] == pytest.approx(1.8e-5)


def test_vector_fields_are_applied():
    opt = _opt({"wind": [1.5, 0.0, -0.25], "density": 1.225})
    assert opt["wind"] == pytest.approx([1.5, 0.0, -0.25])


def test_wind_without_a_medium_is_inert():
    # Not a validation error -- MuJoCo simply has no drag term to apply it to. Asserted so the
    # combination is a documented fact rather than a surprise in someone's run.
    opt = _opt({"wind": [5.0, 0.0, 0.0]})
    assert opt["wind"] == pytest.approx([5.0, 0.0, 0.0])
    assert opt["density"] == 0.0 and opt["viscosity"] == 0.0


def test_medium_lands_in_the_run_record():
    # Applied to the spec BEFORE compile, so it is in the compiled model and therefore in the run's
    # provenance -- not mutated afterwards, where a recording would not carry it.
    config = load_config_from_dict(
        {"sim": {"density": 1.225}, "components": []}, base_dir=Path(".")
    )
    assert config.as_record()["sim"]["density"] == 1.225


def test_unknown_integrator_is_rejected():
    # `integrator` predates this change and is an enum, so a bad value still fails loudly.
    with pytest.raises(KeyError):
        _opt({"integrator": "verlet"})
