# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0
"""``roqsim bake``: its errors say what to do (no Blender, an old one, a GPU type that is not there, a
bake over ``max_mb``), the pipeline from Blender's passes to the stored bake works against a stand-in
for Blender, and -- where Blender is installed -- a small world bakes on the CPU and loads baked."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import textwrap

import pytest
from test_bake_compose import write_world

from roqsim.bake import run
from roqsim.bake.run import BakeRunError, find_blender
from roqsim.config import load_config
from roqsim.engine import Engine

#: A stand-in for Blender: answers --version, and run as a bake writes what blender_bake.py writes --
#: a grid layout, and passes of the given light with noise -- or exits with the given code.
FAKE = r'''#!{python}
import json, math, os, sys
from pathlib import Path
import numpy as np
args = sys.argv[1:]
if "--version" in args:
    print("Blender {version}")
    sys.exit(0)
code = int(os.environ.get("FAKE_BLENDER_EXIT", "0"))
if code:
    print(os.environ.get("FAKE_BLENDER_SAY", "roqsim-bake: failed"))
    sys.exit(code)
opts = dict(zip(args[args.index("--") + 1 :][0::2], args[args.index("--") + 1 :][1::2]))
work, out = Path(opts["--work"]), Path(opts["--out"])
out.mkdir(parents=True, exist_ok=True)
man = json.loads((work / "manifest.json").read_text())
rng = np.random.default_rng(0)
noise = float(os.environ.get("FAKE_BLENDER_NOISE", "0"))
times = {{}}
for e in opts["--entities"].split(","):
    spec = man["entities"][e]
    s, n = spec["size"], spec["triangles"]
    g = math.ceil(math.sqrt(n))
    cell = 1.0 / g
    base = np.array([[0.15, 0.15], [0.85, 0.15], [0.15, 0.85]]) * cell
    i = np.arange(n)
    uv = (np.stack([(i % g) * cell, (i // g) * cell], 1)[:, None] + base[None]).astype(np.float32)
    np.save(out / f"{{e}}.uv2.npy", uv)
    sides = ["", "_back"] if spec["two_sided"] else [""]
    for side in sides:
        light = np.full((s, s, 3), 2.0, np.float32) + noise * rng.random((s, s, 3), np.float32)
        np.save(out / f"{{e}}.light{{side}}.npy", light.astype(np.float16))
        np.save(out / f"{{e}}.direct{{side}}.npy", (light * 0.6).astype(np.float16))
        np.save(out / f"{{e}}.ao{{side}}.npy", np.ones((s, s, 3), np.float16))
        nrm = np.zeros((s, s, 3), np.float16)
        nrm[..., 0] = 1.0
        nrm[..., 2] = 0.5
        np.save(out / f"{{e}}.normal{{side}}.npy", nrm)
    times[e] = {{"size": s, "uv_s": 0.0, "bake_s": {{}}}}
(out / f"times.{{opts['--entities'].split(',')[0]}}.json").write_text(json.dumps({{"entities": times}}))
'''


@pytest.fixture
def fake_blender(tmp_path):
    def make(version="4.2.0"):
        path = tmp_path / "bin" / "blender"
        path.parent.mkdir(exist_ok=True)
        path.write_text(FAKE.format(python=sys.executable, version=version))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return str(path)

    return make


def _plugin(world):
    from roqsim.plugins.baked_lighting import BAKING, BakedLightingPlugin

    engine = Engine(load_config(str(world)), preview=True)
    engine.ctx.blackboard.set(BAKING, True)
    engine.setup()
    plugin = next(p for p in engine.plugins if isinstance(p, BakedLightingPlugin))
    return engine, plugin


def test_no_blender_says_where_to_get_it(monkeypatch, tmp_path):
    monkeypatch.delenv("BLENDER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(BakeRunError, match=r"blender\.org.*--blender PATH"):
        find_blender()


def test_an_old_blender_is_refused(fake_blender):
    with pytest.raises(BakeRunError, match="needs 4.2 or later"):
        find_blender(fake_blender("3.6.5"))


def test_a_gpu_type_that_is_not_there_names_the_cpu(fake_blender, tmp_path, monkeypatch):
    world = write_world(tmp_path)
    engine, plugin = _plugin(world)
    monkeypatch.setenv("FAKE_BLENDER_EXIT", "3")
    monkeypatch.setenv("FAKE_BLENDER_SAY", "roqsim-bake: no OPTIX device found; --device cpu bakes on the CPU")
    try:
        with pytest.raises(BakeRunError, match="--device cpu"):
            run.bake(plugin, device="optix", blender=fake_blender())
    finally:
        engine.shutdown()


def test_the_passes_become_a_bake_the_world_loads(fake_blender, tmp_path):
    world = write_world(tmp_path)
    engine, plugin = _plugin(world)
    try:
        summary = run.bake(plugin, blender=fake_blender(), jobs=2)
    finally:
        engine.shutdown()
    assert summary["path"] == str(tmp_path / "room.bake") and summary["jobs"] == 2
    meta = json.loads((tmp_path / "room.bake" / "bake.json").read_text())
    assert meta["blender"] == "4.2.0" and meta["device"] == "cpu" and set(meta["entities"]) == {"floor", "shell", "P"}
    # The light is 2 everywhere: the exposure puts the shell's at SHELL_P95.
    assert meta["exposure"] == pytest.approx(0.55 / 2.0, rel=0.02)
    engine = Engine(load_config(str(world)), preview=True)
    engine.setup()
    try:
        assert [c["code"] for c in engine.ctx.conditions] == ["bake_used"]
    finally:
        engine.shutdown()


def test_a_rebake_reuses_the_layout_of_what_did_not_change(fake_blender, tmp_path):
    world = write_world(tmp_path)
    for _ in range(2):
        engine, plugin = _plugin(world)
        try:
            summary = run.bake(plugin, blender=fake_blender())
        finally:
            engine.shutdown()
    assert sorted(summary["uv_reused"]) == ["P", "floor", "shell"]


def test_a_bake_over_max_mb_is_refused_naming_its_largest_maps(fake_blender, tmp_path, monkeypatch):
    world = write_world(tmp_path, max_mb=0.05)
    monkeypatch.setenv("FAKE_BLENDER_NOISE", "1.0")  # noise does not compress
    engine, plugin = _plugin(world)
    try:
        with pytest.raises(BakeRunError, match=r"over max_mb 0\.05.*light\.png.*texels_per_m"):
            run.bake(plugin, blender=fake_blender())
    finally:
        engine.shutdown()
    assert not (tmp_path / "room.bake").exists()


def test_a_world_without_the_component_is_told_how_to_add_it(tmp_path, capsys, fake_blender):
    (tmp_path / "w.yaml").write_text("sim: {world: empty_room}\n")
    assert run.main([str(tmp_path / "w.yaml"), "--blender", fake_blender()]) == 1
    assert "- baked_lighting: {}" in capsys.readouterr().err


def test_dry_run_says_what_it_would_bake(tmp_path, capsys):
    world = write_world(tmp_path)
    assert run.main([str(world), "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out["entities"]) == {"floor", "shell", "P"} and out["lights"] == "fixtures"
    assert out["path"] == str(tmp_path / "room.bake")


@pytest.mark.skipif(
    not (os.environ.get("BLENDER") or shutil.which("blender")), reason="Blender is not installed"
)
def test_a_small_world_bakes_on_the_cpu_and_loads_baked(tmp_path):
    """The real thing: Cycles on the CPU, at a coarse resolution so it takes seconds."""
    world = write_world(tmp_path, texels_per_m={"shell": 8, "floor": 16, "props": 60}, samples=16)
    engine, plugin = _plugin(world)
    try:
        summary = run.bake(plugin, device="cpu", jobs=3)
    finally:
        engine.shutdown()
    assert summary["times"]["blender_s"] > 0 and summary["bytes"] < 5e6
    engine = Engine(load_config(str(world)), preview=True)
    engine.setup()
    try:
        assert [c["code"] for c in engine.ctx.conditions] == ["bake_used"]
    finally:
        engine.shutdown()
    print(textwrap.dedent(f"baked on the CPU in {summary['times']['blender_s']} s"))
