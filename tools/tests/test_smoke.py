# SPDX-License-Identifier: Apache-2.0
"""``make smoke`` (``tools/smoke.py``): a world passes when it runs, and a world with a scene only
when the scene finishes within the steps the world runs for."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SMOKE = Path(__file__).resolve().parent.parent / "smoke.py"

# A stand-in for `roqsim`: it exits 6, as `roqsim sim` does on a scene the run cut short, when the
# world's `# needs:` line asks for more steps than it was given, and 0 otherwise.
FAKE_ROQSIM = """\
import re, sys
world, steps = sys.argv[2], int(sys.argv[sys.argv.index("--steps") + 1])
needs = re.search(r"^# needs: (\\d+)$", open(world).read(), re.M)
sys.exit(6 if needs and int(needs.group(1)) > steps else 0)
"""


def _smoke(tmp_path, *worlds, roqsim=None, steps=200):
    if roqsim is None:
        fake = tmp_path / "roqsim"
        fake.write_text(f"#!{sys.executable}\n{FAKE_ROQSIM}")
        fake.chmod(0o755)
        roqsim = str(fake)
    argv = [sys.executable, str(SMOKE), "--roqsim", roqsim, "--steps", str(steps)]
    argv += ["--gitignore", str(tmp_path / "none"), "--worlds", *map(str, worlds)]
    return subprocess.run(argv, capture_output=True, text=True, cwd=tmp_path)


def _world(tmp_path, name, text):
    path = tmp_path / f"{name}.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_a_scene_that_finishes_within_the_steps_passes(tmp_path):
    world = _world(tmp_path, "short", "# needs: 150\ncomponents: []\n")
    done = _smoke(tmp_path, world)
    assert done.returncode == 0, done.stdout
    assert "none failed" in done.stdout


def test_a_scene_the_steps_cut_short_fails_and_says_how_to_fix_it(tmp_path):
    ok = _world(tmp_path, "short", "# needs: 150\n")
    long = _world(tmp_path, "long", "# needs: 500\n")
    done = _smoke(tmp_path, ok, long)
    assert done.returncode == 1
    assert f"{long}: its scene did not finish within 200 steps" in done.stdout
    assert str(ok) not in done.stdout.split("UNFINISHED", 1)[1]
    assert "# smoke-steps: <n>" in done.stdout


def test_a_world_gives_its_scene_the_steps_it_needs(tmp_path):
    world = _world(tmp_path, "long", "# needs: 500\n# smoke-steps: 600\n")
    done = _smoke(tmp_path, world)
    assert done.returncode == 0, done.stdout


@pytest.mark.parametrize("line", ["# smoke-steps: lots", "# smoke-steps: 0"])
def test_a_budget_that_is_no_step_count_fails_the_world(tmp_path, line):
    world = _world(tmp_path, "bad", f"{line}\n")
    done = _smoke(tmp_path, world)
    assert done.returncode == 1
    assert "is not a positive whole number" in done.stdout


def test_any_other_failure_fails_the_world_with_its_output(tmp_path):
    fake = tmp_path / "broken"
    fake.write_text(f"#!{sys.executable}\nimport sys\nprint('no such plugin')\nsys.exit(2)\n")
    fake.chmod(0o755)
    world = _world(tmp_path, "w", "components: []\n")
    done = _smoke(tmp_path, world, roqsim=str(fake))
    assert done.returncode == 1
    assert "no such plugin" in done.stdout
    assert "FAILED (1 worlds run)" in done.stdout


SCENE = """\
from roqsim.scenario import ScenarioPlugin


class Wait(ScenarioPlugin):
    def program(self):
        yield from self.wait(float(self.config["seconds"]))
"""


@pytest.mark.parametrize(("budget", "passes"), [("", False), ("# smoke-steps: 400\n", True)])
def test_roqsim_sim_on_a_scene_holds_the_same_rule(tmp_path, monkeypatch, budget, passes):
    """The real ``roqsim sim``: a 0.5 s scene at a 2 ms step needs 250 steps."""
    roqsim = Path(sys.executable).parent / "roqsim"
    if not roqsim.is_file():
        pytest.skip(f"no roqsim beside {sys.executable}")
    monkeypatch.setenv("ROQSIM_CONTROL", "none")
    (tmp_path / "scene.py").write_text(SCENE)
    body = """\
        sim: {timestep: 0.002, gravity: [0, 0, 0]}
        components:
          - {"scene.py:Wait": {seconds: 0.5, stop_at_end: true}, name: scene}
        """
    world = _world(tmp_path, "scene", budget + textwrap.dedent(body))
    done = _smoke(tmp_path, world, roqsim=str(roqsim))
    assert (done.returncode == 0) is passes, done.stdout
    if not passes:
        assert "its scene did not finish within 200 steps" in done.stdout


def test_worlds_given_as_arguments_run_as_those_of_worlds(tmp_path):
    fake = tmp_path / "roqsim"
    fake.write_text(f"#!{sys.executable}\n{FAKE_ROQSIM}")
    fake.chmod(0o755)
    ok = _world(tmp_path, "short", "# needs: 150\n")
    long = _world(tmp_path, "long", "# needs: 500\n")
    argv = [sys.executable, str(SMOKE), "--roqsim", str(fake), "--steps", "200"]
    argv += ["--gitignore", str(tmp_path / "none"), str(ok), str(long)]
    done = subprocess.run(argv, capture_output=True, text=True, cwd=tmp_path)
    assert done.returncode == 1, done.stdout
    assert f"== {ok}" in done.stdout
    assert f"{long}: its scene did not finish within 200 steps" in done.stdout
