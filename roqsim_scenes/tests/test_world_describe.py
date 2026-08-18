"""`roqsim scenes describe`: what a world provides, for a caller that is not a roqsim process.

The pairing to keep in mind is with `roqsim scenes inputs`, which answers what a world *needs*.
A campaign runner uses both and has neither roqsim nor a way to resolve an `extends` chain
without it.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from roqsim_scenes.cli import world_describe


def _describe(capsys, *argv):
    assert world_describe.main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def world(tmp_path):
    """A world whose one plugin is addressed by its ref, with an entity name in its config."""
    path = tmp_path / "w.yaml"
    path.write_text(
        "sim: {timestep: 0.01}\n"
        "plugins:\n"
        "- boxes:\n"
        "    name: obstacles\n"
        "    instances:\n"
        "    - {pos: [1.0, 1.0], size: [0.4, 0.4, 0.4]}\n"
        "    - {pos: [3.0, 2.0], size: [0.4, 0.4, 0.4]}\n")
    return path


def test_the_override_key_is_the_plugin_ref_not_a_config_name(capsys, world):
    """Two different `name`s, and confusing them is a run-time failure.

    `apply_overrides` resolves a plugin by its reserved `name:` SIBLING, then by its ref. A
    `name:` inside the plugin's own config -- what `boxes` calls its entities -- addresses
    nothing, so an override written against it is refused inside the container.
    """
    plugin = _describe(capsys, str(world))["plugins"][0]
    assert plugin["key"] == "boxes"
    assert plugin["ref"] == "boxes"
    assert plugin["name"] is None


def test_a_reserved_name_sibling_becomes_the_key(capsys, tmp_path):
    path = tmp_path / "named.yaml"
    path.write_text(
        "plugins:\n"
        "- boxes:\n"
        "    instances: []\n"
        "  name: obstacles\n")
    plugin = _describe(capsys, str(path))["plugins"][0]
    assert (plugin["key"], plugin["ref"]) == ("obstacles", "boxes")
    assert "plugins.obstacles.instances" in plugin["paths"]


def test_it_reports_the_paths_that_exist(capsys, world):
    described = _describe(capsys, str(world))
    assert "plugins.boxes.instances" in described["plugins"][0]["paths"]


def test_a_lists_members_are_not_addressable_paths(capsys, world):
    """A campaign overrides the list; it does not address instance 7's y coordinate."""
    paths = _describe(capsys, str(world))["plugins"][0]["paths"]
    assert not any(p.startswith("plugins.boxes.instances.") for p in paths)


def test_entities_are_absent_until_asked_for(capsys, world):
    """Naming them means compiling the model, which a caller resolving paths should not pay."""
    assert _describe(capsys, str(world))["entities"] is None


def test_entities_are_the_ones_the_world_compiles(capsys, world):
    described = _describe(capsys, str(world), "--entities")
    assert described["entities"] == ["obstacles_0", "obstacles_1"]


def test_inputs_are_reported_too(capsys, world):
    """The staging question and the override question answered by one container run."""
    assert str(world.resolve()) in _describe(capsys, str(world))["inputs"]


def test_a_missing_world_fails_rather_than_reporting_nothing(capsys, tmp_path):
    assert world_describe.main([str(tmp_path / "nope.yaml")]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_an_unresolvable_package_ref_names_itself(capsys):
    assert world_describe.main(["no_such_pkg:world"]) == 1
    assert "no_such_pkg:world" in capsys.readouterr().err


def test_the_overridable_fields_need_no_model(capsys, world):
    """The allowlist is a property of MuJoCo, not of this world, so it is always here and free."""
    overridable = _describe(capsys, str(world))["overridable"]
    fields = {row["field"]: row for row in overridable["fields"]}
    assert "geom_friction" in fields
    # What an agent is handed: the physical effect, and the way it can silently do nothing.
    assert fields["geom_friction"]["namespace"] == "geom"
    assert "element-wise MAXIMUM" in fields["geom_friction"]["caveats"]
    assert overridable["targets"] is None, "naming targets needs the model; do not pay for it here"


def test_overridable_targets_report_current_values(capsys, world):
    """The half that stops a caller inventing a name: what exists, and what it is right now."""
    targets = _describe(capsys, str(world), "--overridable", "obstacles_0*")["overridable"]["targets"]
    geoms = targets["geom"]
    assert [g["name"] for g in geoms] == ["obstacles_0_box"]
    assert geoms[0]["geom_friction"] == pytest.approx([1.0, 0.005, 0.0001])
    # Read-only context, and the thing geom_friction's caveat is about.
    assert geoms[0]["geom_priority"] == 0
    assert geoms[0]["body"] == "obstacles_0_box"


def test_a_geom_is_not_named_after_its_entity(capsys, world):
    """Why this exists at all: `obstacles_0` is the ENTITY, `obstacles_0_box` is the geom.

    A caller writing `select: [obstacles_0]` from the entity list would resolve nothing -- which the
    plugin refuses loudly, but only after an image pull. This is where it finds the real name.
    """
    described = _describe(capsys, str(world), "--entities", "--overridable", "*")
    assert "obstacles_0" in described["entities"]
    assert "obstacles_0" not in [g["name"] for g in described["overridable"]["targets"]["geom"]]


def test_the_glob_is_what_bounds_the_answer(capsys, world):
    """A caller after one prop must not be handed the scene: this world has walls and a floor too."""
    everything = _describe(capsys, str(world), "--overridable", "*")["overridable"]["targets"]
    one = _describe(capsys, str(world), "--overridable", "obstacles_1*")["overridable"]["targets"]
    assert len(one["geom"]) == 1
    assert len(everything["geom"]) > len(one["geom"])


def test_asking_for_both_builds_the_world_once(capsys, world, monkeypatch):
    """Compiling is the expensive part, so --entities and --overridable must share one build."""
    builds = []
    original = world_describe._built

    @contextmanager
    def counted(config):
        builds.append(1)
        with original(config) as ctx:
            yield ctx

    monkeypatch.setattr(world_describe, "_built", counted)
    described = _describe(capsys, str(world), "--entities", "--overridable", "*")
    assert described["entities"] == ["obstacles_0", "obstacles_1"]
    assert described["overridable"]["targets"]["geom"]
    assert builds == [1]


@pytest.fixture
def dummy_world(tmp_path):
    """Two named `dummy` instances, each building a body `<name>_box` with one geom.

    Deliberately not the `world` fixture above: `boxes`/`box` need `roqsim_assets` on the
    entry-point path, which not every dev environment has installed, while `dummy` is
    core `roqsim`'s own always-registered plugin -- so these tests hold regardless.
    """
    path = tmp_path / "dummy_world.yaml"
    path.write_text(
        "plugins:\n"
        "- dummy:\n"
        "    size: 0.3\n"
        "  name: box_a\n"
        "- dummy:\n"
        "    size: 0.2\n"
        "  name: box_b\n")
    return path


def test_body_tree_is_absent_until_asked_for(capsys, dummy_world):
    """Same discipline as entities/overridable: nobody pays for a build they didn't ask for."""
    assert _describe(capsys, str(dummy_world))["body_tree"] is None


def test_body_tree_nests_the_geom_under_its_body(capsys, dummy_world):
    """The point of this flag: a geom shows up *under* its body, not beside it in a flat list."""
    matches = _describe(capsys, str(dummy_world), "--body-tree", "box_a_box")["body_tree"]
    assert [m["root"] for m in matches] == ["box_a_box"]
    match = matches[0]
    assert match["truncated"] is False
    assert match["tree"]["name"] == "box_a_box"
    assert match["tree"]["type"] == "body"
    children = match["tree"]["children"]
    # dummy's own geom and free joint are unnamed (dummy.py never names either) -- still
    # shown, since the tree is about structure ("what's nested here"), not addressability.
    assert {c["type"] for c in children} == {"geom", "joint"}
    assert all(c["name"] == "" for c in children)
    assert all("children" not in c for c in children), "a leaf has no children of its own"


def test_body_tree_glob_bounds_the_answer(capsys, dummy_world):
    """Same rule --overridable already follows: a caller after one body is not handed every body."""
    one = _describe(capsys, str(dummy_world), "--body-tree", "box_a_box")["body_tree"]
    both = _describe(capsys, str(dummy_world), "--body-tree", "box_*_box")["body_tree"]
    assert len(one) == 1
    assert len(both) == 2


def test_body_tree_truncates_rather_than_returning_a_partial_tree_silently(
        capsys, dummy_world, monkeypatch):
    """The regression this flag exists to prevent: a broad glob must never risk "the scene"."""
    monkeypatch.setattr(world_describe, "_MAX_TREE_NODES", 1)
    matches = _describe(capsys, str(dummy_world), "--body-tree", "box_a_box")["body_tree"]
    match = matches[0]
    assert match["truncated"] is True
    # The one-node budget is spent on the body itself; its geom must not appear.
    assert "children" not in match["tree"]


def test_body_tree_shares_the_one_build_with_entities_and_overridable(
        capsys, dummy_world, monkeypatch):
    """Compiling is the expensive part -- asking for all three must still build only once."""
    builds = []
    original = world_describe._built

    @contextmanager
    def counted(config):
        builds.append(1)
        with original(config) as ctx:
            yield ctx

    monkeypatch.setattr(world_describe, "_built", counted)
    described = _describe(
        capsys, str(dummy_world), "--entities", "--overridable", "*",
        "--body-tree", "box_a_box")
    assert described["entities"] == ["box_a", "box_b"]
    assert described["overridable"]["targets"]["geom"]
    assert described["body_tree"]
    assert builds == [1]


@pytest.fixture
def ros_world(tmp_path):
    """A `*_ros` world: geometry plus the two bridges that ship in a colcon package.

    Which is to say: exactly what a pip-only environment cannot resolve, and the shape the campaign
    pre-check was silently losing its answer to.
    """
    path = tmp_path / "dummy_ros.yaml"
    path.write_text(
        "plugins:\n"
        "- dummy:\n"
        "    size: 0.3\n"
        "  name: box_a\n"
        "- ros2_bridge: {}\n"
        "- sim_interfaces: {}\n")
    return path


def test_a_ros_world_still_answers_where_the_bridge_does_not_resolve(capsys, ros_world):
    """The reported failure: the build died on plugins that contribute no geometry."""
    assert world_describe.main([str(ros_world), "--entities"]) == 0
    out = capsys.readouterr()
    described = json.loads(out.out)
    assert described["entities"] == ["box_a"]
    assert described["dropped_transport"] == ["ros2_bridge", "sim_interfaces"]
    assert described["errors"] is None
    # Stated out loud too: what a caller reads as "the scene" was built without its transport.
    assert "dropped ros2_bridge, sim_interfaces" in out.err


def test_the_dropped_bridge_is_still_a_reported_plugin(capsys, ros_world):
    """It goes from the BUILD, not from the answer: `plugins.ros2_bridge.*` is a legal override."""
    keys = [p["key"] for p in _describe(capsys, str(ros_world), "--entities")["plugins"]]
    assert keys == ["box_a", "ros2_bridge", "sim_interfaces"]


def test_a_misspelt_geometry_plugin_still_fails_loudly(capsys, tmp_path):
    """The regression guard on which drop helper this uses.

    The lenient one drops any ref that will not resolve, so this world would build without its
    geometry and report an entity list missing `box_a` -- from which a caller concludes the world
    does not have it. An unresolvable ref that is not identifiable as transport must stay fatal.
    """
    path = tmp_path / "typo.yaml"
    path.write_text("plugins:\n- dummmy:\n    size: 0.3\n  name: box_a\n")
    assert world_describe.main([str(path), "--entities"]) == 1
    out = capsys.readouterr()
    assert "dummmy" in out.err
    assert json.loads(out.out.splitlines()[-1])["errors"]["build"]


def test_a_build_failure_keeps_the_half_that_needed_no_build(capsys, dummy_world, monkeypatch):
    """A build-only failure used to discard the plugin list too, which cost the campaign its check."""
    @contextmanager
    def explode(_config):
        raise RuntimeError("mesh is not where it says it is")
        yield  # pragma: no cover - unreachable, and what makes this a context manager

    monkeypatch.setattr(world_describe, "_built", explode)
    assert world_describe.main([str(dummy_world), "--entities"]) == 1, \
        "a partial answer is not a success"
    out = capsys.readouterr()
    described = json.loads(out.out.splitlines()[-1])
    assert [p["key"] for p in described["plugins"]] == ["box_a", "box_b"]
    assert described["entities"] is None, "what needed the build is absent, not guessed"
    assert "mesh is not where it says it is" in described["errors"]["build"]
    assert "cannot build world" in out.err


def test_a_world_that_cannot_load_reports_why(capsys, tmp_path):
    """An unresolvable `extends` is a load failure; an unknown plugin ref is not.

    Plugin refs resolve at build, so describing paths deliberately does not need them -- which
    is what lets this answer for a world whose plugins live in another package.
    """
    bad = tmp_path / "bad.yaml"
    bad.write_text("extends: ./nowhere.yaml\nplugins: []\n")
    assert world_describe.main([str(bad)]) == 1
    out = capsys.readouterr()
    assert "cannot load world" in out.err
    # Unlike a build failure there is no half to hand back: nothing was resolved.
    assert out.out == ""
