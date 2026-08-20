"""``render_scene``: the MCP wrapper's contract. It shells out, so these tests stub the subprocess."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from roqsim_scene_builder import scene_render


@pytest.fixture
def fake_rst(monkeypatch, tmp_path):
    """Capture the argv `render_scene` would run, and answer with a plausible JSON record."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["env"] = kwargs.get("env", {})
        out = Path(argv[argv.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x89PNG\r\n\x1a\n")
        record = {"path": str(out), "width": 960, "height": 540, "nbody": 1, "ngeom": 1}
        return subprocess.CompletedProcess(argv, 0, json.dumps(record) + "\n", "")

    monkeypatch.setattr(scene_render, "_rst", lambda: "/fake/roqsim")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


@pytest.fixture
def world(tmp_path):
    """A real file, because render_scene checks up front that a path-shaped target exists."""
    path = tmp_path / "w.yaml"
    path.write_text("plugins: []\n")
    return str(path)


def test_it_shells_out_to_the_renderer(fake_rst, tmp_path, world):
    """Not an in-process import: a long-lived MCP server must not hold or leak a GL context."""
    scene_render.render_scene(world, out=str(tmp_path / "x.png"))
    argv = fake_rst["argv"]
    assert argv[0] == "/fake/roqsim" and argv[1] == "render" and world in argv


def test_a_path_is_returned_not_an_image_by_default(fake_rst, tmp_path, world):
    """The default costs ~40 tokens; the image's ~700 are spent only when a caller asks."""
    record = scene_render.render_scene(world, out=str(tmp_path / "x.png"))
    assert "image" not in record
    assert Path(record["path"]).exists()


def test_inline_returns_the_image_as_its_own_content_block(fake_rst, tmp_path, world):
    """An Image inside the returned dict is not an image: it is serialized like any other value, so the
    picture reached the caller as an object repr. It has to be a content block of its own."""
    result = scene_render.render_scene(world, out=str(tmp_path / "x.png"), inline=True)
    assert [block.type for block in result.content] == ["image"]
    assert result.structured_content["path"] == str(tmp_path / "x.png")


def test_inline_keeps_the_record_structured(fake_rst, tmp_path, world):
    """Through the real tool wrapper, because that is where it broke: an Image is not JSON-serializable,
    so a dict holding one lost its structured content while the tool still advertised an output schema,
    and the client rejected a render that had in fact succeeded."""
    import asyncio

    from fastmcp.tools import FunctionTool

    tool = FunctionTool.from_function(scene_render.render_scene)
    result = asyncio.run(
        tool.run({"target": world, "out": str(tmp_path / "x.png"), "inline": True})
    )
    assert [block.type for block in result.content] == ["image"]
    assert result.structured_content["width"] == 960


def test_flags_are_forwarded(fake_rst, tmp_path, world):
    scene_render.render_scene(
        world,
        out=str(tmp_path / "x.png"),
        size="640x360",
        view=["elevation=-85", "distance=40"],
        focus="robot",
        no_ceiling=True,
    )
    argv = fake_rst["argv"]
    assert "--size" in argv and "640x360" in argv
    assert argv[argv.index("--view") + 1 : argv.index("--view") + 3] == [
        "elevation=-85",
        "distance=40",
    ]
    assert "--focus" in argv and "robot" in argv
    assert "--no-ceiling" in argv


def test_recording_flags_are_forwarded(fake_rst, tmp_path):
    recording = tmp_path / "run.npz"
    recording.write_bytes(b"x")
    scene_render.render_scene(state=str(recording), at=12.5, out=str(tmp_path / "x.png"))
    argv = fake_rst["argv"]
    assert "--state" in argv and str(recording) in argv
    assert argv[argv.index("--at") + 1] == "12.5"


def test_a_target_is_optional_with_state(fake_rst, tmp_path):
    """The recording names its own world, so a caller need not repeat it."""
    recording = tmp_path / "run.npz"
    recording.write_bytes(b"x")
    scene_render.render_scene(state=str(recording), out=str(tmp_path / "x.png"))
    assert fake_rst["argv"][1] == "render"  # nothing was inserted as a positional target


def test_neither_target_nor_state_is_refused():
    with pytest.raises(ValueError, match="give a target"):
        scene_render.render_scene()


def test_a_missing_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file"):
        scene_render.render_scene(str(tmp_path / "absent.yaml"))
    with pytest.raises(FileNotFoundError, match="no such file"):
        scene_render.render_scene(state=str(tmp_path / "absent.npz"))


def test_a_model_reference_is_not_checked_for_existence(fake_rst, tmp_path):
    """`pkg:name` has no suffix; resolving it is `roqsim render`'s job, not a stat here."""
    scene_render.render_scene("roqsim_assets:industrial_table", out=str(tmp_path / "x.png"))
    assert "roqsim_assets:industrial_table" in fake_rst["argv"]


def test_a_failure_passes_through_the_renderers_own_message(monkeypatch, tmp_path, world):
    """Its message names the missing camera or the GL backend to set; paraphrasing loses that."""
    monkeypatch.setattr(scene_render, "_rst", lambda: "/fake/roqsim")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv,
            2,
            "",
            "roqsim render: --camera 'nope': this world declares no such camera. It has: head.",
        ),
    )
    with pytest.raises(RuntimeError, match="no such camera"):
        scene_render.render_scene(world, camera="nope", out=str(tmp_path / "x.png"))


def test_an_offscreen_backend_is_defaulted_but_never_overridden(
    fake_rst, tmp_path, monkeypatch, world
):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    scene_render.render_scene(world, out=str(tmp_path / "a.png"))
    assert fake_rst["env"]["MUJOCO_GL"] == scene_render._DEFAULT_GL

    monkeypatch.setenv("MUJOCO_GL", "osmesa")  # a GPU-less host's own choice must survive
    scene_render.render_scene(world, out=str(tmp_path / "b.png"))
    assert fake_rst["env"]["MUJOCO_GL"] == "osmesa"


def test_it_is_registered_on_the_server():
    """`get_tool` is a coroutine, so awaiting it is the only way this asserts anything."""
    import asyncio

    from roqsim_scene_builder.server import create_server

    tool = asyncio.run(create_server().get_tool("render_scene"))
    assert tool is not None
    assert "render_scene" in getattr(tool, "name", "render_scene")
