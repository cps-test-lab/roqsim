"""The review_scene_by_human tool: fails loudly on a missing scene file, returns the verdict the
window wrote, surfaces a closed-without-verdict window, times out, and is registered as an MCP tool.

The window subprocess is faked here so the tests need no display."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from roqsim_scene_builder.scene_review import review_scene_by_human

from roqsim_scene_builder import window_runner


class _FakePopen:
    """Stands in for the window subprocess. ``behaviour`` decides what communicate() does."""

    def __init__(self, cmd, behaviour="pass", **kwargs):
        self.cmd = cmd
        self.behaviour = behaviour
        self.returncode = 0

    def _json_out(self) -> Path:
        return Path(self.cmd[self.cmd.index("--json-out") + 1])

    def communicate(self, timeout=None):
        # Only the first, timed call raises; the post-kill cleanup call (timeout=None) returns.
        if self.behaviour == "timeout" and timeout is not None:
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        if self.behaviour == "no_verdict":
            self.returncode = 3
            return ("", "window closed\n")
        payload = {"verdict": "pass", "comment": "looks good", "annotations": []}
        if self.behaviour == "moved":
            payload["moves"] = [
                {
                    "entity": "industrial_table_1",
                    "model": "industrial_table",
                    "pos": [1.0, 2.0, 0.0],
                    "yaw_deg": 90.0,
                }
            ]
        self._json_out().write_text(json.dumps(payload))
        self.returncode = 0
        return ("", "")

    def kill(self):
        pass


def _patch_popen(monkeypatch, behaviour):
    monkeypatch.setattr(
        window_runner.subprocess,
        "Popen",
        lambda cmd, **kw: _FakePopen(cmd, behaviour=behaviour, **kw),
    )


def test_missing_scene_file_raises():
    with pytest.raises(FileNotFoundError):
        review_scene_by_human("/no/such/world.yaml")


def test_returns_window_verdict(monkeypatch):
    _patch_popen(monkeypatch, "pass")
    result = review_scene_by_human("roqsim_assets:industrial_table", message="ok?")
    assert result == {"verdict": "pass", "comment": "looks good", "annotations": []}


def test_returns_prop_moves(monkeypatch):
    _patch_popen(monkeypatch, "moved")
    result = review_scene_by_human("roqsim_scenes:depot", message="tidy up")
    assert result["moves"] == [
        {
            "entity": "industrial_table_1",
            "model": "industrial_table",
            "pos": [1.0, 2.0, 0.0],
            "yaw_deg": 90.0,
        }
    ]


def test_no_verdict_raises_runtime_error(monkeypatch):
    _patch_popen(monkeypatch, "no_verdict")
    with pytest.raises(RuntimeError, match="no verdict"):
        review_scene_by_human("roqsim_assets:industrial_table")


def test_timeout_raises(monkeypatch):
    _patch_popen(monkeypatch, "timeout")
    with pytest.raises(TimeoutError):
        review_scene_by_human("roqsim_assets:industrial_table", timeout_s=0.01)


def test_review_scene_threads_title_only_when_set(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        window_runner.subprocess,
        "Popen",
        lambda cmd, **kw: seen.__setitem__("cmd", cmd) or _FakePopen(cmd, behaviour="pass", **kw),
    )
    review_scene_by_human("roqsim_assets:industrial_table", message="ok?", title="Office table")
    assert "--title" in seen["cmd"] and "Office table" in seen["cmd"]
    review_scene_by_human("roqsim_assets:industrial_table", message="ok?")  # no title -> flag omitted
    assert "--title" not in seen["cmd"]


def test_review_scene_threads_focus_object_only_when_set(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        window_runner.subprocess,
        "Popen",
        lambda cmd, **kw: seen.__setitem__("cmd", cmd) or _FakePopen(cmd, behaviour="pass", **kw),
    )
    review_scene_by_human("roqsim_scenes:depot", focus_object="industrial_table_1")
    assert "--focus-object" in seen["cmd"] and "industrial_table_1" in seen["cmd"]
    review_scene_by_human("roqsim_scenes:depot")  # no focus -> flag omitted, default camera
    assert "--focus-object" not in seen["cmd"]


def test_sketch_floorplan_threads_title_only_when_set(monkeypatch):
    from roqsim_scene_builder.floorplan_sketch import sketch_floorplan_by_human

    seen: dict = {}
    monkeypatch.setattr(
        window_runner.subprocess,
        "Popen",
        lambda cmd, **kw: seen.__setitem__("cmd", cmd) or _FakePopen(cmd, behaviour="pass", **kw),
    )
    sketch_floorplan_by_human(message="draw it", title="Apartment")
    assert "--title" in seen["cmd"] and "Apartment" in seen["cmd"]
    sketch_floorplan_by_human(message="draw it")  # no title -> the flag is omitted
    assert "--title" not in seen["cmd"]


def test_registered_as_mcp_tool():
    import anyio
    from roqsim_scene_builder.server import create_server

    mcp = create_server()
    tools = anyio.run(mcp.list_tools)
    assert "review_scene_by_human" in {t.name for t in tools}
