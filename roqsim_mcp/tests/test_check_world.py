# SPDX-License-Identifier: Apache-2.0
"""``check_world``: ``roqsim check --json`` as an MCP tool -- same input, same report, run out of process."""

from __future__ import annotations

import asyncio
import json
import subprocess
import textwrap

import pytest
from roqsim_mcp.mcp_server import create_server

from roqsim_mcp import check as check_mod


def _call(args: dict):
    return asyncio.run(create_server().call_tool("check_world", args))


def test_a_world_that_loads_comes_back_ok(tmp_path):
    """Through the real command: the report is roqsim check's own."""
    world = tmp_path / "w.yaml"
    world.write_text(
        textwrap.dedent("""
        # the default room, nothing in it
        sim: {}
        components: []
    """).lstrip()
    )
    report = json.loads(_call({"world": str(world)}).content[0].text)
    assert report["ok"] is True and report["reached"] == "reset"


def test_a_world_that_does_not_resolve_is_a_report_not_a_tool_failure():
    """ok=false is an answer: the problem is named with its stage and hint, as the CLI names it."""
    report = json.loads(_call({"world": "no_such_pkg_xyz:world"}).content[0].text)
    assert report["ok"] is False
    assert report["problems"][0]["stage"] == "resolve"
    assert "no_such_pkg_xyz" in report["problems"][0]["message"]


def test_it_runs_the_command_out_of_process(monkeypatch):
    """A plugin that prints must not reach the stdio stream this server speaks on."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, json.dumps({"ok": True}), "")

    monkeypatch.setattr(check_mod.subprocess, "run", fake_run)
    assert check_mod.check_world("roqsim_scenes:depot") == {"ok": True}
    assert seen["argv"][1:] == ["-m", "roqsim.check", "roqsim_scenes:depot", "--json"]


def test_a_check_that_could_not_run_raises_with_the_commands_own_line(monkeypatch):
    monkeypatch.setattr(
        check_mod.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(
            argv, 2, "", "usage: roqsim check ...\nroqsim check: error: unrecognized arguments"
        ),
    )
    with pytest.raises(RuntimeError, match="unrecognized arguments"):
        check_mod.check_world("w.yaml")


def test_an_empty_world_is_refused_naming_the_parameter():
    with pytest.raises(ValueError, match="^world:"):
        check_mod.check_world("  ")


def test_its_input_is_the_clis():
    """One required string, named as roqsim check's positional is."""

    async def _schema():
        return (await create_server().get_tool("check_world")).parameters

    schema = asyncio.run(_schema())
    assert schema["required"] == ["world"]
    assert set(schema["properties"]) == {"world"}
