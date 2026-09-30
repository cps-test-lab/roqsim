"""The `roqsim` command tree must stay complete, cheap, and free of parent-repo assumptions.

A convention that nothing checks decays: a run command in a checked-in Makefile names a binary that
does not exist and stays broken because nothing looks; a tool's --help grows to cost more tokens than
the rest of the tree put together because nothing measures it. So each rule here is a check.

The tests walk the repository rather than the installed distributions: they are about what this source
tree promises, and they must give the same answer in a bare clone as in a checkout nested inside
something larger.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import pytest

from roqsim import exit_status
from roqsim.commands import cli, load_groups, summary_line

REPO = Path(__file__).resolve().parents[2]

# Blender hosts these in its own interpreter, so `main(argv)` is on the far side of a subprocess and
# the module cannot be imported here at all. Listed rather than silently skipped: an exemption worth
# having is worth naming.
_FOREIGN_INTERPRETER = {"usd_to_scene.py"}

_ADDING_A_TOOL = """
Add it to the tree in the same commit that adds the tool -- see docs/developer_guide.rst,
"Adding a tool":

  1. move the logic to <pkg>/src/<pkg>/cli/<name>.py
  2. register it with one line in that package's group:
         group.add_command(tool("<pkg>.cli.<name>"))
  3. leave <pkg>/tools/<name>.py as the three-line wrapper onto it

You write no help text: the listing line is your docstring's first line, `--help` is your own
argparse, and `python -m pydoc <module>` prints the rest.
"""


@pytest.fixture(scope="module")
def tree() -> click.Group:
    load_groups(cli)
    return cli


def _commands(group: click.Group) -> dict[str, click.Command]:
    """Every leaf command in the tree, keyed by the path a user would type."""
    out: dict[str, click.Command] = {}
    for name, cmd in group.commands.items():
        if isinstance(cmd, click.Group):
            out.update({f"{name} {sub}": c for sub, c in _commands(cmd).items()})
        else:
            out[name] = cmd
    return out


def _tool_scripts() -> list[Path]:
    """The runnable scripts under every package's ``tools/`` dir."""
    return sorted(
        p for p in REPO.glob("roqsim*/tools/*.py") if "__main__" in p.read_text(encoding="utf-8")
    )


# -- the tree is complete -------------------------------------------------------------------------
def test_every_tool_script_is_reachable_as_a_command(tree):
    """A tool nobody can find is the failure this whole tree exists to prevent."""
    # A wrapper's filename usually matches its module, but need not: match the command name too, so
    # renaming a command does not read as a missing tool.
    registered = set()
    for path, cmd in _commands(tree).items():
        if hasattr(cmd, "module"):
            registered.add(cmd.module.rsplit(".", 1)[-1])
            registered.add(path.rsplit(" ", 1)[-1].replace("-", "_"))
    missing = [p for p in _tool_scripts() if p.stem not in registered]
    assert not missing, (
        "these tools are not in the `roqsim` command tree, so no --help will ever mention them:\n  "
        + "\n  ".join(str(p.relative_to(REPO)) for p in missing)
        + _ADDING_A_TOOL
    )


def test_tool_wrappers_stay_thin(tree):
    """A wrapper that grows logic splits the tool in two, and the copy nobody runs rots."""
    fat = []
    for p in _tool_scripts():
        body = [
            ln
            for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        tree_ = ast.parse(p.read_text(encoding="utf-8"))
        defines = any(
            isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) for n in tree_.body
        )
        if p.name in _FOREIGN_INTERPRETER:
            continue
        if len(body) > 12 or defines:
            fat.append(
                f"{p.relative_to(REPO)} ({len(body)} lines"
                f"{', defines functions' if defines else ''})"
            )
    assert not fat, (
        "tools/ holds wrappers, not implementations -- the logic belongs in the package so it is "
        "importable, testable and installed:\n  " + "\n  ".join(fat) + _ADDING_A_TOOL
    )


# -- the help stays cheap -------------------------------------------------------------------------
def _help_text(path: str) -> str:
    """`roqsim <path> --help` as a subprocess, because that is what a user actually runs."""
    return subprocess.run(
        [sys.executable, "-m", "roqsim.commands", *path.split(), "--help"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO,
    ).stdout


@pytest.fixture(scope="module")
def helps(tree) -> dict[str, str]:
    """Every leaf command's `--help`, captured once, concurrently.

    Each launch is a clean interpreter that imports its tool's module -- mujoco, scipy, Pillow,
    lxml, torch -- so it costs about a second, and there are ~38 of them. Paid serially inside one
    test this was 34 s, comfortably the most expensive thing in the repository and, because no test
    runner can split a single test, the floor on how fast the suite can go however many workers it
    gets. Threads are the right pool here: the work is in the children, not under the GIL.
    """
    cmds = [path for path, cmd in _commands(tree).items() if hasattr(cmd, "module")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(cmds, pool.map(_help_text, cmds), strict=True))


def test_no_command_dumps_its_whole_docstring_into_help(helps):
    """The docstring is the rationale; `--help` is how to run it.

    Bounding the *description* rather than the whole output is the point: a tool with thirty options
    legitimately prints thirty lines of option help, while a tool that hands argparse its entire
    module docstring prints an essay before the first flag. One of those is worth paying for.
    """
    over = {}
    for path, out in helps.items():
        # everything between the usage block and the first argument section is the description
        body = re.split(r"\n(?:positional arguments|options|Options):", out)[0]
        description = re.sub(r"^usage:.*?(?=\n\S|\Z)", "", body, flags=re.S).strip()
        if len(description) > 400:
            over[path] = len(description)
    assert not over, (
        "these commands print a description far longer than a synopsis -- pass "
        '`description=__doc__.split("\\n")[0]` and leave the rest to '
        f"`python -m pydoc <module>`: {over}"
    )


def test_every_usage_line_names_the_command_the_user_types(helps):
    """argparse takes its program name from argv[0], which the tree sets to the command path; a
    parser that names its own `prog` -- or runs in another interpreter -- has to say the same thing,
    or its usage line sends the reader off to spell an invocation that does not exist."""
    # The launcher's own spelling is fine too: under `python -m roqsim.commands` (which is how the
    # helps here are captured) argv[0] is that, and a parser naming `roqsim <path>` outright is right
    # under either launcher.
    wrong = {
        path: out.splitlines()[:2]
        for path, out in helps.items()
        if not re.search(
            rf"usage: (roqsim|python -m roqsim\.commands) {re.escape(path)}(\s|$)", out.lower()
        )
    }
    assert not wrong, f"these usage lines do not name `roqsim <path>`: {wrong}"


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_a_blender_hosted_tool_answers_help_without_blender(tmp_path, flag):
    """Asking how to run a tool must not need the program it runs in; running it still does."""
    env = {**os.environ, "PATH": str(tmp_path)}  # an empty directory: no `blender` on PATH
    cmd = [sys.executable, "-m", "roqsim.commands", "scenes", "usd-to-scene"]
    shown = subprocess.run(
        [*cmd, flag], capture_output=True, text=True, timeout=60, cwd=REPO, env=env
    )
    assert shown.returncode == 0, shown.stderr
    assert re.search(
        r"usage: (roqsim|python -m roqsim\.commands) scenes usd-to-scene(\s|$)", shown.stdout
    ), shown.stdout

    run = subprocess.run(
        [*cmd, "in.usd", "out", "name"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO,
        env=env,
    )
    assert run.returncode != 0
    assert "not on PATH" in run.stderr


def test_every_command_has_a_one_line_summary(tree):
    """The listing line comes from the docstring's first line, so that line has a job to do."""
    bad = {}
    for path, cmd in _commands(tree).items():
        if not hasattr(cmd, "module"):
            continue
        line = summary_line(cmd.module)
        if not line:
            bad[path] = "no module docstring"
        elif len(line) > 120:
            bad[path] = f"first line is {len(line)} chars; it is a summary, not a paragraph"
        elif "``" in line or "**" in line:
            bad[path] = "Sphinx markup in the summary, which a terminal shows verbatim"
    assert not bad, f"unusable summary lines: {bad}"


def test_listing_the_tree_does_not_import_the_tools(tree):
    """`roqsim --help` must not pay for what it only names.

    The tools import mujoco, scipy, Pillow and lxml, and torch reaches the tree through the policy
    packages. Reading each docstring from source keeps a listing at a tenth of the cost of importing
    the modules behind it -- and it is the only reason a Blender-hosted tool can appear in a listing
    at all, since importing that one raises ImportError outside Blender.
    """
    probe = (
        "import io, sys, contextlib\n"
        "import roqsim.commands as c\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"  # the listing itself is not the answer
        "    try: c.main(['scenes', '--help'])\n"
        "    except SystemExit: pass\n"
        "print('|'.join(m for m in ('torch', 'scipy', 'PIL', 'lxml') if m in sys.modules),"
        " file=sys.stderr)"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300, cwd=REPO
    )
    leaked = out.stderr.strip().splitlines()[-1] if out.stderr.strip() else ""
    assert not leaked, f"listing the tools imported them: {leaked}"


def test_the_listing_is_fast(tree):
    start = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "roqsim.commands", "--help"],
        capture_output=True,
        timeout=300,
        cwd=REPO,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"`roqsim --help` took {elapsed:.1f}s; it only lists names"


# -- one exit-status table ------------------------------------------------------------------------
#: Every tool whose --help must state its exit statuses. Its epilog comes from roqsim.exit_status.
_STATES_ITS_EXIT_STATUS = {
    "assets collision",
    "assets inspect-prop",
    "sim",
    "render",
    "state",
    "check",
    "health",
    "catalog",
    "plugins",
    "export web",
    "export capture",
    "export urdf",
    "export srdf",
    "export mesh",
    "export moveit",
    "scenes describe",
    "scenes inputs",
    "scenes floorplan-to-world",
    "scenes fuel-fetch",
    "scenes scene-to-floorplan",
    "scenes sdf-to-scene",
    "sensors coverage",
}

_NAMED_CODE = re.compile(r"(?:exit status: |; )(\d+) ")


def test_every_help_names_only_codes_from_the_table(tree, helps):
    """A script branches on the status, so a tool may only promise one the table defines, in the
    table's words -- a tool with its own meaning for 2 is the bug this table exists to prevent."""
    installed = {p for p in _STATES_ITS_EXIT_STATUS if p.split()[0] in tree.commands}
    assert installed <= set(helps), f"renamed or gone: {sorted(installed - set(helps))}"
    wrong, silent = {}, []
    for path, out in helps.items():
        text = " ".join(out.split())
        named = [int(c) for c in _NAMED_CODE.findall(text)] if "exit status:" in text else []
        if path in _STATES_ITS_EXIT_STATUS and not named:
            silent.append(path)
        bad = [c for c in named if f"{c} {exit_status.MEANINGS.get(c)}" not in text]
        if bad:
            wrong[path] = bad
    assert not wrong, f"--help names a status outside the table, or in other words: {wrong}"
    assert not silent, f"--help states no exit status (use exit_status.epilog): {silent}"


def _missing_input_cases(root: Path) -> dict[str, list[str]]:
    """One invocation per tool that names an input which is not there, and nothing else wrong."""
    nope = str(root / "nope")
    return {
        "assets reduce-mesh": [f"{nope}.glb", str(root / "out.obj")],
        "sim": [f"{nope}.yaml", "--headless"],
        "render": [f"{nope}.yaml", "--out", str(root / "x.png")],
        "render --state": ["--state", f"{nope}.npz", "--out", str(root / "x.png")],
        "state": ["--state", f"{nope}.npz", "--check"],
        "check": [f"{nope}.yaml"],
        "health": [nope],
        "catalog": ["model", "nope_xyz"],
        "plugins": ["describe", "nope_xyz"],
        "export web": ["--world", f"{nope}.yaml", "--out", str(root / "web")],
        "export web --mjcf": ["--mjcf", f"{nope}.xml", "--out", str(root / "web")],
        "export capture": ["--state", f"{nope}.npz", "--out", str(root / "cap")],
        "export urdf": ["--world", f"{nope}.yaml", "--out", str(root / "x.urdf")],
        "export urdf --mjcf": ["--mjcf", f"{nope}.xml", "--out", str(root / "x.urdf")],
        "export srdf --mjcf": [
            "--mjcf", f"{nope}.xml", "--urdf", f"{nope}.urdf", "--out", str(root / "x.srdf"),
            "--name", "x", "--arm-base", "a", "--arm-tip", "b", "--gripper-joint", "g",
            "--gripper-open", "0", "--gripper-close", "1",
        ],
        "export mesh": ["--world", f"{nope}.yaml", "--out", str(root / "x.stl")],
        "export moveit": ["--world", f"{nope}.yaml", "--out", str(root / "moveit")],
        "scenes describe": [f"{nope}.yaml"],
        "scenes inputs": [f"{nope}.yaml"],
        "scenes fuel-fetch": ["--world", f"{nope}.sdf"],
        "scenes sdf-to-scene": ["--world", f"{nope}.sdf", "--out-dir", str(root / "sdf")],
        "scenes scene-to-floorplan": ["--scene", nope],
        "scenes floorplan-to-world": [
            "--floorplan", f"{nope}.json", "--out-dir", str(root / "scene"),
            "--world-out", str(root / "w.yaml"), "--scene-name", "x",
        ],
        "sensors coverage": [
            "estimate", "--world", f"{nope}.yaml", "--placements", f"{nope}.json",
            "--out", str(root / "cov"),
        ],
    }  # fmt: skip


def test_a_missing_input_exits_with_the_one_bad_input_code(tree, tmp_path):
    """A named input that is not there is one status whichever tool was asked, never a crash's 1."""
    cases = {
        case: args
        for case, args in _missing_input_cases(tmp_path).items()
        if case.split()[0] in tree.commands
    }

    def run(case: str) -> tuple[int, str]:
        path = [word for word in case.split() if not word.startswith("--")]
        out = subprocess.run(
            [sys.executable, "-m", "roqsim.commands", *path, *cases[case]],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=tmp_path,
            env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")},
        )
        return out.returncode, out.stderr.strip().splitlines()[-1] if out.stderr.strip() else ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        got = dict(zip(cases, pool.map(run, cases), strict=True))
    wrong = {c: r for c, r in got.items() if r[0] != exit_status.BAD_INPUT}
    assert not wrong, f"a missing input must exit {exit_status.BAD_INPUT}: {wrong}"


# -- the repository stands alone ------------------------------------------------------------------
#: Instructions that only resolve inside some larger workspace and are a dead end in a bare clone.
#: An agent-harness skill path is the one shape worth matching literally; a *relative* path is not,
#: because `../../../roqsim_walker` from ros2_ws is a legitimate in-repo link. Absolute paths are
#: covered by `make check`, and no enclosing tree is named here on purpose -- a check for what to hide
#: should not itself be the list of it.
#:
#: The consumers this substrate happens to be run BY are also foreign, and that is a separate
#: argument from the one above: naming them is not a leak -- they are public -- but it makes a
#: standalone simulator read as a component of one particular stack, and it dates. A run harness is
#: one caller among several (a shell, a scenario runner, CI, an MCP client), so what is true of all
#: of them is what belongs here. `kubernetes` is the same mistake one layer down: this project is
#: headless or windowed, and where the container runs is not its business.
_FOREIGN = (".claude/skills", "robovast", "RoboVAST", "kubernetes", "Kubernetes")

#: The same argument for instructions addressed to tooling that ships elsewhere: an agent skill by any
#: name, and the gap records of an experiment specification. Matched as a class, so a renamed skill
#: is still caught; a reader of this tree can follow only roqsim's own commands and docs.
_FOREIGN_PATTERNS = (re.compile(r"\bskills?\b"), re.compile(r"resolution_attempt|\bspec gaps?\b"))

#: Lines allowed to spell a foreign name, and why. The rule above is about a path, an instruction or
#: a stack this project does not belong to; an identifier for a format someone else SPECIFIES is
#: none of those. It resolves fine in a bare clone, and renaming it to avoid the word would make this
#: writer's output unreadable by the very consumer whose specification defines it. Keep this list
#: short, per-line rather than per-file, and each entry justified -- an unexplained entry here is how
#: the check decays into a convention.
_FOREIGN_ALLOWED: dict[str, tuple[str, ...]] = {
    # The wire identifier a consumer matches on, and the citation saying where that format is
    # specified. Renaming either would not make this file standalone -- it would make its output
    # unreadable, and leave a reader unable to find the spec it implements.
    "roqsim/src/roqsim/export_capture.py": ('FORMAT = "robovast.run_capture"',
                                            "The format is defined by the consumer that reads it"),
    # Same: the pose table this writer fills in is somebody else's published contract.
    "roqsim/src/roqsim/capture.py": ("pose-table contract",),
}


# -- an input the tool cannot load is one sentence, not a traceback ---------------------------------


_FAKE_TOOL = """\
# A tool that cannot load what it was given.
def main(argv):
    from roqsim.plugin import PluginError

    if argv[0] == "missing":
        raise FileNotFoundError(2, "No such file or directory", "nosuch.json")
    if argv[0] == "unresolved":
        raise PluginError("world ref 'nosuch:world' names no known 'roqsim.worlds' provider")
    return 3
"""


@pytest.fixture
def fake_tool(tmp_path, monkeypatch):
    from roqsim.commands import tool

    (tmp_path / "fake_tool_for_test.py").write_text(_FAKE_TOOL)
    monkeypatch.syspath_prepend(str(tmp_path))
    return tool("fake_tool_for_test", "fake")


def _exit_code(cmd, args) -> int:
    with pytest.raises(SystemExit) as exc:
        cmd.main(args=args, prog_name="roqsim scenes fake", standalone_mode=False)
    return exc.value.code


def test_a_missing_input_file_is_one_sentence_naming_the_command(fake_tool, capsys):
    assert _exit_code(fake_tool, ["missing"]) == exit_status.BAD_INPUT
    assert capsys.readouterr().err == "roqsim scenes fake: No such file or directory: nosuch.json\n"


def test_an_input_that_does_not_resolve_is_one_sentence(fake_tool, capsys):
    assert _exit_code(fake_tool, ["unresolved"]) == exit_status.BAD_INPUT
    err = capsys.readouterr().err
    assert err.startswith("roqsim scenes fake: world ref 'nosuch:world' names no known")
    assert "Traceback" not in err


def test_verbose_keeps_the_traceback(fake_tool):
    from roqsim.plugin import PluginError

    with pytest.raises(PluginError):
        fake_tool.main(args=["unresolved", "-v"], standalone_mode=False)


def test_a_tools_own_exit_code_passes_through(fake_tool):
    assert _exit_code(fake_tool, ["fine"]) == 3


def test_nothing_here_names_a_repository_that_may_not_exist():
    """This tree is developed both on its own and nested inside a larger one.

    A path or a name that only resolves in the larger case is a trap for whoever has the smaller one:
    it reads as instruction and cannot be followed. Package boundaries are the same argument one level
    down, which is why this rule is in the project's own CLAUDE.md.
    """
    offenders = []
    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=REPO).stdout
    # This file necessarily spells every token it searches for.
    self_rel = Path(__file__).resolve().relative_to(REPO).as_posix()
    for rel in tracked.split():
        if not rel.endswith((".py", ".md", ".rst", ".toml", ".yaml", ".yml", ".cfg")):
            continue
        if rel == self_rel:
            continue
        path = REPO / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        allowed = _FOREIGN_ALLOWED.get(rel, ())
        for number, line in enumerate(text.splitlines(), 1):
            if any(fragment in line for fragment in allowed):
                continue
            for token in _FOREIGN:
                if token in line:
                    offenders.append(f"{rel}:{number} names {token!r}")
            for pattern in _FOREIGN_PATTERNS:
                if match := pattern.search(line):
                    offenders.append(f"{rel}:{number} names {match.group()!r}")
    assert not offenders, (
        "these files assume a surrounding repository that a standalone clone does not have:\n  "
        + "\n  ".join(offenders)
    )
