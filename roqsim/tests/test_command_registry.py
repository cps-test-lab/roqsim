"""The `roqsim` command tree must stay complete, cheap, and free of parent-repo assumptions.

Every rule here was a convention first and decayed anyway. The run command in the checked-in scene
Makefiles named a binary that never existed and a flag that had been removed, and it stayed broken
because nothing looked; one tool's --help cost more tokens than the rest of the tree put together
because nothing measured it. A convention that is not checked is a comment about the past, so each of
these is a check.

The tests walk the repository rather than the installed distributions: they are about what this source
tree promises, and they must give the same answer in a bare clone as in a checkout nested inside
something larger.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
from pathlib import Path

import click
import pytest

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
def test_no_command_dumps_its_whole_docstring_into_help(tree):
    """The docstring is the rationale; `--help` is how to run it.

    Bounding the *description* rather than the whole output is the point: a tool with thirty options
    legitimately prints thirty lines of option help, while a tool that hands argparse its entire
    module docstring prints an essay before the first flag. One of those is worth paying for.
    """
    over = {}
    for path, cmd in _commands(tree).items():
        if not hasattr(cmd, "module"):
            continue
        out = subprocess.run(
            [sys.executable, "-m", "roqsim.commands", *path.split(), "--help"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=REPO,
        ).stdout
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


# -- the repository stands alone ------------------------------------------------------------------
#: Instructions that only resolve inside some larger workspace and are a dead end in a bare clone.
#: An agent-harness skill path is the one shape worth matching literally; a *relative* path is not,
#: because `../../../roqsim_walker` from ros2_ws is a legitimate in-repo link. Absolute paths are
#: covered by `make check`, and no enclosing tree is named here on purpose -- a check for what to hide
#: should not itself be the list of it.
#:
#: `robovast` is deliberately absent: it is a published project
#: (github.com/cps-test-lab/robovast), so naming it points a reader at something real, which is the
#: opposite of what this check is for.
_FOREIGN = (".claude/skills",)

#: Lines allowed to spell a foreign name, and why. The rule above is about a path or an instruction
#: that only resolves inside the larger tree; an identifier for a format someone else SPECIFIES is
#: neither. It resolves fine in a bare clone, and renaming it to avoid the word would make this
#: writer's output unreadable by the very consumer whose specification defines it. Keep this list
#: short, per-line rather than per-file, and each entry justified -- an unexplained entry here is how
#: the check decays into the convention it replaced.
_FOREIGN_ALLOWED: dict[str, tuple[str, ...]] = {}


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
    assert not offenders, (
        "these files assume a surrounding repository that a standalone clone does not have:\n  "
        + "\n  ".join(offenders)
    )
