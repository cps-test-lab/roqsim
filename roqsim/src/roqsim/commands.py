"""The ``roqsim`` command tree: one root command, one group per package that ships tools.

``roqsim`` is the only name anyone has to know. Everything else is two ``--help``s away::

    roqsim --help                          # the groups
    roqsim scenes --help                   # one line per tool in that group
    roqsim scenes sdf-to-scene --help      # that tool's own options
    python -m pydoc roqsim_scenes.cli.sdf_to_scene    # the full rationale behind it

Groups are contributed through the ``roqsim.commands`` entry-point group, exactly as plugins, models and
worlds are. The core never names a package that might provide one: a package declares its own group,
and installing it is what makes the group appear. That is what lets a deployment package outside this
repository add commands without changing anything here.

Help text is never written twice. A tool keeps the ``argparse`` parser it already has, and the command
wrapping it is a pass-through: ``--help`` reaches the tool's own parser, so its options are declared in
exactly one place. The line shown in the group listing is the first line of the tool's module
docstring. Authors write a docstring and a parser; nothing else.

Nothing here imports a tool until it has to. Building the tree only records module paths, so ``roqsim
--help`` stays cheap no matter how heavy the tools are -- a listing costs its own group's imports and
no other's, and running one tool never pays for its siblings.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import inspect
import shutil
import subprocess
import sys
from functools import cache
from importlib.metadata import entry_points
from pathlib import Path

import click

COMMAND_GROUP = "roqsim.commands"


@cache
def module_docstring(module: str) -> str:
    """A module's docstring, read from its source **without importing it**.

    Listing a group must not run any tool's imports. Two reasons, one of them fatal: the tools pull
    in mujoco, scipy, Pillow and lxml, which turns a listing into most of a second; and some cannot
    be imported here at all -- the Blender-hosted ones start with ``import bpy``, which exists only
    inside Blender, so importing to reach a docstring would take the whole listing down with it.

    Returns an empty string when the source cannot be found, since a missing summary is worth a blank
    line in a listing, never an exception.
    """
    try:
        spec = importlib.util.find_spec(module)  # locates, does not execute
    except (ImportError, ValueError):
        return ""
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        return ""
    try:
        return ast.get_docstring(ast.parse(Path(spec.origin).read_text(encoding="utf-8"))) or ""
    except (OSError, SyntaxError):
        return ""


def _takes_argv(main) -> bool:
    """True when ``main`` accepts an argv list (as opposed to reading ``sys.argv`` itself)."""
    try:
        params = inspect.signature(main).parameters
    except (TypeError, ValueError):  # a builtin or a C callable: assume the sys.argv convention
        return False
    return any(
        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        for p in params.values()
    )


def summary_line(module: str) -> str:
    """The first line of a module's docstring: what a group listing shows for that tool."""
    doc = module_docstring(module).strip()
    return doc.splitlines()[0].strip() if doc else ""


class ToolCommand(click.Command):
    """A command that forwards its arguments to a module's ``main(argv)``.

    The module is imported only to run it. The listing reads its docstring from source instead, so
    building and showing the tree costs nothing and cannot be broken by a tool's dependencies.
    """

    def __init__(self, module: str, name: str, **kwargs):
        super().__init__(
            name,
            params=[click.Argument(["args"], nargs=-1, type=click.UNPROCESSED)],
            callback=self._forward,
            # No click help option: --help belongs to the tool's own parser, which is where its
            # options are declared. Unknown options must pass through untouched for the same reason.
            context_settings={"ignore_unknown_options": True, "help_option_names": []},
            add_help_option=False,
            **kwargs,
        )
        self.module = module

    def _load(self):
        return importlib.import_module(self.module)

    @contextlib.contextmanager
    def _named(self, ctx):
        """Run with ``argv[0]`` set to the full command path.

        argparse takes its program name from ``argv[0]``, which here is the ``roqsim`` launcher -- so a
        tool's own usage line would read ``usage: roqsim [-h] --grid ...`` and send the reader off to
        spell an invocation that does not exist. It has to name the path they actually typed.
        """
        original = sys.argv[0]
        sys.argv[0] = ctx.command_path if ctx else original
        try:
            yield
        finally:
            sys.argv[0] = original

    def _run(self, args, ctx):
        """Call the module's ``main`` with `args`, however it expects to receive them."""
        main = self._load().main
        with self._named(ctx):
            # `main(argv)` is the convention, but a tool whose main() reads sys.argv directly is
            # perfectly ordinary Python -- and a package outside this repo may well ship one. Give it
            # the arguments the way it expects them rather than a TypeError traceback.
            if _takes_argv(main):
                return main(list(args))
            original, sys.argv[1:] = sys.argv[1:], list(args)
            try:
                return main()
            finally:
                sys.argv[1:] = original

    def _forward(self, args):
        raise SystemExit(self._run(args, click.get_current_context(silent=True)))

    def get_short_help_str(self, limit: int = 45) -> str:
        return summary_line(self.module)

    # A tool's real help is its parser's, so asking click for help means running the tool with
    # --help rather than printing click's own (empty) description.
    def get_help(self, ctx) -> str:
        self._run(["--help"], ctx)
        return ""


class BlenderToolCommand(ToolCommand):
    """A tool that runs inside Blender's own Python, driven from here.

    Blender ships the USD importer and the mesh operators these tools need, and its Python is a
    separate interpreter with its own ``bpy`` -- so the module cannot be imported into this process
    at all. Wrapping it anyway is the point: the incantation
    (``blender --background --python <file> -- <args>``) is the part nobody remembers, and a tool
    that is only reachable by remembering it is one nobody finds.
    """

    def _blender(self) -> str:
        exe = shutil.which("blender")
        if not exe:
            raise click.ClickException(
                f"'{self.name}' runs inside Blender, which is not on PATH. Install Blender 4.x "
                f"(https://www.blender.org/download/) and re-run; every other tool in this group "
                f"runs without it."
            )
        return exe

    def _script(self) -> str:
        spec = importlib.util.find_spec(self.module)
        if spec is None or not spec.origin:
            raise click.ClickException(f"cannot locate the source of {self.module}")
        return spec.origin

    def _forward(self, args):
        cmd = [self._blender(), "--background", "--python", self._script(), "--", *args]
        raise SystemExit(subprocess.call(cmd))

    def get_help(self, ctx) -> str:
        """Its own docstring: argparse lives on the far side of Blender and cannot be asked here."""
        click.echo(f"Usage: {ctx.command_path} [ARGS]...\n")
        click.echo(module_docstring(self.module).strip())
        click.echo(f"\nRuns inside Blender: {self._script()}")
        return ""


def tool(module: str, name: str | None = None, *, blender: bool = False) -> ToolCommand:
    """Wrap ``<module>.main(argv)`` as a subcommand. Name defaults to the module's own, dashed."""
    cls = BlenderToolCommand if blender else ToolCommand
    return cls(module, name or module.rsplit(".", 1)[-1].replace("_", "-"))


def load_groups(root: click.Group) -> None:
    """Attach every group declared under ``roqsim.commands`` by an installed package.

    A name claimed twice is refused rather than resolved: silently letting the second registration
    win would hide a whole package's tools behind another's, and a tool nobody can see is the
    failure this command tree exists to prevent. One broken package must not take the CLI down with
    it, though -- its group is skipped with a warning, and every other group still loads.
    """
    seen: dict[str, str] = {}
    for ep in entry_points(group=COMMAND_GROUP):
        origin = getattr(getattr(ep, "dist", None), "name", "?")
        if ep.name in seen:
            raise click.ClickException(
                f"two packages both register the '{ep.name}' command group: {seen[ep.name]} and "
                f"{origin}. Rename one of them -- otherwise one package's tools become invisible."
            )
        try:
            root.add_command(ep.load(), name=ep.name)
        except Exception as exc:  # a broken package must not hide the working ones
            click.echo(f"roqsim: could not load the '{ep.name}' command group: {exc}", err=True)
            continue
        seen[ep.name] = origin


_TARGET_MARKS = ("/", ".yaml", ".yml", ".xml", ":", "\\")


class RootGroup(click.Group):
    """The root group, with one courtesy: it recognises what used to be `roqsim <world>`.

    Running a world was the whole of this command once, so that spelling is written into other
    people's notes, scripts and muscle memory. `No such command 'worlds/x.yaml'` is a true but
    useless answer to it, and a stale invocation that fails obscurely is how a broken command line
    survives in a checked-in Makefile for months.
    """

    def resolve_command(self, ctx, args):
        name = args[0] if args else ""
        if name not in self.commands and any(m in name for m in _TARGET_MARKS):
            raise click.UsageError(
                f"'{name}' looks like a world, scene or model, and running one is now a subcommand:\n"
                f"    roqsim sim {name} {' '.join(args[1:])}".rstrip()
                + "\n\nSee `roqsim --help` for the other groups.",
                ctx,
            )
        return super().resolve_command(ctx, args)


@click.group(cls=RootGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="roqsim", prog_name="roqsim")
def cli() -> None:
    """Run simulations and the tools that build what they run.

    Each group below comes from an installed package. Use `roqsim <group> --help` for its tools, and
    `python -m pydoc <module>` for the reasoning behind one.
    """


cli.add_command(tool("roqsim.runner", "sim"))
cli.add_command(tool("roqsim.render", "render"))
cli.add_command(tool("roqsim.state", "state"))
cli.add_command(tool("roqsim.introspection", "plugins"))


@cli.group("export")
def export_group() -> None:
    """Export a world or model to another format."""


export_group.add_command(tool("roqsim.export_web", "web"))
export_group.add_command(tool("roqsim.export_capture", "capture"))
export_group.add_command(tool("roqsim.export_urdf", "urdf"))
export_group.add_command(tool("roqsim.export_srdf", "srdf"))


def main(argv: list | None = None) -> int:
    # standalone_mode=False so a tool's own SystemExit code reaches the shell unchanged; that also
    # means click's exceptions are ours to report, and a usage error must read as a message rather
    # than as a traceback.
    try:
        load_groups(cli)
        return cli.main(args=argv, standalone_mode=False) or 0
    except click.UsageError as err:
        click.echo(f"roqsim: {err.format_message()}", err=True)
        return 2
    except click.ClickException as err:
        click.echo(f"roqsim: {err.format_message()}", err=True)
        return err.exit_code
    except click.Abort:
        click.echo("roqsim: aborted", err=True)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
