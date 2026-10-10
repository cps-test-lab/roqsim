"""The commands that talk to a running simulation: ``roqsim ls|endpoints|describe|read|call|sub|ctl``.

Each is a module of its own with a ``main(argv)`` (the command tree wraps modules), and all of them
are this module's :func:`run`, which parses, finds the simulator (:func:`roqsim.ipc.discover`) and
prints. A value is printed as JSON; an array larger than ``--max-items`` numbers is summarised by
dtype, shape and range unless ``--full`` asks for all of it.

Exit status: 0 on success; 2 when there is no simulator to talk to or the request names something
that does not exist; 5 when the simulator refused a command or it timed out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .. import exit_status
from ..ipc import ControlError, running, wire

_NOTE = "2 is also no simulator found; 5 is a command the simulator refused or that timed out."


def _parser(name: str, doc: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"roqsim {name}",
        description=doc.strip().splitlines()[0],
        epilog=exit_status.epilog(exit_status.BAD_INPUT, exit_status.FINDING, note=_NOTE),
    )
    parser.add_argument(
        "--control",
        metavar="URI",
        default=None,
        help="the simulator's control URI (default: $ROQSIM_CONTROL, or the only simulator "
        "running; `roqsim ls` lists them)",
    )
    return parser


def _values(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-items",
        type=int,
        default=16,
        metavar="N",
        help="summarise an array with more than N numbers (default 16)",
    )
    parser.add_argument("--full", action="store_true", help="print every array in full")


def _show(value, args) -> None:
    limit = 0 if getattr(args, "full", False) else getattr(args, "max_items", 16)
    print(json.dumps(wire.plain(value, max_items=limit), indent=2, default=str))


def _json_value(text: str | None):
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # a bare word is a string: `roqsim call x/y/mode fast`


def run(name: str, argv: list | None, doc: str) -> int:
    """Parse *argv* for command *name* and run it; the exit status."""
    parser = _parser(name, doc)
    if name == "endpoints":
        parser.add_argument("prefix", nargs="?", default="", help="only the endpoints under it")
    elif name == "describe":
        parser.add_argument("path", help="an endpoint's path, e.g. robot/lidar/scan")
    elif name == "read":
        parser.add_argument("path")
        parser.add_argument("--field", default="", help="one (dotted) field of the value")
        _values(parser)
    elif name == "call":
        parser.add_argument("path")
        parser.add_argument("value", nargs="?", default=None, help="the payload, as JSON")
        parser.add_argument("--timeout", type=float, default=None, metavar="S")
        _values(parser)
    elif name == "sub":
        parser.add_argument("prefix", nargs="?", default="", help="the endpoints under it")
        parser.add_argument("--count", type=int, default=0, help="stop after N values")
        _values(parser)
    elif name == "ctl":
        parser.add_argument("action", choices=("pause", "resume", "step", "reset", "state"))
        parser.add_argument("n", nargs="?", type=int, default=1, help="steps, for step")
    args = parser.parse_args(argv)
    if name == "ls":
        return _ls()
    from ..control_client import Client

    try:
        with Client(args.control) as sim:
            return _COMMANDS[name](sim, args)
    except ControlError as err:
        print(f"roqsim {name}: {err}", file=sys.stderr)
        refused = err.kind in ("refused", "timeout")
        return exit_status.FINDING if refused else exit_status.BAD_INPUT
    except KeyboardInterrupt:
        return exit_status.OK


def _ls() -> int:
    live = running()
    if not live:
        print("no simulator running")
        return exit_status.OK
    for entry in live:
        age = time.time() - float(entry.get("started", time.time()))
        print(f"{entry['uri']}  pid {entry['pid']}  up {age:.0f} s  {entry.get('world', '')}")
    return exit_status.OK


def _endpoints(sim, args) -> int:
    for entry in sim.endpoints(args.prefix):
        rate = f"{entry['rate_hz']:g} Hz" if entry.get("rate_hz") else ""
        print(f"{entry['path']:<48} {entry['kind']:<8} {rate:<9} {entry.get('doc') or ''}".rstrip())
    return exit_status.OK


def _describe(sim, args) -> int:
    entry = sim.describe(args.path)
    if isinstance(entry, list):  # a prefix
        for row in entry:
            print(f"{row['path']:<48} {row['kind']}")
        return exit_status.OK
    print(entry["path"])
    rate = f", {entry['rate_hz']:g} Hz" if entry.get("rate_hz") else ""
    if rate and entry.get("rate_from"):
        rate += f" (from {entry['rate_from']})"
    print(f"  kind: {entry['kind']}{rate}")
    lazy = entry.get("lazy")
    if lazy is True:
        print("  lazy: not read while nobody subscribes")
    elif isinstance(lazy, dict):
        default = f" (default {lazy['default']})" if "default" in lazy else ""
        print(f"  lazy: per instance, from {lazy['from']}{default}")
    elif lazy:
        print(f"  lazy: {lazy} per instance")
    if entry.get("when"):
        print(f"  present when: {entry['when']}")
    if entry.get("family"):
        print(f"  one of a family, per item of: {entry['family']}")
    if entry.get("doc"):
        print("  " + entry["doc"].replace("\n", "\n  "))
    if entry["kind"] != "out" and entry.get("payload"):
        print(f"  takes: {_typed(entry['payload'])}")
    for param in entry.get("params") or []:
        print(
            f"  parameter {_typed(param)}" + ("" if param["required"] else f" = {param['default']}")
        )
    if entry.get("result"):
        print(f"  {'value' if entry['kind'] == 'out' else 'returns'}: {_typed(entry['result'])}")
        for field in entry["result"].get("fields") or []:
            print(f"    {_typed(field)}")
    if entry.get("confirm"):
        print(f"  confirmed by: {entry['confirm']}")
    for backend, named in (entry.get("bridges") or {}).items():
        print(f"  on {backend}: {json.dumps(named)}")
    if entry["kind"] == "out":
        print(f"  example: roqsim read {entry['path']}")
    elif "params" in entry:
        example = {p["name"]: _example(p) for p in entry["params"]}
        print(
            f"  example: roqsim call {entry['path']}"
            + (f" '{json.dumps(example)}'" if example else "")
        )
    else:
        print(f"  example: roqsim call {entry['path']} '<json payload>'")
    return exit_status.OK


def _typed(row: dict) -> str:
    """``name: type [unit] -- doc`` of one described parameter, field or value."""
    text = f"{row['name']}: {row['type']}" if "name" in row else row["type"]
    if row.get("unit"):
        text += f" [{row['unit']}]"
    if row.get("doc"):
        text += f" -- {row['doc']}"
    return text


_PLACEHOLDERS = {"float": 0.0, "int": 0, "bool": False, "str": ""}


def _example(param: dict):
    if not param["required"]:
        return param.get("default")
    return _PLACEHOLDERS.get(
        param["type"], [] if param["type"].startswith(("list", "tuple", "array")) else None
    )


def _read(sim, args) -> int:
    _show(sim.read(args.path, args.field), args)
    return exit_status.OK


def _call(sim, args) -> int:
    _show(sim.call(args.path, _json_value(args.value), timeout=args.timeout), args)
    return exit_status.OK


def _sub(sim, args) -> int:
    limit = 0 if args.full else args.max_items
    with sim.subscribe(args.prefix) as sub:
        for n, (path, t, value) in enumerate(sub, start=1):
            print(f"{t:.6f} {path} {json.dumps(wire.plain(value, max_items=limit), default=str)}")
            sys.stdout.flush()
            if args.count and n >= args.count:
                break
    return exit_status.OK


def _ctl(sim, args) -> int:
    action = args.action
    result = sim.step(args.n) if action == "step" else getattr(sim, action)()
    print(json.dumps(wire.plain(result), indent=2))
    return exit_status.OK


_COMMANDS = {
    "endpoints": _endpoints,
    "describe": _describe,
    "read": _read,
    "call": _call,
    "sub": _sub,
    "ctl": _ctl,
}
