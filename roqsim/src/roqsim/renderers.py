# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""List the installed renderers, and each one's capabilities and settings, as JSON.

A renderer is chosen in a world with ``sim.renderer: <name>`` and configured in
``sim.renderers.<name>``; this says which names exist, what each renders and which keys its block
takes, with their defaults and meanings, the way ``roqsim plugins describe`` does for plugins::

    roqsim renderers list
    roqsim renderers describe mujoco
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys

from roqsim import exit_status
from roqsim.entry_points import entry_points
from roqsim.plugin import PluginError
from roqsim.registry import RENDERER_GROUP, renderer_names, resolve_renderer


def _summary(cls) -> str:
    doc = inspect.getdoc(cls) or ""
    return doc.split("\n\n")[0].replace("\n", " ").strip()


def _package(name: str) -> str:
    for ep in entry_points(RENDERER_GROUP):
        if ep.name == name:
            dist = getattr(ep, "dist", None)
            return getattr(dist, "name", "") or ep.value.split(".")[0]
    return ""


def list_renderers() -> dict:
    """Every installed renderer: its name, package and one-line summary."""
    rows = []
    for name in renderer_names():
        row = {"name": name, "package": _package(name)}
        try:
            row["summary"] = _summary(resolve_renderer(name))
        except PluginError as err:
            row["error"] = str(err)
        rows.append(row)
    return {"group": RENDERER_GROUP, "renderers": rows}


def describe_renderer(name: str) -> dict:
    """One renderer in full: summary, capabilities, and every setting with default and meaning."""
    try:
        cls = resolve_renderer(name)
    except PluginError as err:
        return {"name": name, "error": str(err)}
    return {
        "name": name,
        "package": _package(name),
        "class": f"{cls.__module__}:{cls.__qualname__}",
        "summary": _summary(cls),
        "capabilities": cls.capabilities.as_dict(),
        "settings": {
            key: {"default": s.default, "meaning": s.meaning} for key, s in cls.SETTINGS.items()
        },
        "chosen_by": f"sim.renderer: {name}",
        "settings_block": f"sim.renderers.{name}",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="roqsim renderers",
        description="JSON introspection of the installed renderers (the roqsim.renderers group).",
        epilog=exit_status.epilog(
            exit_status.BAD_INPUT, note="2 includes `describe` naming no installed renderer."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Every installed renderer, as JSON.")
    p_describe = sub.add_parser(
        "describe", help="One renderer's capabilities and settings, as JSON."
    )
    p_describe.add_argument("name", help="A renderer name, e.g. 'mujoco'")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(list_renderers(), indent=2))
        return exit_status.OK
    result = describe_renderer(args.name)
    print(json.dumps(result, indent=2))
    sys.exit(exit_status.BAD_INPUT if "error" in result else exit_status.OK)


if __name__ == "__main__":
    main()
