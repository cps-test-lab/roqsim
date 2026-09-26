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

"""A real MCP server for roqsim's introspection -- for use without any surrounding harness.

The JSON CLIs (``python -m roqsim.introspection list``, ``roqsim catalog models``) answer
"what does this container have -- plugins, models, worlds" from a shell, but not from an MCP
client. This registers the exact same functions as MCP tools -- no new logic, just a
second, equally thin adapter, the same one-line pattern any MCP plugin uses to register
theirs.

Runnable via ``roqsim mcp serve``, the ``roqsim-mcp`` console script, or
``python -m roqsim_mcp`` (stdio transport), so anyone with a shell in the image -- and
without core ``roqsim`` having ever heard of ``fastmcp`` -- can point an MCP client at it
directly.
"""

from fastmcp import FastMCP

from roqsim.catalog import get_model_details, list_models, list_worlds
from roqsim.introspection import get_plugin_details, list_plugins

#: Everything a caller needs to ask before writing a world: what plugins exist, what can be spawned,
#: and what worlds are already there. Adding one is one line here, because each is already a plain
#: function returning plain dicts -- the reason this server has no logic of its own.
_TOOLS = [list_plugins, get_plugin_details, list_models, get_model_details, list_worlds]

#: What a client is told once, at connect time: the tools answer for THIS installation, a listed name
#: is one that resolves, and the ``use`` line each row carries is what goes in a world file. Stated
#: here rather than left to the tool descriptions because the loop -- list, then detail, then write --
#: is not visible from any one tool.
_INSTRUCTIONS = """\
What this roqsim installation can put in a world, read from the same registries its loader resolves
against -- so a name a tool returns is a name that resolves, and a name it does not return is not.

- list_plugins / get_plugin_details: the installed `roqsim.plugins`, then one plugin's config keys.
  `schema` (types, defaults, units, bounds) is authoritative where present; `parameters` is the
  documented block otherwise.
- list_models / get_model_details: what `spawn_model` can spawn, by the `ref` that resolves it, with
  the components its manifest brings along.
- list_worlds: every world YAML (`roqsim sim <ref>`, `extends: <ref>`), baked scene and built-in
  definition (`sim.world` values).

Every row carries `use`, the line to write. A name that resolves to nothing comes back as
`{"error": "..."}` rather than as a tool failure. These tools describe; they run nothing -- a run is
`roqsim sim`. The same answers from a shell: `roqsim plugins` and `roqsim catalog`.
"""


def create_server() -> FastMCP:
    mcp = FastMCP("roqsim", instructions=_INSTRUCTIONS)
    for fn in _TOOLS:
        mcp.tool()(fn)
    return mcp


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
