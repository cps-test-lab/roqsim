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

"""A real MCP server for roqsim's introspection and a running simulation -- for use without any
surrounding harness.

The JSON CLIs (``python -m roqsim.introspection list``, ``roqsim catalog models``) answer
"what does this container have -- plugins, models, worlds" from a shell, but not from an MCP
client. This registers the exact same functions as MCP tools -- no new logic, just a
second, equally thin adapter, the same one-line pattern any MCP plugin uses to register
theirs.

The same holds for a simulation that is running: ``roqsim.control_client``'s plain-JSON calls
list, describe, read and call its endpoints and pause, resume and step it, over the control socket
``roqsim sim`` serves. Each finds the simulator the way ``roqsim ls`` does (``ROQSIM_CONTROL``, the
run directory, or the only one running) unless given ``control``; an error is a ``{"error": ...}``
answer, and an array of more than a few dozen numbers is summarised rather than listed.

Runnable via ``roqsim mcp serve``, the ``roqsim-mcp`` console script, or
``python -m roqsim_mcp`` (stdio transport), so anyone with a shell in the image -- and
without core ``roqsim`` having ever heard of ``fastmcp`` -- can point an MCP client at it
directly.
"""

from fastmcp import FastMCP

from roqsim.catalog import get_model_details, list_models, list_worlds
from roqsim.control_client import (
    call_endpoint,
    describe_endpoint,
    list_endpoints,
    pause,
    read_endpoint,
    resume,
    step,
)
from roqsim.introspection import get_plugin_details, list_plugins
from roqsim_mcp.check import check_world

#: Everything a caller needs to ask before writing a world -- what plugins exist, what can be
#: spawned, what worlds are already there -- and what it asks of one that runs. Adding one is one
#: line here, because each is already a plain function returning plain dicts -- the reason this
#: server has no logic of its own.
_TOOLS = [
    list_plugins,
    get_plugin_details,
    list_models,
    get_model_details,
    list_worlds,
    list_endpoints,
    describe_endpoint,
    read_endpoint,
    call_endpoint,
    pause,
    resume,
    step,
]


def create_server() -> FastMCP:
    mcp = FastMCP("roqsim")
    for fn in _TOOLS:
        mcp.tool()(fn)
    # After writing a world, whether it loads -- the one tool here that builds something, which is
    # why it is not a plain function of core's but a wrapper that runs it out of process.
    mcp.tool()(check_world)
    return mcp


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
