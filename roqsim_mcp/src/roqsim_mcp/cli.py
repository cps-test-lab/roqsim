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

"""``roqsim mcp`` -- registered into the shared ``roqsim.commands`` group like every other
family package's CLI group. Not installed by default: ``roqsim mcp`` simply does not
appear in ``roqsim --help`` unless this package is.
"""

import click

from roqsim_mcp.mcp_server import main as _serve


@click.group("mcp")
def mcp_group() -> None:
    """Run the standalone MCP server for roqsim's plugin, model and world catalogs."""


@mcp_group.command("serve")
def serve() -> None:
    """Start the MCP server (stdio transport)."""
    _serve()
