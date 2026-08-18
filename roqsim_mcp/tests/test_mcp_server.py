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

"""The standalone MCP server registers roqsim.introspection's own functions, unchanged --
this confirms the registration, not the introspection logic itself (covered by
roqsim/tests/test_introspection.py).
"""

from __future__ import annotations

import asyncio
import json

from roqsim_mcp.mcp_server import create_server


def _run(coro):
    return asyncio.run(coro)


def test_both_tools_are_registered():
    async def _names():
        return {t.name for t in await create_server().list_tools()}

    assert _run(_names()) == {"list_plugins", "get_plugin_details"}


def test_list_plugins_returns_dummy_always_registered_by_core_rst():
    async def _call():
        server = create_server()
        return await server.call_tool("list_plugins", {})

    result = _run(_call())
    catalog = json.loads(result.content[0].text)
    dummy = next((item for item in catalog["items"] if item["name"] == "dummy"), None)
    assert dummy is not None
    assert dummy["doc"]


def test_get_plugin_details_unknown_name_is_error_not_an_exception():
    async def _call():
        server = create_server()
        return await server.call_tool("get_plugin_details", {"name": "not_a_real_plugin_xyz"})

    result = _run(_call())
    payload = json.loads(result.content[0].text)
    assert "error" in payload
