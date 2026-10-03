# roqsim_mcp

A standalone MCP server exposing what this roqsim installation can put in a world, for a client with
no other roqsim knowledge: `list_plugins` / `get_plugin_details` (the `roqsim.plugins` registry,
from `roqsim.introspection`) and `list_models` / `get_model_details` / `list_worlds` (the model and
world catalogs, from `roqsim.catalog`). Its instructions tell a client the loop -- list, then detail,
then write the `use` line each model and world row carries, then `check_world` -- and that a name a
tool returns is one that resolves.

It also reaches a simulation that is running: `list_endpoints`, `describe_endpoint`,
`read_endpoint`, `call_endpoint`, `pause`, `resume` and `step` talk to the control socket
`roqsim sim` serves (see the docs page "Talking to a running simulation"). Each finds the
simulator as `roqsim ls` does -- `ROQSIM_CONTROL`, the run directory, or the only one running -- or
takes its address as `control`.

The real logic lives in core `roqsim` (`roqsim.introspection`, `roqsim.catalog`,
`roqsim.control_client`); this package is a thin adapter, registering the same functions as MCP
tools rather than duplicating anything. From a shell the same answers are `roqsim plugins`,
`roqsim catalog` and the control commands (`roqsim ls`, `roqsim endpoints`, ...).

## Usage

Inside an experiment image with this package installed:

```console
roqsim mcp serve
```

or, without the `roqsim` CLI wrapper:

```console
python -m roqsim_mcp
```

Both start an MCP server on stdio.

## Checking a world

`check_world` runs `roqsim check --json` on a world path or ref, so a client that writes a world
through MCP alone can ask whether it loads before anything runs it. It runs the command out of
process: a check builds the model and runs every plugin, and nothing one of them prints may reach the
stdio stream this server speaks on.
