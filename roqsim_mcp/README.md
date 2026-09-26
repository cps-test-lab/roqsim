# roqsim_mcp

A standalone MCP server exposing `roqsim.introspection`'s `list_plugins`/`get_plugin_details`
(the `roqsim.plugins` registry) as MCP tools, for a client with no other roqsim knowledge.

The real logic lives in core `roqsim` (`roqsim.introspection`); this package is a thin adapter,
registering the same functions as MCP tools rather than duplicating anything.

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
