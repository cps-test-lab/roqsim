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
