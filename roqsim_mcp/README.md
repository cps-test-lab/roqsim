# roqsim_mcp

A standalone MCP server exposing what this roqsim installation can put in a world, for a client with
no other roqsim knowledge: `list_plugins` / `get_plugin_details` (the `roqsim.plugins` registry,
from `roqsim.introspection`) and `list_models` / `get_model_details` / `list_worlds` (the model and
world catalogs, from `roqsim.catalog`). Its instructions tell a client the loop -- list, then detail,
then write the `use` line each row carries -- and that a name a tool returns is one that resolves.

The real logic lives in core `roqsim`; this package is a thin adapter, registering the same functions
as MCP tools rather than duplicating anything. From a shell the same answers are `roqsim plugins` and
`roqsim catalog`.

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
