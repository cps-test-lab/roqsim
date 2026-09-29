# roqsim_mcp

A standalone MCP server exposing `roqsim.introspection`'s `list_plugins`/`get_plugin_details`
(the `roqsim.plugins` registry) as MCP tools, for a client with no other roqsim knowledge.

It also reaches a simulation that is running: `list_endpoints`, `describe_endpoint`,
`read_endpoint`, `call_endpoint`, `pause`, `resume` and `step` talk to the control socket
`roqsim sim` serves (see the docs page "Talking to a running simulation"). Each finds the
simulator as `roqsim ls` does -- `ROQSIM_CONTROL`, the run directory, or the only one running -- or
takes its address as `control`.

The real logic lives in core `roqsim` (`roqsim.introspection`, `roqsim.catalog`,
`roqsim.control_client`); this package is a thin adapter, registering the same functions as MCP
tools rather than duplicating anything.

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
