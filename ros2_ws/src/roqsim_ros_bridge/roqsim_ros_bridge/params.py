"""What an inbound message becomes for the endpoint it reaches. Free of ROS imports.

A decoder (:mod:`roqsim_ros_bridge.registry`) turns a message into its named parameters. An endpoint
that declares ``params`` (:mod:`roqsim.endpoint`) takes exactly that mapping, and checks it itself
before queueing. An untyped endpoint (``params is None``) takes the positional form its ``write``
was written for: the values in the decoder's order, or the one value alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def payload_for(endpoint, params: Mapping[str, Any]) -> Any:
    """*params* as *endpoint*'s ``write`` takes them."""
    if getattr(endpoint, "params", None) is not None:
        return dict(params)
    values = tuple(params.values())
    return values[0] if len(values) == 1 else values


def one_value_name(endpoint, name: str) -> str:
    """The parameter a goal's single value (its field *name*, ``position``) goes to: the endpoint's
    own where it declares exactly one (a door's ``openness``), so a producer names its parameter for
    its own quantity; *name* where it declares several and one of them is *name*, or none (untyped).
    Raises :class:`ValueError` for several parameters none of which is *name*: which one the value
    is for cannot be told, which the bridge refuses when it starts, naming the producer."""
    params = getattr(endpoint, "params", None)
    if params is None:
        return name
    if len(params) == 1:
        return params[0].name
    if any(p.name == name for p in params):
        return name
    where = f"{getattr(endpoint, 'owner', '?')}/{getattr(endpoint, 'name', '?')}"
    raise ValueError(
        f"{where} takes {', '.join(p.name for p in params)}: a goal's single value goes to a "
        f"producer's only parameter, or to one named {name!r}; name one so"
    )


def one_value(endpoint, name: str, value: Any) -> dict[str, Any]:
    """*value* as the parameters of an endpoint that takes one value (:func:`one_value_name`)."""
    return {one_value_name(endpoint, name): value}
