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
