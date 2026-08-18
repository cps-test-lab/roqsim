"""ROS 2 service handlers for the bridge, keyed by srv type string.

The third inbound kind, and the one for a **command whose outcome the caller needs**. A topic is a
stream with no answer, so a caller cannot learn whether what it asked for happened; an action is a
goal that takes time, reports feedback and can be cancelled, which is machinery an instantaneous
command has no use for. A service is the shape in between, and it is what lets a scenario's
``service_call()`` *fail a trial* when the simulator did not do the thing.

A producer declares an ``in`` endpoint whose ros2 hint block carries ``service`` (the type string) and
``name`` (the relative service name); the bridge looks up the handler here and serves it. Handler
contract::

    handler(request, response, ctx, on_payload, endpoint) -> response

Deliberately free of ROS imports, unlike :mod:`roqsim_ros_bridge.actions`: a handler only touches
duck-typed request/response members, so the reply *policy* -- which is where the interesting mistakes
live -- is unit-testable without a ROS installation. The registries are still populated by import, and
a handler in another package still travels through the ``roqsim_ros_bridge.extensions`` entry-point group
(see :mod:`roqsim_ros_bridge.extensions`).

Services declared by hand, on a node, rather than through an endpoint: see
:mod:`roqsim_ros_bridge.sim_interfaces`. Those implement a *standardised* message set with its own result
codes and several of them are queries, which the endpoint model has no direction for -- endpoints
exist so a **ROS-free** producer can be served without knowing a backend, and that plugin is not one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import physics
from .extensions import EXTENSION_GROUP, load_extensions
from .physics import barrier

# srv-type-string -> handler(request, response, ctx, on_payload, endpoint) -> response message
SERVICE_HANDLERS: dict[str, Callable[[Any, Any, Any, Callable[[Any], None], Any], Any]] = {}


def service_handler(type_path: str):
    def register(fn):
        SERVICE_HANDLERS[type_path] = fn
        return fn

    return register


def get_service_handler(
    type_path: str,
) -> Callable[[Any, Any, Any, Callable[[Any], None], Any], Any]:
    load_extensions()  # a handler may live in another package
    fn = SERVICE_HANDLERS.get(type_path)
    if fn is None:
        raise KeyError(
            f"no service handler registered for {type_path!r}; "
            f"known: {sorted(SERVICE_HANDLERS)} (see roqsim_ros_bridge.services; a handler in "
            f"another package must be advertised in the {EXTENSION_GROUP!r} entry-point group)"
        )
    return fn


@service_handler("std_srvs.srv.SetBool")
def set_bool(request, response, ctx, on_payload, endpoint=None):
    """Switch a producer on or off, and report whether it took effect.

    The generic policy for "a command with an outcome": the request goes to the producer as the
    neutral ``bool`` payload its ``write`` expects, and the reply says what the simulator *did* rather
    than that the message was delivered.

    The verdict is the producer's, not this handler's. A producer that publishes a state object
    carrying ``verified`` (see ``roqsim.plugins.model_override``) has it read off the blackboard under the
    endpoint's ros2 ``state_key`` hint and reported in ``message``, and ``verified == "no_effect"``
    becomes ``success = False`` -- which is the whole reason to serve a service instead of a topic. A
    producer without such a reader simply reports that the command was applied. Nothing here knows
    what was switched.

    Timing: ``ctx.post`` is FIFO and drained at the start of ``pre_step``, so one barrier proves the
    write has run, and a second spans the step whose ``post_step`` records the verdict. Without the
    second, this would read the verdict from *before* the change and report it as this call's.
    """
    want = bool(request.data)
    on_payload(want)  # queued for the physics thread by the bridge's inbound marshaller
    if not barrier(ctx):
        response.success = False
        response.message = (
            f"the simulation did not apply the command within {physics.DEFAULT_TIMEOUT_S} s "
            "(is it paused?)"
        )
        return response

    read_state = None
    if endpoint is not None:
        state_key = endpoint.backend.get("ros2", {}).get("state_key", "")
        if state_key:
            read_state = getattr(ctx.blackboard.get(state_key), "read_state", None)
    if read_state is None:
        response.success = True
        response.message = "applied" if want else "restored"
        return response

    barrier(ctx)  # let the step whose post_step verifies the change complete
    verdict = str(getattr(read_state(), "verified", "") or "unknown")
    response.success = verdict != "no_effect"
    response.message = verdict
    return response
