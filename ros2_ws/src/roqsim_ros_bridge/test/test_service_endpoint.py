"""The ``service`` inbound kind: a command whose reply says what the simulator did.

Guards the reason a service exists here at all. A topic publish is fire-and-forget, so a scenario that
injects a fault and gets no answer records a run that *believes* it injected something -- the failure
mode that produces plausible wrong data. The reply turns that into ``success: false``, which a
scenario's ``service_call()`` can fail the trial on.

ROS-free on purpose, like the handler it tests: the policy is duck-typed over request/response
members, so the interesting mistakes are catchable without a ROS installation.
"""

from __future__ import annotations

from dataclasses import dataclass

from roqsim.context import Endpoint, SimContext
from roqsim_ros_bridge.physics import barrier, run_on_physics
from roqsim_ros_bridge.services import get_service_handler, set_bool


class _Request:
    def __init__(self, data: bool):
        self.data = data


class _Response:
    def __init__(self):
        self.success = False
        self.message = ""


@dataclass
class _Report:
    verified: str


class _Producer:
    """Stand-in for a plugin with a switch and a self-verification, e.g. ``model_override``."""

    def __init__(self, verdict="landed"):
        self.active = False
        self.verdict = verdict

    def set_active(self, on: bool) -> None:
        self.active = bool(on)

    def read_state(self) -> _Report:
        return _Report(self.verdict)


def _wire(producer=None, *, state_key="producer", steps=True):
    """A ctx whose command queue is drained by a background 'physics thread', as the engine does."""
    import threading

    ctx = SimContext(config={})
    if producer is not None:
        ctx.blackboard.set(state_key, producer)
    stop = threading.Event()

    def physics():
        while not stop.is_set():
            ctx.drain_commands()  # the engine drains at the start of every pre_step
            stop.wait(0.001)

    thread = threading.Thread(target=physics, daemon=True)
    if steps:
        thread.start()
    return ctx, stop, thread


def _endpoint(write, state_key="producer") -> Endpoint:
    return Endpoint(
        name="override",
        direction="in",
        owner="thing",
        write=write,
        backend={"ros2": {"service": "std_srvs.srv.SetBool", "state_key": state_key}},
    )


def _call(ctx, endpoint, data: bool) -> _Response:
    # What the bridge hands a handler: the payload marshalled onto the physics thread via ctx.post.
    on_payload = lambda payload: ctx.post(lambda _c, p=payload: endpoint.write(p))  # noqa: E731
    return set_bool(_Request(data), _Response(), ctx, on_payload, endpoint)


def test_the_command_reaches_the_producer_and_the_reply_carries_the_verdict():
    producer = _Producer("landed")
    ctx, stop, thread = _wire(producer)
    try:
        response = _call(ctx, _endpoint(producer.set_active), True)
    finally:
        stop.set()
        thread.join(timeout=1.0)

    assert producer.active is True
    assert response.success is True
    assert response.message == "landed"


def test_an_override_that_did_not_land_is_a_failed_call():
    """The whole point of a service: a fault that silently did nothing must not read as success."""
    producer = _Producer("no_effect")
    ctx, stop, thread = _wire(producer)
    try:
        response = _call(ctx, _endpoint(producer.set_active), True)
    finally:
        stop.set()
        thread.join(timeout=1.0)

    assert response.success is False
    assert response.message == "no_effect"


def test_nothing_to_verify_still_succeeds():
    """A producer with no state reader reports that the command was applied, not a false verdict."""
    producer = _Producer()
    ctx, stop, thread = _wire()  # nothing published on the blackboard
    try:
        response = _call(ctx, _endpoint(producer.set_active), False)
    finally:
        stop.set()
        thread.join(timeout=1.0)

    assert response.success is True
    assert response.message == "restored"


def test_a_stalled_simulator_fails_rather_than_claiming_success():
    """No physics thread: the command is queued and never runs, which is a failure, not an OK."""
    producer = _Producer()
    ctx, _stop, _thread = _wire(producer, steps=False)

    from roqsim_ros_bridge import physics

    original = physics.DEFAULT_TIMEOUT_S
    physics.DEFAULT_TIMEOUT_S = 0.05  # the test must not wait 2 s to prove a timeout
    try:
        response = _call(ctx, _endpoint(producer.set_active), True)
    finally:
        physics.DEFAULT_TIMEOUT_S = original

    assert response.success is False
    assert "did not apply" in response.message
    assert producer.active is False


def test_run_on_physics_reports_whether_it_ran():
    ctx, stop, thread = _wire()
    try:
        seen = []
        assert run_on_physics(ctx, lambda _c: seen.append(1)) is True
        assert seen == [1]
        assert barrier(ctx) is True
    finally:
        stop.set()
        thread.join(timeout=1.0)


def test_the_handler_is_registered_under_its_type():
    assert get_service_handler("std_srvs.srv.SetBool") is set_bool
