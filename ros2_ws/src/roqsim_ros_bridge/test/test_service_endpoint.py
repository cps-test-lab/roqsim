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


# -- Trigger: a command that carries nothing -----------------------------------------------------


class _Button:
    """Stand-in for a plugin with a button and no argument, e.g. ``force_torque``'s tare."""

    def __init__(self):
        self.presses = 0

    def press(self, _payload=None) -> None:
        self.presses += 1


def test_a_trigger_reaches_the_producer_on_the_physics_thread():
    """The whole point of the kind: no argument to marshal, and still an answer.

    A real FT driver's zero is a service taking nothing, and a scenario needs to know it landed --
    a run that carried on believing it had tared would measure against an offset never applied.
    """
    button = _Button()
    ctx, stop, thread = _wire()
    endpoint = Endpoint(
        name="tare",
        direction="in",
        owner="ft",
        write=button.press,
        backend={"ros2": {"service": "std_srvs.srv.Trigger"}},
    )
    handler = get_service_handler("std_srvs.srv.Trigger")
    response = _Response()
    try:
        out = handler(
            object(),
            response,
            ctx,
            lambda p: run_on_physics(ctx, lambda _c: button.press(p)),
            endpoint,
        )
    finally:
        stop.set()
        thread.join(timeout=1.0)

    assert out.success is True
    assert out.message == "applied"
    assert button.presses == 1


def test_a_trigger_on_a_stalled_simulation_reports_that_it_did_not_land():
    """Not a silent success: the barrier is the only thing that can tell the caller."""
    button = _Button()
    ctx, stop, _thread = _wire(steps=False)  # nothing drains the queue
    endpoint = Endpoint(
        name="tare",
        direction="in",
        owner="ft",
        write=button.press,
        backend={"ros2": {"service": "std_srvs.srv.Trigger"}},
    )
    handler = get_service_handler("std_srvs.srv.Trigger")
    response = _Response()
    out = handler(object(), response, ctx, button.press, endpoint)
    stop.set()

    assert out.success is False
    assert "did not apply" in out.message


def test_every_service_a_shipped_plugin_declares_has_a_handler():
    """The bridge resolves a handler by type path and RAISES when there is none.

    So a plugin declaring a service type nobody serves does not fail at its own call -- it takes
    the whole bridge down at configure, for every world that lists that plugin. The declaration
    and the handler live in different packages, which is exactly how they come apart.

    Every plugin package is scanned, not just core: the declaration that first came apart this way
    was a sensor's, and a scan of ``roqsim.plugins`` alone would have watched the wrong shelf.
    """
    import importlib
    from importlib.metadata import entry_points

    declared: dict = {}
    for entry in entry_points(group="roqsim.plugins"):
        module_name = entry.value.split(":")[0]
        try:
            source = importlib.import_module(module_name).__file__
        except Exception:  # noqa: BLE001 - an optional extra's plugin is not this test's business
            continue
        if not source:
            continue
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        for marker in ('"service": "', "'service': '"):
            for chunk in text.split(marker)[1:]:
                declared.setdefault(chunk.split(marker[-1])[0], set()).add(entry.name)

    assert declared, "the scan found no service hints; has the declaration shape changed?"
    missing = {}
    for type_path, plugins in declared.items():
        try:
            get_service_handler(type_path)
        except KeyError:
            missing[type_path] = sorted(plugins)
    assert not missing, (
        f"declared by a plugin and served by nobody: {missing}. The bridge raises on this at "
        "configure, so every world listing that plugin fails to start. Add a handler in "
        "roqsim_ros_bridge.services, or advertise one from the declaring package."
    )
