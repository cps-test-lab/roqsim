"""``ur_msgs/SetSpeedSliderFraction``: the handler ``roqsim_manipulation`` registers for
``ur_driver_interface``'s ``set_speed_slider``, through the bridge's extension entry point.

Its response carries ``success`` and nothing else, so the reply is the producer's own verdict: a
fraction the producer refuses is ``success: false``, as it is on the UR driver.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("roqsim_manipulation")
pytest.importorskip("rclpy")

from roqsim_manipulation.plugins.ur_driver_interface import URDriverInterfacePlugin  # noqa: E402

from roqsim.context import Endpoint, SimContext  # noqa: E402
from roqsim_ros_bridge.registry import resolve_type  # noqa: E402
from roqsim_ros_bridge.services import get_service_handler  # noqa: E402

TYPE = "ur_msgs.srv.SetSpeedSliderFraction"


def _call(fraction: float, write=None):
    """Call the handler with real ur_msgs request and response messages; return the reply and the
    producer."""
    srv = resolve_type(TYPE)
    driver = URDriverInterfacePlugin({"ft": "ft"})
    ctx = SimContext(config={})
    stop = threading.Event()

    def physics():
        while not stop.is_set():
            ctx.drain_commands()
            stop.wait(0.001)

    thread = threading.Thread(target=physics, daemon=True)
    thread.start()
    endpoint = Endpoint(
        name="set_speed_slider",
        direction="in",
        owner="ur5e",
        # Untyped, so the handler hands it the bare fraction.
        write=write or (lambda p: ctx.submit(lambda _c: driver.set_speed_slider(p))),
        backend={"ros2": {"service": TYPE, "name": "io_and_status_controller/set_speed_slider"}},
    )
    try:
        reply = get_service_handler(TYPE)(
            srv.Request(speed_slider_fraction=fraction),
            srv.Response(),
            ctx,
            endpoint.write,
            endpoint,
        )
    finally:
        stop.set()
        thread.join(timeout=1.0)
    return reply, driver


def test_an_accepted_fraction_is_applied_and_answered_success():
    reply, driver = _call(0.4)
    assert reply.success is True
    assert driver.speed_scaling() == pytest.approx(40.0)


@pytest.mark.parametrize("fraction", [0.0, 1.5, -0.1])
def test_a_refused_fraction_is_answered_failure_and_changes_nothing(fraction):
    reply, driver = _call(fraction)
    assert reply.success is False
    assert driver.speed_scaling() == 100.0


def test_a_command_that_raises_is_answered_failure():
    ctx = SimContext(config={})

    def boom(_c):
        raise RuntimeError("no slider")

    reply, _ = _call(0.5, write=lambda _p: _raised(ctx, boom))
    assert reply.success is False


def _raised(ctx, fn):
    future = ctx.submit(fn)
    ctx.drain_commands()
    return future
