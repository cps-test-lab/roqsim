# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""How the ROS bridge serves this package's own service types.

Imported by the bridge through the ``roqsim_ros_bridge.extensions`` entry point, so nothing else in
the package imports the bridge. Free of ROS imports, as the bridge's own service registry is: a
handler touches only the request's and response's members, and the type is resolved by the bridge
when an endpoint asks for it.

``ur_msgs/SetSpeedSliderFraction`` is ``ur_driver_interface``'s ``set_speed_slider``. Its response is
a bare ``success``, with no message to carry a reason, so the reply is the command's own verdict and
anything that kept the command from giving one is logged.
"""

from __future__ import annotations

import logging

from roqsim.context import CommandFuture
from roqsim_ros_bridge import physics
from roqsim_ros_bridge.params import payload_for
from roqsim_ros_bridge.services import service_handler

logger = logging.getLogger(__name__)


@service_handler("ur_msgs.srv.SetSpeedSliderFraction")
def set_speed_slider_fraction(request, response, ctx, on_payload, endpoint=None):  # noqa: ARG001
    """Hand the fraction to the producer and reply with whether it accepted it.

    The producer's command returns ``True`` for a fraction it applied and ``False`` for one it
    refused, as the driver's ``GPIOController`` answers; a command that raised, or that the physics
    thread did not run in time, is ``success: false`` too.
    """
    fraction = float(request.speed_slider_fraction)
    outcome = on_payload(payload_for(endpoint, {"speed_slider_fraction": fraction}))
    if not isinstance(outcome, CommandFuture):
        raise TypeError(
            "ur_msgs/SetSpeedSliderFraction is served for a command, whose future carries the "
            f"verdict; the endpoint returned {type(outcome).__name__}"
        )
    try:
        accepted = outcome.result(physics.DEFAULT_TIMEOUT_S)
    except TimeoutError:
        logger.warning(
            "set_speed_slider: the simulation did not apply the command within %s s",
            physics.DEFAULT_TIMEOUT_S,
        )
        accepted = None
    except Exception as exc:  # noqa: BLE001 - the producer's failure is the reply
        logger.warning("set_speed_slider failed: %s: %s", type(exc).__name__, exc)
        accepted = None
    if accepted is False:
        logger.warning("set_speed_slider: fraction %s refused", fraction)
    response.success = accepted is True
    return response
