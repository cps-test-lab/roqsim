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

"""The commanded-speed ramp of the car-like drives: one rate to gain speed, another to lose it.

A vehicle brakes harder than it accelerates -- a truck's drive and brakes are different machines --
so ``accel_limit`` and ``decel_limit`` are separate keys. Braking is any step that takes speed off:
towards a slower target in the same direction, to a stop, or, on a reversal, down to rest, after
which the new direction is gained at ``accel_limit``. ``decel_limit`` defaults to ``accel_limit``.
"""

from __future__ import annotations


def ramp_speed(current: float, target: float, accel: float, decel: float, dt: float) -> float:
    """``current`` moved one step of ``dt`` towards ``target`` (m/s); a limit of 0 is instant."""
    reversing = current * target < 0.0
    braking = reversing or abs(target) < abs(current)
    # A reversal stops first: the part of the change below rest is not braking.
    goal = 0.0 if reversing else target
    limit = decel if braking else accel
    if limit <= 0.0:
        return goal
    step = limit * dt
    return current + max(-step, min(step, goal - current))


def decel_limit_from(config: dict, accel_limit: float) -> float:
    """``decel_limit`` from a drive's config, ``accel_limit`` when it states none."""
    return float(config.get("decel_limit", accel_limit))
