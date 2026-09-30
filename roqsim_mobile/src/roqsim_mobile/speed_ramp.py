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
which the new direction is gained at ``accel_limit``.
"""

from __future__ import annotations


def ramp_speed(current: float, target: float, accel: float, decel: float, dt: float) -> float:
    """``current`` moved one step of ``dt`` towards ``target`` (m/s); a limit of 0 is instant.

    A reversal brakes to rest at ``decel`` and spends what is left of the step gaining the new
    direction at ``accel``, so with the two limits equal the ramp is one straight line through zero.
    """
    if current * target < 0.0:
        stop_time = 0.0 if decel <= 0.0 else abs(current) / decel
        if stop_time >= dt:
            return current - decel * dt * (1.0 if current > 0.0 else -1.0)
        return _toward(0.0, target, accel, dt - stop_time)
    return _toward(current, target, decel if abs(target) < abs(current) else accel, dt)


def _toward(current: float, target: float, limit: float, dt: float) -> float:
    if limit <= 0.0:
        return target
    step = limit * dt
    return current + max(-step, min(step, target - current))
