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

"""The car-like drives' speed ramp: accel_limit gains speed, decel_limit takes it off."""

import pytest

from roqsim_mobile.speed_ramp import ramp_speed

DT = 0.01


def test_speeding_up_is_limited_by_accel():
    assert ramp_speed(0.5, 1.0, 0.5, 2.0, DT) == pytest.approx(0.505)


def test_slowing_down_is_limited_by_decel():
    assert ramp_speed(1.0, 0.5, 0.5, 2.0, DT) == pytest.approx(0.98)


def test_braking_backwards_is_limited_by_decel_too():
    assert ramp_speed(-1.0, 0.0, 0.5, 2.0, DT) == pytest.approx(-0.98)


def test_a_reversal_brakes_to_rest_and_then_pulls_away_at_accel():
    v, t = 0.8, 0.0
    while v > 0.0:
        v = ramp_speed(v, -0.8, 0.5, 2.0, DT)
        t += DT
    # rest is reached at 0.8 / 2.0 s; the rest of that step is spent pulling away at accel
    assert v == pytest.approx(-0.5 * (t - 0.8 / 2.0))
    assert ramp_speed(v, -0.8, 0.5, 2.0, DT) == pytest.approx(v - 0.005)


def test_with_equal_limits_a_reversal_is_one_line_through_zero():
    assert ramp_speed(0.003, -1.0, 1.0, 1.0, DT) == pytest.approx(-0.007)
    assert ramp_speed(0.003, -0.002, 1.0, 1.0, DT) == pytest.approx(-0.002)


def test_it_stops_on_the_target_rather_than_past_it():
    assert ramp_speed(0.001, 0.0, 0.5, 2.0, DT) == 0.0
    assert ramp_speed(0.999, 1.0, 0.5, 2.0, DT) == 1.0


def test_a_zero_limit_is_instant():
    assert ramp_speed(0.0, 1.0, 0.0, 2.0, DT) == 1.0
    assert ramp_speed(1.0, 0.0, 0.5, 0.0, DT) == 0.0
    # an instant stop, then the new direction gained at accel for the whole step
    assert ramp_speed(1.0, -1.0, 0.5, 0.0, DT) == pytest.approx(-0.005)
