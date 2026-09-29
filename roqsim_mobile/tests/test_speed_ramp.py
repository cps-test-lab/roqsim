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

from roqsim_mobile.speed_ramp import decel_limit_from, ramp_speed

DT = 0.01


def test_speeding_up_is_limited_by_accel():
    assert ramp_speed(0.5, 1.0, 0.5, 2.0, DT) == pytest.approx(0.505)


def test_slowing_down_is_limited_by_decel():
    assert ramp_speed(1.0, 0.5, 0.5, 2.0, DT) == pytest.approx(0.98)


def test_braking_backwards_is_limited_by_decel_too():
    assert ramp_speed(-1.0, 0.0, 0.5, 2.0, DT) == pytest.approx(-0.98)


def test_a_reversal_brakes_to_rest_and_then_pulls_away_at_accel():
    v, steps_braking = 0.8, 0
    while v > 0.0:
        v = ramp_speed(v, -0.8, 0.5, 2.0, DT)
        steps_braking += 1
    assert v == 0.0
    assert steps_braking * DT == pytest.approx(0.8 / 2.0, abs=DT)
    assert ramp_speed(v, -0.8, 0.5, 2.0, DT) == pytest.approx(-0.005)


def test_it_stops_on_the_target_rather_than_past_it():
    assert ramp_speed(0.001, 0.0, 0.5, 2.0, DT) == 0.0
    assert ramp_speed(0.999, 1.0, 0.5, 2.0, DT) == 1.0


def test_a_zero_limit_is_instant():
    assert ramp_speed(0.0, 1.0, 0.0, 2.0, DT) == 1.0
    assert ramp_speed(1.0, 0.0, 0.5, 0.0, DT) == 0.0
    # an instant stop still stops before the reversal ramps
    assert ramp_speed(1.0, -1.0, 0.5, 0.0, DT) == 0.0


def test_decel_defaults_to_accel():
    assert decel_limit_from({}, 0.7) == 0.7
    assert decel_limit_from({"decel_limit": 1.5}, 0.7) == 1.5
