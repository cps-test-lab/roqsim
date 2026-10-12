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

"""Which of a model's lights MuJoCo's renderer leaves dark.

MuJoCo's OpenGL renderer draws at most :data:`RENDERED_LIGHTS` lights in a frame, the headlight
included when it is on, and takes the model's active lights in index order: every light after
that is compiled, listed and never drawn, whatever the camera. Nothing says so -- a world with a
lamp per bay simply renders with its last bays dark, and an image or a camera's training data is
made under lighting other than the world's.

:func:`undrawn_lights` names those lights, so the engine can say it when the world compiles.
"""

from __future__ import annotations

import mujoco

#: The lights MuJoCo's renderer draws in one frame (the fixed-function pipeline's eight), the
#: headlight among them when it is active.
RENDERED_LIGHTS = 8


def undrawn_lights(model: mujoco.MjModel) -> list[str]:
    """The active lights the renderer never draws, by name (``light <index>`` for an unnamed one)."""
    room = RENDERED_LIGHTS - (1 if model.vis.headlight.active else 0)
    active = [i for i in range(model.nlight) if model.light_active[i]]
    return [model.light(i).name or f"light {i}" for i in active[room:]]


def summary(undrawn: list[str], model: mujoco.MjModel) -> str:
    """One line for the run's log."""
    active = sum(1 for i in range(model.nlight) if model.light_active[i])
    head = " plus the headlight" if model.vis.headlight.active else ""
    shown = ", ".join(undrawn[:6]) + (f" and {len(undrawn) - 6} more" if len(undrawn) > 6 else "")
    return (
        f"the world has {active} active lights{head}; MuJoCo's renderer draws {RENDERED_LIGHTS} "
        f"in index order, so {len(undrawn)} are never drawn: {shown}"
    )
