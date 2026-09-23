# SPDX-License-Identifier: Apache-2.0
"""Pick MuJoCo's offscreen GL backend before any test module in this package imports mujoco.

Same reason as ``roqsim_sensors/tests/conftest.py``: pytest imports ``conftest.py`` ahead of the test
modules beside it, which is the only hook that runs early enough. These modules ``import mujoco`` at
the top, above their ``roqsim`` imports (isort sorts third-party above first-party), and
``MUJOCO_GL`` is read exactly once, while ``import mujoco`` runs.

It bites here because a robot arrives with the sensors its manifest names: driving a TurtleBot 4
brings its OAK-D camera along, and constructing that needs an offscreen renderer. Running this
directory on its own with ``MUJOCO_GL`` unset binds glfw and those tests die in
``check_gl_backend`` -- while ``make test`` stays green, because it collects roqsim's own suite first
and that imports roqsim before anything touches mujoco.
"""

from __future__ import annotations

from roqsim.gl import select_offscreen_gl

select_offscreen_gl()
