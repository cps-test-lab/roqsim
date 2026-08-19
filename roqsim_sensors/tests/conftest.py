# SPDX-License-Identifier: Apache-2.0
"""Pick MuJoCo's offscreen GL backend before any test module in this package imports mujoco.

pytest imports ``conftest.py`` ahead of the test modules beside it, which is the only hook that runs
early enough: several of these modules ``import mujoco`` at the top, above their ``roqsim`` imports
(isort sorts third-party above first-party), and ``MUJOCO_GL`` is read exactly once, while ``import
mujoco`` runs. Without this, running *this directory on its own* with ``MUJOCO_GL`` unset bound glfw
and the camera tests died in ``check_gl_backend`` -- while ``make test`` stayed green, because it
collects roqsim's own suite first and that imports roqsim before anything touches mujoco.

roqsim's package ``__init__`` makes the same call for every entry point that reaches mujoco through
it; see roqsim/gl.py for why the ordering is load-bearing rather than cosmetic.
"""

from __future__ import annotations

from roqsim.gl import select_offscreen_gl

select_offscreen_gl()
