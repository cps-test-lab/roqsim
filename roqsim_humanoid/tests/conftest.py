# SPDX-License-Identifier: Apache-2.0
"""Pick MuJoCo's offscreen GL backend before any test module in this package imports mujoco.

pytest imports ``conftest.py`` ahead of the test modules beside it, which is the only hook that runs
early enough: these modules ``import mujoco`` at the top, above their ``roqsim`` imports (isort sorts
third-party above first-party), and ``MUJOCO_GL`` is read exactly once, while ``import mujoco`` runs.
A humanoid's manifest brings cameras that render every step (OLI's head camera), so a module run on
its own with ``MUJOCO_GL`` unset binds glfw and dies in ``check_gl_backend``.

Same reason and same call as ``roqsim_sensors/tests/conftest.py``; see roqsim/gl.py for why the
ordering is load-bearing rather than cosmetic.
"""

from __future__ import annotations

from roqsim.gl import select_offscreen_gl

select_offscreen_gl()
