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

"""MuJoCo's warnings, as records of the ``roqsim.mujoco`` logger.

Without a warning handler, MuJoCo prints each warning to stdout and appends it to
``MUJOCO_LOG.TXT`` in the current directory, so a process that runs in a source tree or a results
directory leaves that file behind. ``import roqsim`` calls :func:`route_warnings`, which installs a
handler that logs the warning at ``WARNING`` instead; MuJoCo then writes no file.

A handler installed before ``import roqsim`` is kept, and one installed after replaces this one.
``mujoco.set_mju_user_warning(None)`` restores MuJoCo's own printing and file.

MuJoCo's errors are left alone: the Python bindings already turn each into a raised
``mujoco.FatalError``, and expose no setter for the error handler.
"""

from __future__ import annotations

import logging

import mujoco

LOGGER = logging.getLogger("roqsim.mujoco")


def _log_warning(message: str) -> None:
    LOGGER.warning("%s", message.strip())


def route_warnings() -> bool:
    """Log MuJoCo's warnings through :data:`LOGGER` unless another handler is installed.

    Returns whether this handler is the one in effect. Idempotent.
    """
    current = mujoco.get_mju_user_warning()
    if current is None:
        mujoco.set_mju_user_warning(_log_warning)
        return True
    return current is _log_warning
