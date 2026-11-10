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

"""The ``ROQSIM_*`` environment variables roqsim reads: one registry, and the check against it.

Every ``ROQSIM_*`` variable is read through :func:`get`, which refuses a name the registry does not
hold, so a variable cannot be read without being registered. A run's entry points (``roqsim sim``
and the scenario adapter) call :func:`check` first, which refuses any ``ROQSIM_*`` variable in the
environment that is not registered, as an unknown parameter: a misspelt or unread variable would
otherwise be ignored and the run would look correctly configured.

roqsim's own variables are :data:`VARIABLES`. Another package declares the ones it reads under the
``roqsim.environment`` entry-point group, as a mapping of name to a one-line description, so a
package's variable is known wherever that package is installed and nowhere else.

Only the standard library is imported here: :mod:`roqsim.gl` reads through this module while
``import roqsim`` is still selecting the GL backend.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping

from .exit_status import BAD_INPUT

#: The prefix that makes a variable roqsim's.
PREFIX = "ROQSIM_"

#: The entry-point group another package declares its ``ROQSIM_*`` variables under.
GROUP = "roqsim.environment"

#: The ``ROQSIM_*`` variables the ``roqsim`` package reads, each with what it sets.
VARIABLES: dict[str, str] = {
    "ROQSIM_CAPTURE_FPS": "samples per simulated second of a recording",
    "ROQSIM_CONTROL": "the control URI a simulator serves and its clients connect to",
    "ROQSIM_GL_PRELOADED": "set by roqsim on the viewer process it re-executes with libGLEW preloaded",
    "ROQSIM_GLTF_VALIDATOR": "'required' makes the glTF tests fail when the Khronos validator is absent",
    "ROQSIM_LOG_FORMAT": "the log line format: plain or stamped",
    "ROQSIM_NO_GL_PRELOAD": "do not preload libGLEW for a viewer window",
    "ROQSIM_NO_GL_SELECT": "leave the offscreen GL backend to MuJoCo's own default",
    "ROQSIM_RECORD": "record the run to this .mcap file",
    "ROQSIM_RECORD_EXCLUDE": "bodies and joints left out of the recording's decoded channels",
    "ROQSIM_RECORD_TRACKS": "bodies and joints the recording's decoded channels carry",
    "ROQSIM_ROS": "the scenario adapter publishes the world over ROS 2",
    "ROQSIM_SCENE_EXPORT_DIR": "the scenario adapter exports the built world here, as scene.glb",
    "ROQSIM_SEED": "the scenario adapter's noise seed",
    "ROQSIM_SIM_CONTROL": "the scenario adapter also serves the simulation_interfaces control plane",
    "ROQSIM_TF_NAMESPACE": "the scenario adapter publishes TF under this namespace",
    "ROQSIM_WORLD": "the world the scenario adapter runs",
    "ROQSIM_WORLD_OVERRIDES": "a world-overrides YAML the scenario adapter applies",
}


class UnknownEnvironmentError(ValueError):
    """A ``ROQSIM_*`` variable is set that roqsim does not read."""

    exit_status = BAD_INPUT


@functools.cache
def registry() -> dict[str, str]:
    """Every registered ``ROQSIM_*`` variable: :data:`VARIABLES` and each :data:`GROUP` entry's."""
    from importlib.metadata import entry_points

    known = dict(VARIABLES)
    for entry in entry_points(group=GROUP):
        declared = entry.load()
        if not isinstance(declared, Mapping):
            raise TypeError(
                f"{GROUP} entry {entry.name!r} ({entry.value}) must be a mapping of variable name "
                f"to description, not {type(declared).__name__}"
            )
        for name, text in declared.items():
            if not str(name).startswith(PREFIX):
                raise ValueError(
                    f"{GROUP} entry {entry.name!r} declares {name!r}, which lacks the {PREFIX} prefix"
                )
            known[str(name)] = str(text)
    return known


def get(name: str, default: str | None = None) -> str | None:
    """The value of the registered variable *name*, else *default*.

    Raises :class:`KeyError` for a name that is not registered: reading a variable is what puts it
    in the registry's scope, so an unregistered read is a bug in the reader.
    """
    if name not in VARIABLES and name not in registry():
        raise KeyError(
            f"{name} is not a registered roqsim environment variable: add it to "
            f"roqsim.env.VARIABLES, or declare it under the {GROUP} entry-point group"
        )
    return os.environ.get(name, default)


def check(environ: Mapping[str, str] | None = None) -> None:
    """Refuse a ``ROQSIM_*`` variable in *environ* (default: the process's) that is not registered.

    Raises :class:`UnknownEnvironmentError` naming each unknown variable and listing the ones
    roqsim reads.
    """
    from .document import refuse_unknown_keys

    environ = os.environ if environ is None else environ
    ours = {name: None for name in environ if name.startswith(PREFIX)}
    refuse_unknown_keys(
        ours, registry(), "ROQSIM_* environment variables", error=UnknownEnvironmentError
    )
