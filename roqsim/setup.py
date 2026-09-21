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

"""Bakes the build identity into a regular install (see ``roqsim.build_identity``).

Everything else about the package is declared in ``pyproject.toml``; this file exists only for the
build step. An editable install is left alone: it runs the working tree, whose commit is read from
git each time rather than frozen at install.
"""

import importlib.util
import json
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_HERE = Path(__file__).resolve().parent
_MODULE = _HERE / "src" / "roqsim" / "build_identity.py"


def _build_identity_module():
    # By path, not `import roqsim`: the package's dependencies are not installed at build time.
    spec = importlib.util.spec_from_file_location("_roqsim_build_identity", _MODULE)
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: a dataclass looks its module up in sys.modules while it is built.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildPyWithIdentity(build_py):
    """``build_py`` that also writes ``roqsim/_build_identity.json`` into the build directory."""

    def run(self):
        super().run()
        if getattr(self, "editable_mode", False):
            return
        module = _build_identity_module()
        record = module.for_build(_HERE)  # raises when the commit cannot be determined
        target = Path(self.build_lib) / "roqsim" / module.BAKED_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record) + "\n", encoding="utf-8")


setup(cmdclass={"build_py": BuildPyWithIdentity})
