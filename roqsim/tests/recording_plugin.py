# SPDX-License-Identifier: Apache-2.0
"""The ``RecordingPlugin`` test double, in a module of its own rather than in ``conftest``.

It is imported by name from two places -- ``test_plugin_lifecycle`` imports the class, and the
registry resolves it from the plugin ref ``"recording_plugin:RecordingPlugin"`` -- so it needs a
module name that means one thing. ``conftest`` does not: pytest puts every test directory on
``sys.path``, so the moment a second package grew a ``conftest.py`` the bare name resolved to
whichever was imported first, and this class vanished from under the import. (Importing a conftest
directly is discouraged for exactly this reason.) A unique module name has no such ambiguity.
"""

from __future__ import annotations

from roqsim.plugin import Plugin


class RecordingPlugin(Plugin):
    """Appends ``(name, hook)`` to a shared list on every hook, to assert ordering."""

    #: class-level shared log so ordering across multiple plugin instances is observable
    LOG: list[tuple] = []

    def build(self, spec, ctx):
        self.LOG.append((self.name, "build"))

    def configure(self, ctx):
        self.LOG.append((self.name, "configure"))

    def on_reset(self, ctx):
        self.LOG.append((self.name, "on_reset"))

    def pre_step(self, ctx):
        self.LOG.append((self.name, "pre_step"))

    def post_step(self, ctx):
        self.LOG.append((self.name, "post_step"))

    def shutdown(self, ctx):
        self.LOG.append((self.name, "shutdown"))
