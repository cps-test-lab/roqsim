"""Coverage reads a world's geometry, so its transport plugins are left out of the build."""

from __future__ import annotations

from roqsim_sensors.coverage import cli

from roqsim.plugin import Plugin


class Transport(Plugin):
    """A transport that would bind a port and wait for a peer, if it were configured."""

    transport_only = True
    configured = False

    def configure(self, ctx):
        Transport.configured = True


def test_a_transport_plugin_is_left_out_of_the_coverage_build(tmp_path):
    world = tmp_path / "w.yaml"
    world.write_text(
        'sim: {}\ncomponents:\n  - "test_coverage_load_world:Transport": {}\n    name: t\n'
    )
    model, _data = cli.load_world(str(world))
    assert model.nbody >= 1
    assert not Transport.configured
