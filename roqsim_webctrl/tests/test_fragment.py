"""The shipped sim-control web.yaml fragment exists and enables the sim-control plugin."""

import yaml

from roqsim_webctrl import sim_control_fragment_path


def test_fragment_enables_sim_control():
    path = sim_control_fragment_path()
    assert path.is_file()
    data = yaml.safe_load(path.read_text())
    ids = [p.get("id") for p in data.get("plugins", [])]
    assert "sim-control" in ids
