"""``fetch_policy`` fails when it has nothing to fetch with, rather than reporting success."""

from __future__ import annotations

from roqsim_quadruped.policy import fetch_policy


def test_a_missing_runner_is_a_failed_fetch(monkeypatch, tmp_path, capsys):
    # An installed wheel has no source tree around it, so the external-resources runner is absent.
    monkeypatch.setattr(fetch_policy, "_RUNNER", tmp_path / "external" / "external_resources.py")
    assert fetch_policy.fetch() != 0
    assert "make external-resources RESOURCE=spot_locomotion_policy" in capsys.readouterr().err
