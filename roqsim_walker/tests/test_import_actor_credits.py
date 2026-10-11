"""Every walker blueprint carries complete credits: its own terms, and the CC-BY clips it walks with.

``roqsim walker import-actor`` writes a blueprint's ``CREDITS.txt``. Without a source, licence and
author it would write a blueprint with no recorded licence, which ``THIRD_PARTY.md`` says must not
be committed -- so the tool refuses to run rather than produce one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roqsim_walker.blueprint import models_dir
from roqsim_walker.cli.import_actor import credits_text, main

CREDITS = ("--credits-source", "--credits-licence", "--credits-author")
VALUES = ("https://example.org/actor", "CC0 1.0 Universal", "Someone")


@pytest.mark.parametrize("missing", CREDITS)
def test_an_import_without_full_credits_is_refused_before_writing(tmp_path, missing, capsys):
    argv = [str(tmp_path / "actor.dae"), "--name", "Actor", "--out", str(tmp_path / "people")]
    for flag, value in zip(CREDITS, VALUES, strict=True):
        if flag != missing:
            argv += [flag, value]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert missing in capsys.readouterr().err
    assert not (tmp_path / "people").exists(), "it wrote a blueprint before refusing"


def test_the_credits_name_the_source_and_the_clips_cc_by_attribution():
    text = credits_text("Actor", *VALUES)
    for value in VALUES:
        assert value in text
    assert "CARLA" in text and "CC-BY 4.0" in text


@pytest.mark.parametrize(
    "folder", sorted(p for p in (Path(models_dir()) / "people").iterdir() if p.is_dir())
)
def test_each_bundled_blueprint_credits_the_clips_it_walks_with(folder):
    text = (folder / "CREDITS.txt").read_text()
    assert "Licence:" in text and "UNKNOWN" not in text
    assert "CARLA" in text, f"{folder.name}/CREDITS.txt does not credit the CC-BY clips"
