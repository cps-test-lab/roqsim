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


def test_an_import_with_finger_bones_writes_their_weights_and_credits_them():
    from roqsim_walker.cli.import_actor import (
        FINGER_CHAINS,
        FINGERS_CREDIT,
        _finger_bone,
        has_fingers,
    )

    bones = [_finger_bone(*c) for c in FINGER_CHAINS]
    assert has_fingers(["Hips", *bones]) and not has_fingers(["Hips"])
    with pytest.raises(SystemExit, match="some of its finger bones"):
        has_fingers(["Hips", *bones[1:]])
    assert FINGERS_CREDIT in credits_text("Actor", *VALUES, fingers=True)
    assert FINGERS_CREDIT not in credits_text("Actor", *VALUES)


@pytest.mark.parametrize(
    "folder", sorted(p for p in (Path(models_dir()) / "people").iterdir() if p.is_dir())
)
def test_a_bundled_blueprints_finger_weights_are_credited_as_the_import_writes_them(folder):
    from roqsim_walker.cli.import_actor import FINGERS_CREDIT

    has_weights = any(folder.glob("*.fingers.npz"))
    assert (FINGERS_CREDIT in (folder / "CREDITS.txt").read_text()) == has_weights


def test_fingers_only_credits_its_weights_once_in_the_current_wording():
    from roqsim_walker.cli.import_actor import FINGERS_CREDIT, with_fingers_credit

    plain = credits_text("Actor", *VALUES)
    older = plain + "\nThe finger weights (*.fingers.npz) came from elsewhere.\n"
    for text in (plain, older, credits_text("Actor", *VALUES, fingers=True)):
        out = with_fingers_credit(text)
        assert out.count("The finger weights") == 1 and out.endswith(FINGERS_CREDIT)
    assert with_fingers_credit(plain) == credits_text("Actor", *VALUES, fingers=True)
