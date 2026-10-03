"""``--lock`` never succeeds without writing a lock file.

A world whose models are inline or resolved through ``--model-path`` fetches nothing, so there is no
asset whose URI and digest the importer could record. Importing it anyway would leave a loadable
scene with nothing saying where its geometry came from -- the state a lock file exists to prevent.
"""

from __future__ import annotations

from roqsim import exit_status
from roqsim_scenes.cli import sdf_to_scene

# One inline box: no meshes, so no external asset can be resolved however `--model-path` is set.
MESHLESS_WORLD = """<?xml version="1.0"?>
<sdf version="1.6">
  <world name="t">
    <model name="b1">
      <static>true</static>
      <pose>0 0 0.5 0 0 0</pose>
      <link name="l">
        <visual name="v"><geometry><box><size>1 1 1</size></box></geometry></visual>
        <collision name="c"><geometry><box><size>1 1 1</size></box></geometry></collision>
      </link>
    </model>
  </world>
</sdf>
"""


def _world(tmp_path):
    w = tmp_path / "w.world"
    w.write_text(MESHLESS_WORLD)
    return w


def test_lock_with_nothing_to_pin_is_refused_as_bad_input(tmp_path, capsys):
    out = tmp_path / "out"
    lock = out / "assets.lock.json"
    rc = sdf_to_scene.main(
        [
            "--world",
            str(_world(tmp_path)),
            "--out-dir",
            str(out),
            "--scene-name",
            "t",
            "--lock",
            str(lock),
        ]
    )
    assert rc == exit_status.BAD_INPUT
    err = capsys.readouterr().err
    assert err.startswith("roqsim scenes sdf-to-scene: ")
    assert "nothing to pin" in err
    # The message says what to pin instead, or it is only a refusal.
    assert "repository" in err and "licence" in err
    assert not lock.exists()
    assert not (out / "scene.json").exists(), "a refused import writes no scene either"


def test_the_same_import_succeeds_without_lock(tmp_path):
    """The refusal is about the unmet ``--lock``, not about the world being importable."""
    out = tmp_path / "out"
    rc = sdf_to_scene.main(
        ["--world", str(_world(tmp_path)), "--out-dir", str(out), "--scene-name", "t"]
    )
    assert rc == exit_status.OK
    assert (out / "scene.json").exists()
