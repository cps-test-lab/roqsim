"""`--lock` must never succeed silently without writing a lock file.

A world whose geometry is vendored in its own source tree resolves every `model://` locally, so
there is no fetched asset whose URI and digest the importer could record. Writing nothing in that
case is the worst available outcome: the import succeeds, the scene loads, and nothing says where
the geometry came from -- which is exactly the state a lock file exists to prevent.
"""

from __future__ import annotations

import pytest

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


def test_lock_with_nothing_to_pin_fails_loudly(tmp_path):
    w = _world(tmp_path)
    lock = tmp_path / "out" / "assets.lock.json"
    with pytest.raises(SystemExit) as e:
        sdf_to_scene.main(["--world", str(w), "--out-dir", str(tmp_path / "out"),
                           "--scene-name", "t", "--lock", str(lock)])
    msg = str(e.value)
    assert "nothing to pin" in msg
    # The message has to say what to do instead, or it is just a refusal.
    assert "repository" in msg and "licence" in msg
    assert not lock.exists(), "refused, so no misleading half-written lock file may remain"


def test_the_same_import_succeeds_without_lock(tmp_path):
    """The refusal is about the unmet `--lock` promise, not about the world being importable."""
    w = _world(tmp_path)
    out = tmp_path / "out"
    sdf_to_scene.main(["--world", str(w), "--out-dir", str(out), "--scene-name", "t"])
    assert (out / "scene.json").exists()
