"""Bundled humanoid MJCF models and meshes.

A model ships a ``<model>.manifest.yaml`` alongside its MJCF listing the plugins intrinsic to it
(``g1_locomotion``, ``lidar``); ``spawn_robot`` pulls it in via
:func:`roqsim.manifest.expand_manifest`. Registered as a ``roqsim.models`` provider so any
world can spawn ``unitree_g1``.
"""

from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).parent
MESHES_DIR = MODELS_DIR / "meshes"
