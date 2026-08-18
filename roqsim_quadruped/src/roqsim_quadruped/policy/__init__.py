"""Bundled locomotion deploy config + the (not-committed) pretrained policy.

``spot.yaml`` is our deploy config (joint order, default pose, action scale, PD gains, obs layout,
timing) derived from NVIDIA's Isaac ``Isaac-Velocity-Flat-Spot-v0`` env -- vendored so the policy
runs on exactly the conventions it was trained on.

``spot_policy.pt`` is the pretrained TorchScript flat-terrain policy. It is **NVIDIA-licensed and not
redistributable**, so it is *not committed* to this repo (see ``.gitignore``). Fetch it locally with::

    python -m roqsim_quadruped.policy.fetch_policy

or point ``spot_locomotion``'s ``policy_path`` (or the ``SPOT_POLICY_PATH`` env var) at your own copy.
The ``spot_locomotion`` plugin loads both from :data:`POLICY_DIR` unless its config overrides them.
"""

from __future__ import annotations

from pathlib import Path

POLICY_DIR = Path(__file__).parent
DEFAULT_POLICY = POLICY_DIR / "spot_policy.pt"  # git-ignored; fetched by fetch_policy.py
DEFAULT_CONFIG = POLICY_DIR / "spot.yaml"
