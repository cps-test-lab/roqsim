"""Fetch the pretrained Boston Dynamics Spot flat-terrain policy (NVIDIA Isaac).

    python -m roqsim_quadruped.policy.fetch_policy [--force]

Thin wrapper around the repo's external-resources system. The policy is declared as the
``spot_locomotion_policy`` resource in ``roqsim/external/external_assets.yaml`` (its URLs, target
paths, and NVIDIA license note), and this delegates to ``roqsim/external/external_resources.py`` so
there is a single place that lists the asset, git-ignores it, and fetches it -- the same system the
Livox meshes use.

``spot_policy.pt`` / ``spot_env.yaml`` are **NVIDIA-asset-licensed: local/internal R&D use only, not
modifiable or redistributable**, so they are fetched locally and never committed. The fetch is
fail-soft (the resource is marked ``optional``): a network error warns and returns 0 so ``make venv``
and offline builds never break; the ``spot_locomotion`` plugin raises a clear error at load time if the
policy is still missing. Point the plugin's ``policy_path`` / ``$SPOT_POLICY_PATH`` at your own copy to
use a different one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from . import POLICY_DIR

RESOURCE = "spot_locomotion_policy"
# This file lives at <roqsim>/roqsim_quadruped/src/roqsim_quadruped/policy/ in the editable
# source tree, so the external-resources runner is four parents up (works for both this repo and the
# parent repo, which vendor the same roqsim tree).
_RUNNER = POLICY_DIR.parents[3] / "external" / "external_resources.py"


def fetch(force: bool = False) -> int:
    if not _RUNNER.exists():
        print(
            f"[fetch_policy] external-resources runner not found at {_RUNNER}.\n"
            f"[fetch_policy] From the roqsim tree run: make external-resources RESOURCE={RESOURCE}",
            file=sys.stderr,
        )
        return 0
    cmd = [sys.executable, str(_RUNNER), "convert", "--resource", RESOURCE]
    if force:
        cmd.append("--force")
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if the policy exists")
    return fetch(force=ap.parse_args().force)


if __name__ == "__main__":
    raise SystemExit(main())
