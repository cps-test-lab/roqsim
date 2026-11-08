# SPDX-License-Identifier: Apache-2.0
"""Tests that need more than the package: run where it is there, deselected and named where it is not.

Two kinds, each a marker:

- ``preview``: the ``roqsim_scenes[preview]`` extra (ezdxf, matplotlib). ``make test`` installs it.
- ``gltf_validator``: every bundled world through the Khronos glTF Validator, a Node.js program.
  ``make test-gltf`` installs it and sets ``ROQSIM_GLTF_VALIDATOR=required``, and the test then fails
  when the validator is missing rather than skipping.

Where one is missing its tests are deselected, not skipped: the run's header says which and why, and
the summary counts them, so a run without them reads as what it is rather than as a skipped check.
"""

from __future__ import annotations

import importlib.util

from roqsim import env

_PREVIEW = ("ezdxf", "matplotlib")


def _missing() -> dict[str, str]:
    """Marker -> why its tests cannot run here (absent when they can)."""
    out = {}
    absent = [m for m in _PREVIEW if importlib.util.find_spec(m) is None]
    if absent:
        out["preview"] = f"roqsim_scenes[preview] is not installed ({', '.join(absent)} missing)"
    if env.get("ROQSIM_GLTF_VALIDATOR") != "required":
        out["gltf_validator"] = "the Khronos glTF Validator runs under `make test-gltf`"
    return out


def pytest_configure(config):
    config.addinivalue_line("markers", "preview: needs the roqsim_scenes[preview] extra")
    config.addinivalue_line(
        "markers", "gltf_validator: needs the Khronos glTF Validator (`make test-gltf`)"
    )


def pytest_collection_modifyitems(config, items):
    missing = _missing()
    keep, drop = [], []
    for item in items:
        (drop if any(item.get_closest_marker(m) for m in missing) else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def pytest_report_header(config):
    return [f"roqsim_scenes: '{m}' tests deselected: {why}" for m, why in _missing().items()]
