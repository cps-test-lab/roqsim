# SPDX-License-Identifier: Apache-2.0
"""Hooks for running every package's tests in one process, as ``make test`` does.

A built world is a reference cycle: the ``SimContext`` holds the plugins' endpoints and blackboard
entries, and those hold the plugins, which hold the context. So a finished test's ``MjModel``,
``MjData`` and renderer buffers are freed only when the cycle collector reaches them. Their memory
is allocated in C, invisible to the collector's thresholds, which count Python objects, and from
Python 3.14 on the collector reaches old cycles so late that a test process grows by gigabytes of
dead worlds. So each test is followed by a full collection.

A full collection walks every tracked object, and most of them are the modules the first tests
imported (mujoco, torch, scipy), which live as long as the process. They are frozen out of the
collector once, after the first test, so a collection costs what the tests themselves left behind.
"""

from __future__ import annotations

import gc

import pytest

_frozen = False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Once a test's fixtures are torn down, free the worlds it built."""
    global _frozen
    yield
    gc.collect()
    if not _frozen:
        gc.freeze()
        _frozen = True
