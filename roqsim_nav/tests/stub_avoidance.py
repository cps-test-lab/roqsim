"""An AvoidanceModel that avoids nothing and records everything.

Not in ``conftest.py``: it must be reachable by a ``file.py:Class`` reference, which is one of the
three forms the registry supports and a conftest is not a file a world can name.
"""

from __future__ import annotations

import numpy as np

from roqsim_nav.avoidance import AvoidanceModel


class RecordingModel(AvoidanceModel):
    """Records the call sequence so the interface's ordering contract can be asserted."""

    params_schema = ("gain",)

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.agents: dict[int, dict] = {}
        self.statics: list = []
        self.submissions: dict[int, dict] = {}
        self.solves = 0
        self.resets = 0
        self._next = 0
        self.configured_with: dict | None = None

    def configure(self, ctx, params):
        self.calls.append("configure")
        self.configured_with = dict(params)

    def add_agent(self, key, *, radius, max_speed, yields, params):
        aid = self._next
        self._next += 1
        self.agents[aid] = {
            "key": key,
            "radius": radius,
            "max_speed": max_speed,
            "yields": yields,
            "params": params,
        }
        self.calls.append(f"add_agent:{key}")
        return aid

    def add_static(self, polygons):
        self.statics.extend(polygons)
        self.calls.append(f"add_static:{len(polygons)}")

    def submit(self, aid, pos, vel, pref_vel, *, present=True):
        self.submissions[aid] = {
            "pos": np.asarray(pos, float).copy(),
            "vel": np.asarray(vel, float).copy(),
            "pref": np.asarray(pref_vel, float).copy(),
            "present": present,
        }
        self.calls.append("submit")

    def solve(self, dt):
        self.solves += 1
        self.calls.append("solve")

    def result(self, aid):
        self.calls.append("result")
        sub = self.submissions.get(aid)
        if sub is None:
            return np.zeros(2)
        if not self.agents.get(aid, {}).get("yields", True):
            return sub["pref"]
        # A yielding agent is deflected by a fixed amount, so a test can see the model was consulted.
        return sub["pref"] + np.array([0.0, 0.05])

    def reset(self):
        self.resets += 1
        for sub in self.submissions.values():
            sub["pref"] = np.zeros(2)


class NotAModel:
    """Not an AvoidanceModel subclass, so the registry must refuse it."""
