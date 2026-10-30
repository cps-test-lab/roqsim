# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Whether a gripper holds a part: by its pads' contacts and by how far the part slips.

A gripper that closed is not one that holds. A position-servo gripper closing on a thin part stops
near its own end of travel and reports the same "reached" whether the part is between its fingers
or not; a part that slips moves without any command saying so. So a hold is measured on what the
physics says:

- :func:`pads_touch` -- each of the gripper's pads is in contact with the part now.
- :class:`HoldMonitor` -- from the moment of the grip, where the part is relative to the gripper's
  site: the slip of the part's material point that was at the site when it was gripped (m), the
  part's turn against the gripper since (rad), and every step on which a pad lost contact. The slip
  is measured where the fingers hold, not at the part's origin, where a small turn of a long part
  would read as a slip it is not. A part held by friction alone, carried through a move, should drift by a fraction of a
  millimetre and never lose a pad.
"""

from __future__ import annotations

import numpy as np

from .collision import body_geoms, subtree_bodies
from .kinematics import rotation_error


def pads_touch(model, data, pads, part_body: int) -> bool:
    """Whether every geom of ``pads`` touches a geom of the part's subtree now."""
    part = body_geoms(model, subtree_bodies(model, part_body))
    touching = set()
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in pads and g2 in part:
            touching.add(g1)
        elif g2 in pads and g1 in part:
            touching.add(g2)
    return touching >= set(pads)


class HoldMonitor:
    """The drift of a part relative to a site since it was gripped, and lost pad contacts."""

    def __init__(self, model, site: int, pads, part_body: int) -> None:
        self.model = model
        self.site = int(site)
        self.pads = {int(g) for g in pads}
        self.part = int(part_body)
        self._rel_pos = None
        self._rel_rot = None
        self.max_drift = 0.0
        self.max_turn = 0.0
        self.steps = 0
        self.lost = 0

    def _relative(self, data):
        """The site's position in the part's frame, and the part's rotation in the site's frame."""
        sp, sr = data.site_xpos[self.site], data.site_xmat[self.site].reshape(3, 3)
        pr = data.xmat[self.part].reshape(3, 3)
        rel_pos = pr.T @ (sp - data.xpos[self.part])
        rel_rot = sr.T @ pr
        return rel_pos, rel_rot

    def grip(self, data) -> None:
        """Note where the part is relative to the site: the grip every later step is compared with."""
        self._rel_pos, self._rel_rot = self._relative(data)
        self.max_drift = self.max_turn = 0.0
        self.steps = self.lost = 0

    @property
    def gripping(self) -> bool:
        return self._rel_pos is not None

    def update(self, data) -> None:
        """Measure one step while the part is held."""
        if self._rel_pos is None:
            return
        pos, rot = self._relative(data)
        # The material point that was at the site, against where the site is now, in metres.
        self.max_drift = max(self.max_drift, float(np.linalg.norm(pos - self._rel_pos)))
        self.max_turn = max(
            self.max_turn, float(np.linalg.norm(rotation_error(self._rel_rot, rot)))
        )
        self.steps += 1
        if not pads_touch(self.model, data, self.pads, self.part):
            self.lost += 1

    def release(self) -> dict:
        """End the hold; what it measured."""
        out = {
            "max_drift_m": self.max_drift,
            "max_turn_rad": self.max_turn,
            "steps": self.steps,
            "steps_without_both_pads": self.lost,
        }
        self._rel_pos = self._rel_rot = None
        return out
