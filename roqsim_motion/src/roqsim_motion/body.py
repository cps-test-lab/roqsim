# SPDX-License-Identifier: Apache-2.0
"""What the moves need of a humanoid body, whichever body it is.

A body -- the walker's mocap skeleton (``roqsim_walker.handling.WalkerBody``) or a humanoid robot's
joints (:class:`roqsim_motion.robot.RobotBody`) -- is published on the blackboard as
``humanoid_body:<entity>`` and offers the moves (:mod:`roqsim_motion.moves`) one interface: each
hand's goal (a world pose its grip frame goes to, eased in and out), its pose (``open``, ``power``,
...), the posture and the gaze, where its grip frame is now, and how far its arm reaches. Every
method runs on the physics thread; the body solves and writes at its own update.
"""

from __future__ import annotations

import numpy as np

from .hands import HandState
from .ik import Posture

SIDES = ("l", "r")


class HumanoidBody:
    """The state every body shares; a body implements the methods that raise here."""

    #: Whether the body's hands touch by contact proxies the moves switch (a walker's), or by its
    #: own geometry (a robot's).
    has_proxies = False

    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.goal = {s: None for s in SIDES}  # side -> (pos, quat | None) world
        self.hand = {s: HandState() for s in SIDES}
        self.closing = {s: None for s in SIDES}  # side -> touching(finger) while a grip closes
        self.posture: Posture | None = None
        self.gaze = None

    # -- set by the moves -------------------------------------------------------------------------
    def set_hand(self, side: str, pos, quat=None) -> None:
        """Put ``side``'s grip frame on a world pose (``quat`` ``None``: the position only)."""
        self.goal[side] = (np.asarray(pos, float), None if quat is None else np.asarray(quat, float))

    def free_hand(self, side: str) -> None:
        """Let ``side`` go back to the gait, easing out."""
        self.goal[side] = None
        self.closing[side] = None

    def set_pose(self, side: str, pose: str, touching=None) -> None:
        """Blend ``side`` toward a named pose; with ``touching(finger) -> bool``, each finger stops
        where it meets what it closes on."""
        self.hand[side].set_pose(pose)
        self.closing[side] = touching

    def hand_done(self, side: str) -> bool:
        """``side``'s fingers have arrived at its pose, or stopped on what they hold."""
        return self.hand[side].advance(0.0)

    def set_posture(self, posture: Posture | None) -> None:
        self.posture = posture

    def set_gaze(self, point) -> None:
        self.gaze = None if point is None else np.asarray(point, float)

    def press_proxy(self, data, side: str, on: bool) -> None:
        """Pose ``side``'s press proxy on the hand (``on``) or park it; a body without proxies
        touches with its own geometry and ignores this."""

    # -- what a body answers ------------------------------------------------------------------
    def grip(self, data, side: str):
        """``side``'s grip frame now: world ``(position, rotation)``."""
        raise NotImplementedError

    def grip_site(self, side: str) -> str:
        """The name of ``side``'s grip site in the model (a hold welds it)."""
        raise NotImplementedError

    def shoulder(self, data, side: str):
        raise NotImplementedError

    def arm_reach(self, side: str) -> float:
        """How far ``side``'s grip frame reaches from its shoulder, standing (m)."""
        raise NotImplementedError

    def root(self) -> tuple[float, float, float]:
        """The nav root ``(x, y, yaw)``."""
        raise NotImplementedError

    def face(self, yaw) -> None:
        """Turn the standing body to ``yaw`` (world, rad); ``None``: wherever it walks."""
        raise NotImplementedError

    def settled(self, side: str) -> bool:
        """``side``'s goal has fully taken over from the gait (or it has none)."""
        raise NotImplementedError
