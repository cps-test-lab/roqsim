# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""How an action reaches the world: one seam, two transports, the same names.

An roqsim simulation is driven two ways, and a scenario action must work in both:

* **stepped, in-process** -- scenario-execution's own runner owns the loop, the simulator shares its
  process, and the action is handed the adapter in ``setup(**kwargs)`` as ``simulation``.
* **over ROS** -- the simulator is in another container. The action is handed a ``node`` instead, and
  everything it needs is a topic or a service.

Writing an action twice would be the obvious answer and the wrong one: the *semantics* are identical,
only the plumbing differs. So the plumbing is the abstraction, and the actions are written once
against it.

It works out cleanly because both transports already speak the same vocabulary -- which is not an
accident, it is :mod:`roqsim.context`'s ``Endpoint`` design one layer down (architecture.rst §13: a
capability is declared once, in-process callables plus inert per-backend hints the bridge turns into
topics and services). This module is the same idea on the scenario side:

===========================  ==================================  ====================================
need                         in-process                          over ROS
===========================  ==================================  ====================================
pose of entity ``X``         ``ctx.entities`` -> ``data.xpos``    ``get_entity_state``, keyed by
                                                                 entity NAME
apply/restore fault ``F``    blackboard ``model_override:F``      ``F/override`` (``SetBool``)
did it land                  ``read_state().verified``            the same service's reply
time                         the runner's ``clock``               the runner's ``clock``
===========================  ==================================  ====================================

Two things make it hold:

**The names do not change.** ``simulation_interfaces`` is keyed on ENTITY names, exactly like
``ctx.entities``, so ``'parcel'`` means one thing on both paths and no frame naming enters. TF is
deliberately not the ROS pose source: it would arrive with ``map -> odom`` localisation error folded
in (43 mm in x, 73 mm in y, measured -- the reason ``object_detector`` exists in the tiago world),
while ``get_entity_state`` is ground truth like the in-process read.

**Time is not asked of the transport.** ``Clock.now()`` is already the framework's abstraction:
``SimulationClock`` under the stepped runner, ``RosClock`` (i.e. ``/clock``) under the ROS one. An
action takes the clock it is handed and never knows which.

What DOES differ, and is stated rather than hidden: over ROS a pose is a service round-trip, so the
instant a threshold is crossed is resolved at the tick period rather than at the physics step. A dwell
shorter than one tick means "the first tick past the threshold" on both paths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class AccessError(RuntimeError):
    """The world cannot answer: no transport, an unknown entity, a missing plugin instance.

    Always an AUTHORING error -- a name that does not exist, a world without the plugin the scenario
    fires -- so an action turns it into ``ActionError``. A runtime verdict (the fault did not land) is
    not one of these; that is a result, and results are returned, not raised.
    """


@dataclass(frozen=True)
class Pose:
    """A body's world pose. ``quat`` is ``(w, x, y, z)``.

    MuJoCo's ``xquat`` order, which is also the order the bridge fills
    ``geometry_msgs/Quaternion`` in (``sim_interfaces._get_entity_state``), so the two transports
    hand back the same numbers in the same order and the caller never asks which it is talking to.
    """

    pos: np.ndarray
    quat: np.ndarray


@dataclass(frozen=True)
class OverrideOutcome:
    """What became of an apply/restore.

    ``ok`` is about the TRANSPORT and the simulator ("the command was applied"); ``verified`` is the
    plugin's own verdict about the physics (``landed`` / ``no_effect`` / ``untested``). They are
    separate because "the simulator never applied it" and "it applied and changed nothing" call for
    different messages, and only the caller knows whether either should fail the trial.
    """

    ok: bool
    verified: str
    detail: str


class OverrideCall(ABC):
    """An apply/restore in flight. ``poll()`` returns ``None`` until the outcome is known.

    Two-phase on both transports, for the same reason the plugin's inbound endpoint is a service
    rather than a topic: this is a command whose outcome the caller needs. In-process the wait is for
    ``ctx.post`` to be drained and the next ``post_step`` to have run; over ROS it is for the service
    future. Neither may block -- an action that blocks the tick either stalls the tree or, in the
    stepped shape, deadlocks the very step it is waiting for.
    """

    @abstractmethod
    def poll(self) -> OverrideOutcome | None: ...


@dataclass(frozen=True)
class TeleportOutcome:
    """What became of a teleport. ``ok`` is false only for an authoring-adjacent runtime fact that
    is still a result rather than a raise -- the named entity has no free joint to place (e.g. a
    static prop), which :meth:`WorldAccess.set_entity_pose` reports here rather than as
    :class:`AccessError`, because "this entity cannot be teleported" is a fact about the WORLD a
    campaign chose, not about the call being malformed.
    """

    ok: bool
    detail: str


class TeleportCall(ABC):
    """A pose write in flight. ``poll()`` returns ``None`` until the outcome is known.

    Two-phase for the same reason :class:`OverrideCall` is: in-process the wait is for ``ctx.post``
    to be drained by the next ``pre_step``; over ROS it is for the ``SetEntityState`` future.
    """

    @abstractmethod
    def poll(self) -> TeleportOutcome | None: ...


class WorldAccess(ABC):
    """The seam. See the module docstring."""

    #: For messages, so a failure says which transport answered.
    transport: str = "unknown"

    @abstractmethod
    def ready(self) -> bool:
        """Can the world be asked anything yet?

        False in the stepped shape until the world is built -- the tree is set up before the first
        ``reset()``, and a caller must wait a tick rather than trigger a compile.
        """

    @abstractmethod
    def entity_pose(self, name: str) -> Pose | None:
        """The entity's world pose, or ``None`` if it is not known YET (a reply in flight).

        Raises :class:`AccessError` when the name can never resolve, which is a different thing from
        not knowing yet and must not be confused with it.
        """

    #: Blackboard prefix per fault channel, and the only thing that differs between them in-process.
    #: The ROS backend needs no entry: it addresses a fault by its endpoint, which both channels
    #: scope the same way.
    OVERRIDE_KINDS = {
        "model": "model_override",  # roqsim.plugins.model_override -- a PHYSICS fault
        "sensor": "sensor_fault",  # roqsim_sensors.live_config -- a sensor's REPORT fault
    }

    @abstractmethod
    def apply_override(self, instance: str, active: bool, kind: str = "model") -> OverrideCall:
        """Switch a fault on or off. Never blocks.

        *kind* selects the channel (see :attr:`OVERRIDE_KINDS`); *instance* names the fault within
        it -- a ``model_override`` instance's ``name:`` for the physics channel, a component
        **address** (``robot.lidar``) for the sensor channel.
        """

    @abstractmethod
    def set_entity_pose(self, name: str, pos: np.ndarray, quat: np.ndarray) -> TeleportCall:
        """Teleport a free-jointed entity to ``pos`` (metres) / ``quat`` (w, x, y, z). Never blocks.

        For placing a robot at a per-configuration pose a MuJoCo compile cannot vary (a campaign's
        random start pose, unlike ``spawn_robot.pos`` in the world YAML): the mechanism a
        ``config_generation``-time factor cannot reach because it is decided per RUN, after the
        world already compiled. Zeroes the entity's velocity, matching a fresh spawn rather than a
        mid-flight relocation.
        """

    def teardown(self) -> None:
        """Drop anything the transport allocated. Called from the action's ``shutdown``."""


def select(kwargs: dict, *, what: str) -> WorldAccess:
    """Pick the backend from the setup kwargs the RUNNER provided.

    Not from configuration: which transport is present is a property of how the scenario is being
    executed, and a scenario that had to declare it would have to be edited to move between the two.
    ``simulation`` is offered by the stepped runner, ``node`` by the ROS one.

    Imported lazily, per backend, so that :mod:`scenario_execution_roqsim.access.ros` -- and with
    it ``rclpy`` and ``simulation_interfaces`` -- is never imported in a plain venv, and the
    in-process path never pays for MuJoCo at tree-build time either.
    """
    sim = kwargs.get("simulation")
    if sim is not None:
        from .in_process import InProcessAccess

        return InProcessAccess(sim)
    node = kwargs.get("node")
    if node is not None:
        from .ros import RosAccess

        return RosAccess(node)
    raise AccessError(
        f"{what} needs a simulation to talk to and the runner offered none. Either run the stepped "
        "runner with `--simulation <module>:<Class>` (scenario-execution's own binary, RoboVAST's "
        "`mode: base`), or run under the ROS runner, where the simulator is reached over "
        "simulation_interfaces. Note this action can never run under `remote()`: a remote server is "
        "handed neither."
    )


def clock_of(kwargs: dict):
    """The runner's clock. ``clock`` on the stepped runner, ``sim_clock`` on the ROS one.

    Both are ``scenario_execution.simulation.Clock``, so ``now()`` is sim-time seconds on either --
    which is what makes a dwell mean the same thing as ``timeout()`` and as a recorded timestamp.
    Wall clock is never an option: under ``pacing: asap`` the two differ by orders of magnitude.
    """
    return kwargs.get("clock") or kwargs.get("sim_clock")
