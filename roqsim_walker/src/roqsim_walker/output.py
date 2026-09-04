"""The ``walker`` embodiment: a navigator's motion, rendered as a walking human.

Registered into ``roqsim_nav.outputs``, which is what makes a pedestrian the *same* navigator a
robot and a prop use. Everything above the output line -- the A* plan, the behaviour tree, the
caution probe, the goal interface, local avoidance -- is shared. What is walker-specific is only
this: the nav root moves, and the body has to look like it walked there.

That is the whole of the difference, and it is a real one. A wheeled base takes a twist and its
drive turns wheels; a prop takes a pose. A walker takes a preferred velocity, integrates the nav
root, then runs CARLA's 2D blendspace -- a speed axis (idle / short shuffle / walk / run) and a
direction axis (turn-in-place, lean, strafe) -- phased by *distance travelled* rather than by time,
so the feet do not slide, and writes all seventeen mocap bodies through forward kinematics.

The gait is what a viewer sees, so this output asks for a faster tick than planning needs
(:attr:`update_hz`), rather than the navigator special-casing a pedestrian by name.
"""

from __future__ import annotations

import numpy as np

from roqsim_nav.outputs import NavOutput, OutputUnavailable
from roqsim_walker.nav.controller import animate

#: Blackboard key under which the ``walker`` plugin publishes the animation state it built. The
#: geometry (mocap bodies, skin) and its clip set belong to the plugin that put them in the model;
#: this output only drives them.
STATE_KEY = "walker:anim"

#: The gait is watched, not merely arrived at. 20 Hz is ample to plan at and visibly judders a walk
#: cycle, so this embodiment asks for the rate the pedestrian stack has always run at.
WALKER_UPDATE_HZ = 60.0


class WalkerOutput(NavOutput):
    """Moves a walker's nav root and animates its body to match."""

    kinematics = "holonomic"  # a person can step in any direction, including sideways
    update_hz = WALKER_UPDATE_HZ

    def __init__(self, config: dict):
        super().__init__(config)
        self._st = None

    def attach(self, ctx, entity) -> None:
        state = (ctx.blackboard.get(STATE_KEY) or {}).get(entity.name)
        if state is None:
            raise OutputUnavailable(
                f"entity {entity.name!r} has no walker animation state, so there is no skeleton to "
                f"drive. This output belongs to a `walker` entry, which builds the humanoid's mocap "
                f"bodies and resolves its motion clips."
            )
        self._st = state

    def emit(self, ctx, pref_vel: np.ndarray, yaw: float, dt: float) -> None:
        st = self._st
        # The blendspace faces where the walker *wants* to go rather than where it is being pushed,
        # so a sidestep reads as a strafe instead of the body whipping round. `animate` reads that
        # intent from the state, so it has to be the velocity we were handed -- after avoidance, but
        # it is still what this walker is trying to do this tick.
        st.pref_vel = np.asarray(pref_vel, dtype=float)
        animate(ctx.data, st, st.pos + st.pref_vel * dt, dt)

    def pose(self, ctx) -> tuple[float, float, float]:
        return float(self._st.pos[0]), float(self._st.pos[1]), float(self._st.yaw)

    def stop(self, ctx) -> None:
        """Stand still: re-pose the body where it is, without advancing any clock.

        ``dt = 0`` rather than a frame's worth, and the difference is visible. Stopping happens on
        reset -- right after the owning plugin has posed the skeleton at its start -- so animating a
        frame here would advance the idle clip and settle the body a millimetre off the pose every
        previous episode began from. It also happens when traffic blocks the walker, where holding
        the current pose is what standing still means; the blendspace eases into idle on the next
        real tick because the speed it is driven by has fallen to zero.
        """
        st = self._st
        st.pref_vel = np.zeros(2)
        animate(ctx.data, st, st.pos.copy(), 0.0)
