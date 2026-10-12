"""Plugin: the parts of ``ur_robot_driver``'s ROS interface a simulated UR arm has no other source for.

A stack written against a real Universal Robots arm talks to the driver's controller manager and
expects, besides the trajectory controller and the joint-state and force-torque broadcasters (which
``arm_controller`` and ``force_torque`` provide under the driver's names), three more of its
controllers:

* ``speed_scaling_state_broadcaster`` -- the robot's speed scaling, published on
  ``speed_scaling_state_broadcaster/speed_scaling`` (``std_msgs/Float64``, percent, as the driver's
  broadcaster publishes the hardware's factor times 100). Here the factor is the speed slider's,
  since a simulated arm has no controller-side slowdown to multiply it by. It starts at 100 % and is
  what the scaled trajectory controller runs at.
* ``io_and_status_controller`` -- the driver's GPIO controller, with two of its services:

  - ``io_and_status_controller/set_speed_slider`` (``ur_msgs/SetSpeedSliderFraction``) sets the
    slider at run time. A fraction in [0.01, 1.0] is applied and answered ``success: true``; anything
    else, NaN included, is refused with ``success: false`` and leaves the slider as it was, the
    range the driver's ``GPIOController`` accepts. A reset returns the slider to 1.0.
  - ``io_and_status_controller/zero_ftsensor`` (``std_srvs/Trigger``), the wrist sensor's zero
    button. Here it presses the tare of the ``force_torque`` entry named by ``ft`` -- the same
    operation as that sensor's own ``tare`` endpoint, under the name the driver gives it.

* ``joint_trajectory_controller`` -- the driver's unscaled trajectory controller, which it loads
  inactive beside the scaled one. ``ur_moveit_config`` lists both, and MoveIt chooses between them by
  their state on the controller manager: a stack whose controller manager does not list this one at
  all gets its trajectories sent to an action nobody serves. Activating it is refused with a message
  naming the scaled controller.

All three register with the controller manager under the driver's names and types, claiming no
command interface; the first two active, the third inactive, as they are on the robot.

The slider reaches the motion through the blackboard key ``speed_scaling:<entity>``, a
``() -> float`` fraction: an ``arm_controller`` on the same entity with ``controller_type:
ur_controllers/ScaledJointTrajectoryController`` advances its trajectories' time at it, whether the
trajectory came in-process or as a ROS ``FollowJointTrajectory`` goal. One with the plain
``JointTrajectoryController`` type is not slowed, as the driver's unscaled controller is not.

The service type comes from ``ur_msgs``, which the ROS side must have installed; this package
registers the bridge's handler for it (``roqsim_manipulation.ros2_services``).

The driver's other GPIO surface (digital outputs, payload, tool voltage, robot and safety mode) has
no simulated counterpart and is not served: a stack that relies on it fails at the missing name
rather than reading an invented value.

Config -- a component of the entry that spawns the arm, AFTER the ``force_torque`` entry it names::

    ur_driver_interface:
      ft: force_torque_sensor_broadcaster   # REQUIRED: label of the force_torque entry zero_ftsensor tares
      rate_hz: 100.0                        # speed_scaling publish rate (the driver's default)
"""

from __future__ import annotations

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.controllers import ACTIVE, INACTIVE, Controller, registry_for
from roqsim.plugin import Plugin

from .arm_controller import speed_scaling_key

#: The speed slider fractions the driver's ``GPIOController`` accepts; it refuses anything else.
SLIDER_RANGE = (0.01, 1.0)


class URDriverInterfacePlugin(Plugin):
    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.ft = self.config.get("ft", "")
        self.rate_hz = float(self.config.get("rate_hz", 100.0))
        self._reader = None
        self._slider = 1.0

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if not config.get("ft"):
            errors.append(
                "'ft' is required: the label of the force_torque entry the driver's "
                "zero_ftsensor tares"
            )
        if float(config.get("rate_hz", 100.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        return errors

    def configure(self, ctx: SimContext) -> None:
        self._reader = ctx.blackboard.get(f"ft:{self.ft}")
        if self._reader is None or self._reader.tare is None:
            raise RuntimeError(
                f"ur_driver_interface: no force_torque sensor labelled {self.ft!r} with a tare. "
                "Declare that force_torque entry under the arm BEFORE this one."
            )
        key = speed_scaling_key(self.entity)
        if ctx.blackboard.get(key) is not None:
            raise RuntimeError(
                f"ur_driver_interface: {key!r} is already registered; an arm has one driver "
                "interface, and so one speed slider"
            )
        ctx.blackboard.set(key, self.speed_scaling_factor)
        entity = ctx.entities.get(self.entity)
        ns = entity.meta.get("namespace", "") if entity else ""
        registry = registry_for(ctx)
        for name, kind in (
            ("speed_scaling_state_broadcaster", "ur_controllers/SpeedScalingStateBroadcaster"),
            ("io_and_status_controller", "ur_controllers/GPIOController"),
        ):
            registry.register(
                Controller(name=name, type=kind, state=ACTIVE, namespace=ns, owner=self.entity)
            )
        registry.register(
            Controller(
                name="joint_trajectory_controller",
                type="joint_trajectory_controller/JointTrajectoryController",
                state=INACTIVE,
                namespace=ns,
                owner=self.entity,
                apply=self._refuse_unscaled,
            )
        )

    def on_reset(self, ctx: SimContext) -> None:
        self._slider = 1.0

    @staticmethod
    def _refuse_unscaled(active: bool) -> None:
        if active:
            raise RuntimeError(
                "joint_trajectory_controller is listed as ur_robot_driver lists it, inactive, and is "
                "not served in simulation: use scaled_joint_trajectory_controller"
            )

    def speed_scaling_factor(self) -> float:
        """The speed scaling in effect, as a fraction: what the scaled trajectory controller runs at."""
        return self._slider

    @endpoint.out(
        rate="rate_hz",
        ros2={
            "type": "std_msgs.msg.Float64",
            "topic": "speed_scaling_state_broadcaster/speed_scaling",
        },
    )
    def speed_scaling(self) -> float:
        """The speed scaling in effect, percent."""
        return 100.0 * self._slider

    @endpoint.command(
        ros2={
            "service": "ur_msgs.srv.SetSpeedSliderFraction",
            "name": "io_and_status_controller/set_speed_slider",
        }
    )
    def set_speed_slider(self, speed_slider_fraction: float) -> bool:
        """Set the speed slider; returns whether the fraction was accepted.

        Args:
            speed_slider_fraction: the slider, a fraction in [0.01, 1.0]; anything else is refused
                and leaves the slider as it was
        """
        fraction = float(speed_slider_fraction)
        low, high = SLIDER_RANGE
        if not low <= fraction <= high:  # NaN compares false, so it is refused too
            return False
        self._slider = fraction
        return True

    @endpoint.command(ros2={"name": "io_and_status_controller/zero_ftsensor"})
    def zero_ftsensor(self) -> None:
        """Zero the wrist sensor at its current pose and load: the named sensor's tare."""
        self._reader.tare()
