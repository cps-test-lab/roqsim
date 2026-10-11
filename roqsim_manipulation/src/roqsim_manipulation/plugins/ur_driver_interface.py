"""Plugin: the parts of ``ur_robot_driver``'s ROS interface a simulated UR arm has no other source for.

A stack written against a real Universal Robots arm talks to the driver's controller manager and
expects, besides the trajectory controller and the joint-state and force-torque broadcasters (which
``arm_controller`` and ``force_torque`` provide under the driver's names), three more of its
controllers:

* ``speed_scaling_state_broadcaster`` -- the pendant's speed slider times the controller's own
  scaling, published on ``speed_scaling_state_broadcaster/speed_scaling`` (``std_msgs/Float64``,
  percent). A simulated arm has no slider and is never slowed by its controller, so it is
  ``speed_scaling`` (default 100) throughout. A stack that waits for, or checks, the scaling reads
  the value it would read on an unslowed robot.
* ``io_and_status_controller`` -- the driver's GPIO controller. Of its services, the one a contact
  task needs is ``io_and_status_controller/zero_ftsensor`` (``std_srvs/Trigger``), the wrist sensor's
  zero button. Here it presses the tare of the ``force_torque`` entry named by ``ft`` -- the same
  operation as that sensor's own ``tare`` endpoint, under the name the driver gives it.
* ``joint_trajectory_controller`` -- the driver's unscaled trajectory controller, which it loads
  inactive beside the scaled one. ``ur_moveit_config`` lists both, and MoveIt chooses between them by
  their state on the controller manager: a stack whose controller manager does not list this one at
  all gets its trajectories sent to an action nobody serves. Activating it is refused with a message
  naming the scaled controller, which a simulated arm runs at 100 % and so is the same controller.

All three register with the controller manager under the driver's names and types, claiming no
command interface; the first two active, the third inactive, as they are on the robot.

The driver's other GPIO surface (digital outputs, payload, tool voltage, robot and safety mode) has
no simulated counterpart and is not served: a stack that relies on it fails at the missing name
rather than reading an invented value.

Config -- a component of the entry that spawns the arm, AFTER the ``force_torque`` entry it names::

    ur_driver_interface:
      ft: force_torque_sensor_broadcaster   # REQUIRED: label of the force_torque entry zero_ftsensor tares
      speed_scaling: 100.0                  # percent, what speed_scaling publishes
      rate_hz: 100.0                        # speed_scaling publish rate (the driver's default)
"""

from __future__ import annotations

from roqsim import endpoint
from roqsim.context import SimContext
from roqsim.controllers import ACTIVE, INACTIVE, Controller, registry_for
from roqsim.plugin import Plugin


class URDriverInterfacePlugin(Plugin):
    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.ft = self.config.get("ft", "")
        self.scaling = float(self.config.get("speed_scaling", 100.0))
        self.rate_hz = float(self.config.get("rate_hz", 100.0))
        self._reader = None

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if not config.get("ft"):
            errors.append(
                "'ft' is required: the label of the force_torque entry the driver's "
                "zero_ftsensor tares"
            )
        if not 0.0 <= float(config.get("speed_scaling", 100.0)) <= 100.0:
            errors.append("'speed_scaling' is a percentage in [0, 100]")
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

    @staticmethod
    def _refuse_unscaled(active: bool) -> None:
        if active:
            raise RuntimeError(
                "joint_trajectory_controller is listed as ur_robot_driver lists it, inactive, and is "
                "not served in simulation: use scaled_joint_trajectory_controller, which a simulated "
                "arm runs at 100 % speed scaling and so is the same controller"
            )

    @endpoint.out(
        rate="rate_hz",
        ros2={
            "type": "std_msgs.msg.Float64",
            "topic": "speed_scaling_state_broadcaster/speed_scaling",
        },
    )
    def speed_scaling(self) -> float:
        """The speed scaling, percent."""
        return self.scaling

    @endpoint.command(ros2={"name": "io_and_status_controller/zero_ftsensor"})
    def zero_ftsensor(self) -> None:
        """Zero the wrist sensor at its current pose and load: the named sensor's tare."""
        self._reader.tare()
