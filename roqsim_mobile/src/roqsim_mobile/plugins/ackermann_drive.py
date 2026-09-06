# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Controller plugin: Ackermann (car-like) steering + wheel-encoder odometry.

The third base geometry, beside ``diff_drive`` and ``omni_drive``, and the one whose *constraint* is
the point. A differential base turns in place; a car cannot, and a planner written for one is not a
planner for the other. An experiment that reconstructs a car-like platform -- an F1TENTH, a delivery
pod, a scaled road vehicle -- is not approximated by a differential base with a small angular limit:
the reachable set is different, reversing out of a dead end is a manoeuvre rather than a rotation,
and the failure it is trying to provoke is the one this plugin has and the others do not.

**A zero-speed turn command does nothing, and that is the feature.** A ``cmd_vel`` of
``(v=0, w=1.0)`` steers the wheels nowhere and moves the car nowhere, because curvature is
``w / v`` and a stationary car has none. A planner that emits it is a planner that would not move
this robot in reality, and quietly rotating the base -- which is what reusing ``diff_drive`` here
would do -- hides exactly the failure the experiment exists to find. The steering is left where it
is rather than centred, since a real rack holds its angle when the car stops.

**Both front wheels are steered, and not by the same angle.** On a curve the inner wheel follows a
tighter radius than the outer one, so a single shared angle scrubs both tyres; the plugin computes
the pair from the geometry (``atan(L / (R -/+ steer_track/2))``), which is what the linkage the
mechanism is named after does mechanically. The driven wheels are split the same way, across the
driven axle's own ``track``: the outer wheel of a turn travels further, so a common speed on a rigid
axle would scrub. Both splits vanish as the curve straightens, so a straight-line run is unaffected
by either.

**The two splits are measured across different widths**, which is why there are two keys. The steer
split pivots about the steering axes, so it is set by their separation -- the kingpin separation,
inboard of the wheels on most vehicles; the drive split is measured across the driven axle. A model
that gives only ``track`` gets that width for both, which is exact for a design whose steering axes
sit at its wheel centres and an overstated steer split otherwise.

Config::

    ackermann_drive:
      wheel_radius: 0.05
      wheelbase: 0.32               # front axle to rear axle -- what turns a curvature into an angle
      track: 0.24                   # driven axle width, for the drive split (and the default below)
      steer_track: 0.24             # steering-axis (kingpin) separation, for the steer split
      max_linear_vel: 2.0
      max_steer_angle: 0.5          # rad; the rack's mechanical limit, and the turning circle with it
      steer_rate: 4.0               # rad/s slew on the steering angle (0 = instant)
      accel_limit: 2.0              # m/s^2 on the commanded speed (0 = instant)
      steer_actuators: [left_steer_motor, right_steer_motor]    # POSITION servos, left then right
      steer_joints:    [left_steer_joint, right_steer_joint]
      drive_actuators: [rear_left_motor, rear_right_motor]      # VELOCITY servos, left then right
      drive_joints:    [rear_left_joint, rear_right_joint]
      base_body: base_link
      odom_child_frame: base_link   # link the odometry TF points at (see below)
      test_cmd: [1.0, 0.4]          # optional [v, w] applied every tick (standalone demo)

``odom_child_frame`` names the link the ``odom ->`` transform points at, and it must be the ROOT of
whatever URDF ``robot_state_publisher`` is running beside the simulator: a description rooted at
``base_footprint`` already gives ``base_link`` a parent, and a second parent from here leaves that
frame with two, which tf2 cannot resolve.

Endpoints are the ones every base here publishes, so a stack does not know which geometry it is
driving until it tries to turn in place: ``cmd_vel`` in (``geometry_msgs/Twist``), ``odom`` out with
TF, and ``joint_states`` out carrying the steer joints as well as the driven ones -- a car's
steering angle is state a stack watches, and leaving it out is how a URDF's front wheels stay
straight in RViz while the robot corners.

**Odometry is what the encoders say**, as in ``diff_drive``: the driven wheels' measured speed for
``v``, and the *measured* steering angle for the yaw rate through the same bicycle relation
(``w = v * tan(delta) / L``). Reading back the commanded angle instead would report a car that
corners perfectly while the rack is still slewing.

It is therefore dead reckoning and it drifts, which is deliberate. On a straight run the test
vehicle's odometry lands within a few percent; cornering, the tyres slip and the bicycle relation
under-reports the turn -- measured, a car that came round 1.1 rad believes it came round 0.8. No
scrub factor is offered to hide it: unlike a skid-steer's, whose scrub is systematic enough for
``diff_drive``'s ``slip_factor`` to correct, a tyre's slip angle varies with speed and load, so a
single constant would be a fudge that makes the odometry look better than the sensor it stands for.
:mod:`roqsim_sensors.plugins.ground_truth_pose` is what a grader compares against, and the gap
between the two is what a localisation experiment is about.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.context import Endpoint, RobotHandle, SimContext
from roqsim.plugin import Plugin

#: Below this speed a curvature command has no meaning (see the module docstring).
_MIN_SPEED = 1e-3


class AckermannDrivePlugin(Plugin):
    """See the module docstring."""

    #: Drives an entity's actuators, so it belongs inside that entity's ``components:`` block.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self.r = float(self.config.get("wheel_radius", 0.05))
        self.wheelbase = float(self.config.get("wheelbase", 0.32))
        self.track = float(self.config.get("track", 0.24))
        self.steer_track = float(self.config.get("steer_track", self.track))
        self.max_v = float(self.config.get("max_linear_vel", 2.0))
        self.max_steer = float(self.config.get("max_steer_angle", 0.5))
        self.steer_rate = float(self.config.get("steer_rate", 4.0))
        self.accel_limit = float(self.config.get("accel_limit", 2.0))
        self.steer_actuator_names = list(self.config.get("steer_actuators") or [])
        self.steer_joint_names = list(self.config.get("steer_joints") or [])
        self.drive_actuator_names = list(self.config.get("drive_actuators") or [])
        self.drive_joint_names = list(self.config.get("drive_joints") or [])
        self.base_body = self.config.get("base_body", "base_link")
        self.odom_child_frame = self.config.get("odom_child_frame", "base_link")

        self._target_v = 0.0
        self._target_w = 0.0
        self._cmd_v = 0.0  # ramped speed
        self._steer = 0.0  # slewed centre (bicycle) steering angle
        #: Centre angle commanded directly (Ackermann), or None when the last command was a twist.
        #: Which of the two arrived last decides where the angle comes from; they are not merged,
        #: because a twist's curvature and a stated angle are two ways of saying the same thing and
        #: averaging them would obey neither.
        self._steer_cmd: float | None = None
        self._odom = [0.0, 0.0, 0.0, 0.0, 0.0]  # x, y, yaw, v, w
        self._steer_aid: list[int] = []
        self._steer_jid: list[int] = []
        self._drive_aid: list[int] = []
        self._drive_jid: list[int] = []
        self._roll_sign: list[float] = []
        self._jnames: list[str] = []
        self._jpos = np.zeros(0)
        self._jvel = np.zeros(0)

    # -- validation ---------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        for key in (
            "wheel_radius",
            "wheelbase",
            "track",
            "steer_track",
            "max_linear_vel",
            "max_steer_angle",
        ):
            if key in config and float(config[key]) <= 0:
                errors.append(f"'{key}' must be > 0")
        for key in ("steer_rate", "accel_limit"):
            if key in config and float(config[key]) < 0:
                errors.append(f"'{key}' must be >= 0 (0 means no limit)")
        if float(config.get("max_steer_angle", 0.5)) >= np.pi / 2:
            # tan(delta) is the curvature: at 90 degrees it is infinite, and beyond it changes sign,
            # so the car would steer the other way. A rack that reaches it is not a rack.
            errors.append("'max_steer_angle' must be < pi/2 -- the curvature is tan(angle)")
        for pair in (("steer_actuators", "steer_joints"), ("drive_actuators", "drive_joints")):
            acts, joints = config.get(pair[0]), config.get(pair[1])
            for key in pair:
                if config.get(key) is not None and len(config[key]) != 2:
                    errors.append(f"'{key}' must name exactly two, left then right")
            if acts is not None and joints is not None and len(acts) != len(joints):
                errors.append(f"'{pair[0]}' and '{pair[1]}' must have the same length")
        for key in ("steer_actuators", "steer_joints", "drive_actuators", "drive_joints"):
            if not config.get(key):
                errors.append(f"'{key}' is required: name the model's two, left then right")
        if "test_cmd" in config and len(config["test_cmd"]) != 2:
            errors.append("'test_cmd' must be [v, w]")
        return errors

    # -- lifecycle ----------------------------------------------------------------------------

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        m = ctx.model

        def resolve(kind, names):
            ids = [mujoco.mj_name2id(m, kind, prefix + n) for n in names]
            missing = [n for n, i in zip(names, ids, strict=True) if i < 0]
            if missing:
                raise RuntimeError(
                    f"ackermann_drive: could not resolve {missing} for robot {self.robot!r}"
                )
            return ids

        self._steer_aid = resolve(mujoco.mjtObj.mjOBJ_ACTUATOR, self.steer_actuator_names)
        self._steer_jid = resolve(mujoco.mjtObj.mjOBJ_JOINT, self.steer_joint_names)
        self._drive_aid = resolve(mujoco.mjtObj.mjOBJ_ACTUATOR, self.drive_actuator_names)
        self._drive_jid = resolve(mujoco.mjtObj.mjOBJ_JOINT, self.drive_joint_names)

        # Per-wheel roll sign, read off the model exactly as diff_drive does: a wheel carries the
        # base forward when it spins about the base's +y, and a source URDF may express the same
        # wheel about either y direction with both being correct.
        base_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + self.base_body)
        if base_b < 0:
            raise RuntimeError(
                f"ackermann_drive: base body {prefix + self.base_body!r} not found; the wheel roll "
                f"signs are read off the wheel axes expressed in it"
            )
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        rb = d0.xmat[base_b].reshape(3, 3)
        self._roll_sign = [
            1.0
            if float((rb.T @ (d0.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j]))[1]) > 0
            else -1.0
            for j in self._drive_jid
        ]

        self._jnames = list(self.steer_joint_names) + list(self.drive_joint_names)
        self._jpos = np.zeros(len(self._jnames))
        self._jvel = np.zeros(len(self._jnames))

        ctx.blackboard.set(
            f"robot:{self.robot}",
            RobotHandle(
                name=self.robot,
                drive=self.drive,
                read_odom=self.read_odom,
                kinematics="ackermann",
            ),
        )
        ctx.interface.add(
            Endpoint(
                name="cmd_vel",
                direction="in",
                owner=self.robot,
                namespace=ns,
                write=lambda twist: self.drive(twist[0], twist[1], twist[2]),
                backend={
                    "ros2": {
                        "type": "geometry_msgs.msg.Twist",
                        "topic": self.topic_override("cmd_vel") or "cmd_vel",
                    }
                },
            )
        )
        ctx.interface.add(
            Endpoint(
                name="ackermann_cmd",
                direction="in",
                owner=self.robot,
                namespace=ns,
                write=lambda cmd: self.steer(cmd[0], cmd[1]),
                backend={
                    "ros2": {
                        # The message's own steering_angle is documented as "the yaw of a virtual
                        # wheel located at the center of the front axle", which is exactly the angle
                        # this plugin splits into two. The representations line up field for field,
                        # so nothing is converted on the way in.
                        "type": "ackermann_msgs.msg.AckermannDriveStamped",
                        # `drive` rather than the endpoint's own name: this interface exists to be
                        # spoken to by stacks that already emit AckermannDriveStamped, and they emit
                        # it there. A world that wants another topic says so with a topic override.
                        "topic": self.topic_override("ackermann_cmd") or "drive",
                    }
                },
            )
        )
        ctx.interface.add(
            Endpoint(
                name="odom",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self.read_odom,
                rate_hz=50.0,
                backend={
                    "ros2": {
                        "type": "nav_msgs.msg.Odometry",
                        "topic": self.topic_override("odom") or "odom",
                        "frame_id": "odom",
                        "child_frame_id": self.odom_child_frame,
                        "emit_tf": True,
                    }
                },
            )
        )
        ctx.interface.add(
            Endpoint(
                name="joint_states",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=self.read_joint_states,
                rate_hz=50.0,
                backend={
                    "ros2": {
                        "type": "sensor_msgs.msg.JointState",
                        "topic": self.topic_override("joint_states") or "joint_states",
                    }
                },
            )
        )

    # -- commands -----------------------------------------------------------------------------

    def drive(self, vx: float, vy: float, w: float) -> None:
        """Body-frame twist target (``vy`` dropped: a car cannot strafe either)."""
        self._target_v = float(np.clip(vx, -self.max_v, self.max_v))
        self._target_w = float(w)
        self._steer_cmd = None  # a twist states a curvature; the angle is derived from it again

    def steer(self, delta: float, speed: float) -> None:
        """Ackermann target: the centre (bicycle) steering angle, and a speed.

        The direct form of what :meth:`drive` has to infer, and it can say one thing a twist cannot.
        A twist states a *curvature*, which is ``w / v`` -- undefined at rest, so below ``_MIN_SPEED``
        the rack simply holds whatever angle it has. An Ackermann command states the angle itself, so
        **a stationary car can turn its wheels**, which is what a real one does while parking and what
        a car-like stack sends when it is lining up before moving off.

        ``delta`` is the angle of a virtual wheel at the centre of the front axle -- the same quantity
        :meth:`steer_angles` splits into the two real ones, and the same one ``ackermann_msgs``
        defines its ``steering_angle`` to be.
        """
        self._steer_cmd = float(np.clip(delta, -self.max_steer, self.max_steer))
        self._target_v = float(np.clip(speed, -self.max_v, self.max_v))
        self._target_w = 0.0

    def steer_angles(self, delta: float) -> tuple[float, float]:
        """(left, right) wheel angles for a centre (bicycle) angle -- the geometry the linkage does.

        On a curve of radius ``R = L / tan(delta)`` the two front wheels ride circles that differ by
        the separation of their steering axes, so their angles are ``atan(L / (R -/+ steer_track/2))``:
        the inner one turns MORE. A single shared angle would scrub both tyres, and the difference is
        what the mechanism this plugin is named after exists to produce.
        """
        if abs(delta) < 1e-6:
            return 0.0, 0.0
        radius = self.wheelbase / np.tan(delta)
        # A left turn (delta > 0) has positive radius, and the left wheel is the inner one.
        left = np.arctan2(self.wheelbase, radius - self.steer_track / 2.0)
        right = np.arctan2(self.wheelbase, radius + self.steer_track / 2.0)
        # arctan2 returns the angle to the centre; for a right turn both come back near pi, so bring
        # them into (-pi/2, pi/2) where a steering angle lives.
        return (
            float(left - np.pi if left > np.pi / 2 else left),
            float(right - np.pi if right > np.pi / 2 else right),
        )

    def pre_step(self, ctx: SimContext) -> None:
        if ctx.manual_control:
            return  # the viewer's sliders own the actuators this run
        if "test_cmd" in self.config:
            v, w = self.config["test_cmd"]
            self.drive(float(v), 0.0, float(w))

        # Speed first: the steering angle a twist implies depends on the speed it is asking for.
        if self.accel_limit > 0:
            dv = self.accel_limit * ctx.dt
            self._cmd_v += float(np.clip(self._target_v - self._cmd_v, -dv, dv))
        else:
            self._cmd_v = self._target_v

        # Where the target angle comes from, and the one place the two command forms differ.
        #
        # An Ackermann command states the angle, so it applies at any speed including zero -- the
        # rack turns while the car stands still. A twist states a curvature, `w / v`, which says
        # nothing at all at rest: there the command is not approximated and the rack holds its angle
        # rather than centring itself, because a real one does.
        target_steer = None
        if self._steer_cmd is not None:
            target_steer = self._steer_cmd
        elif abs(self._target_v) >= _MIN_SPEED:
            curvature = self._target_w / self._target_v
            target_steer = float(
                np.clip(np.arctan(self.wheelbase * curvature), -self.max_steer, self.max_steer)
            )
        if target_steer is not None:
            if self.steer_rate > 0:
                step = self.steer_rate * ctx.dt
                self._steer += float(np.clip(target_steer - self._steer, -step, step))
            else:
                self._steer = target_steer

        left, right = self.steer_angles(self._steer)
        ctx.data.ctrl[self._steer_aid[0]] = left
        ctx.data.ctrl[self._steer_aid[1]] = right

        # The driven wheels are split by the same geometry: the outer wheel of a turn travels
        # further, so a common speed on a rigid axle scrubs it. Both splits vanish going straight.
        curvature = np.tan(self._steer) / self.wheelbase
        v_left = self._cmd_v * (1.0 - curvature * self.track / 2.0)
        v_right = self._cmd_v * (1.0 + curvature * self.track / 2.0)
        for aid, sign, v in zip(self._drive_aid, self._roll_sign, (v_left, v_right), strict=True):
            ctx.data.ctrl[aid] = sign * v / self.r

    # -- odometry -----------------------------------------------------------------------------

    def post_step(self, ctx: SimContext) -> None:
        m, d = ctx.model, ctx.data
        speeds = [
            sign * d.qvel[m.jnt_dofadr[j]] * self.r
            for j, sign in zip(self._drive_jid, self._roll_sign, strict=True)
        ]
        v = float(np.mean(speeds))
        # The MEASURED steering angle, not the commanded one: reading back the command would report
        # a car cornering perfectly while its rack is still slewing.
        measured = [float(d.qpos[m.jnt_qposadr[j]]) for j in self._steer_jid]
        delta = float(np.mean(measured))
        w = v * np.tan(delta) / self.wheelbase

        o = self._odom
        o[0] += v * np.cos(o[2]) * ctx.dt
        o[1] += v * np.sin(o[2]) * ctx.dt
        o[2] = (o[2] + w * ctx.dt + np.pi) % (2 * np.pi) - np.pi
        o[3], o[4] = v, w

        for k, jid in enumerate(self._steer_jid + self._drive_jid):
            self._jpos[k] = d.qpos[m.jnt_qposadr[jid]]
            self._jvel[k] = d.qvel[m.jnt_dofadr[jid]]

    def read_odom(self):
        x, y, yaw, v, w = self._odom
        return (x, y, yaw, v, 0.0, w)

    def read_joint_states(self):
        return (self._jnames, self._jpos, self._jvel)

    def on_reset(self, ctx: SimContext) -> None:
        self._target_v = self._target_w = 0.0
        self._cmd_v = 0.0
        self._steer = 0.0
        self._odom = [0.0, 0.0, 0.0, 0.0, 0.0]
