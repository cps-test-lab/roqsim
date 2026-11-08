"""Machine: a velocity-driven belt conveyor that runs at a speed and stops.

``build`` attaches the belt MJCF; ``pre_step`` forces the invisible drive slab's joint velocity
(an ideal belt motor) and spins the end rollers to match; ``post_step`` wraps the belt's position
so it runs forever. Objects ride the belt purely through the ``belt_package`` friction pair.

The belt is a **benchtop** unit: its feet run from z=0.76 to 0.915, so it needs a ~0.76 m table
under it, spawned separately (``industrial_table`` is the matching prop; any other ~0.76 m top
also works). Nothing here places or checks that table -- a conveyor with no
table under it simply floats.

A machine (:class:`roqsim.machine.MachinePlugin`) whose procedure is continuous: the program
``run(speed)`` brings the belt to a speed (m/s, its sign the direction) -- at once, or ramped at
``ramp`` (m/s^2) -- and ends there, the belt running on as the machine's state; ``stop()`` brings it
to rest. Each start of a stopped belt is a ``started`` event, each stop a ``stopped`` event, between
the programs' own. Whether the belt stops for a part is a scenario's decision, made from a
``zone_trigger`` at its end (``docs/architecture.rst``, "Machines and scenarios").

Handle: a :class:`~roqsim.machine.MachineHandle` under ``machine:<name>`` -- ``start("run",
speed=0.3)``, ``start("stop")``, ``status()``, ``busy``, ``cancel()``, ``value("speed")``. Its
``start``/``status``/``cancel`` are endpoints too. The belt speed is also a ``speed`` stream
(``std_msgs/Float64`` under ROS) that the generic ``ros2_bridge`` drives: a new value replaces what
the belt was doing (``run``, preempting); a repeated one changes nothing.

Config::

    conveyor:
      namespace: ""         # optional transport scope for the speed endpoint (/<ns>/speed)
      prefix: ""            # MJCF name prefix (distinct prefixes for >1 belt)
      model: conveyor       # bundled model name / path
      pose:                 # the belt model in the world, a geometry_msgs/Pose (roqsim.pose); omitted
        position: {x: 0.0, y: 0.0, z: 0.0}          #   components are 0
        orientation: {roll: 0.0, pitch: 0.0, yaw: 0.0}   #   or a quaternion x/y/z/w
      length: 2.442         # optional full belt length (X, m); default keeps the base model
      width: 0.58           # optional full belt width  (Y, m); default keeps the base model
      speed: 0.1            # belt speed at the start of each episode (m/s); negative reverses
      ramp: 0.5             # acceleration of run/stop (m/s^2); unset: the speed changes at once
      friction: 0.6         # optional belt<->package sliding friction override
      roller_radius: 0.0275
      belt_wrap: 0.025      # +/- position wrap (m); keep <= half the slab overhang
      package_pose:         # the package's free-body reset pose, in the BELT's frame (the belt's
        position: {x: 1.0, y: 0.6, z: 0.996}   #   own pose is applied to it); a geometry_msgs/Pose,
                                               #   omitted components 0; default at the feed end
      object_name: package  # entity name the ridden package registers under
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim import endpoint
from roqsim.context import Entity, SimContext
from roqsim.machine import MachinePlugin, program
from roqsim.models import apply_assets, resolve_model
from roqsim.pose import config_pose, config_pose_errors
from roqsim.types import Speed, Transform


class ConveyorPlugin(MachinePlugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True
    SETPOINTS = {"speed": "m/s"}
    ENTITY_KIND = "conveyor"
    # Base belt geometry (half-extents, m) as authored in conveyor.xml; the resize helper below
    # rescales the loaded spec to the configured length/width relative to these. Height-class
    # constants (half-height, roller radius) are fixed.
    _L0 = 1.221  # base belt half-length (X)
    _W0 = 0.29  # base belt half-width  (Y)
    _HALF_H = 0.0275  # belt half-height (Z), fixed
    _SLAB_OVER = 0.029  # drive-slab overhang past belt end (belt_surface 1.25 - belt_visual 1.221)
    _RAIL_OFF = 0.006  # rail centre offset past belt edge (0.296 - 0.29); == rail half-thickness
    _SURF_Z = 0.9425  # belt-surface height inside the model (rail/roller z), fixed

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.conveyor_name = self.address
        self.prefix = self.config.get("prefix", "")
        self.pos, self.quat = config_pose(self.config)
        # Optional belt size (full length/width, m). Absent -> keep the base model exactly. Bad
        # values are tolerated here (kept as the base half-extent) so validate_config can report
        # them with a friendly message rather than crashing during construction.
        self.length = self.config.get("length")
        self.width = self.config.get("width")
        self._half_len = self._half_extent(self.length, self._L0)
        self._half_wid = self._half_extent(self.width, self._W0)
        self.HOME = {"speed": float(self.config.get("speed", 0.1))}
        ramp = self.config.get("ramp")
        self.ramp_rate = float(ramp) if ramp is not None else None
        self.roller_radius = float(self.config.get("roller_radius", 0.0275))
        self.belt_wrap = float(self.config.get("belt_wrap", 0.025))
        # Default free-body start pose tracks the +x (feed) end so the package starts on the belt
        # for any length (1.0 == base 1.221 - 0.221). Explicit package_pose still wins.
        default_pkg_x = self._half_len - 0.221
        if "package_pose" in self.config:
            pkg_pos, pkg_quat = config_pose({"pose": self.config["package_pose"]})
            self._package_pose = [*pkg_pos, *pkg_quat]
        else:
            self._package_pose = [default_pkg_x, 0.6, 0.996, 1.0, 0.0, 0.0, 0.0]
        self._package_pose_world = self._to_world(self._package_pose)
        # resolved in configure()
        self._belt_dadr = self._belt_qadr = -1
        self._roller_dadr: list[int] = []
        self._pkg_qadr = -1
        self._pkg_bid = -1
        self._pkg_frame = ""
        self._pair_id = -1

    def _to_world(self, pose: list[float]) -> list[float]:
        """Lift a model-local ``[x y z qw qx qy qz]`` pose into world coordinates.

        ``package_pose`` is expressed in the belt model's own frame (its default tracks the belt's
        feed end), but the package hangs off a **free joint**, whose ``qpos`` MuJoCo reads as a WORLD
        pose -- the attach frame that places every welded part is not applied to it. So the belt's
        own ``pose`` has to be composed in here; without it a belt spawned anywhere but the
        world origin resets its package onto the floor at the origin.
        """
        local_pos = np.asarray(pose[:3], dtype=float)
        local_quat = np.asarray(pose[3:], dtype=float)
        quat = np.asarray(self.quat, dtype=float)
        world_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(world_pos, local_pos, quat)
        world_quat = np.zeros(4)
        mujoco.mju_mulQuat(world_quat, quat, local_quat)
        return [*(world_pos + np.asarray(self.pos, dtype=float)), *world_quat]

    @staticmethod
    def _half_extent(value, default: float) -> float:
        """Half of a configured full extent, falling back to ``default`` for absent/invalid input."""
        if value is None:
            return default
        try:
            return float(value) / 2
        except (TypeError, ValueError):
            return default

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if "package_pose" in config:
            errors += config_pose_errors({"pose": config["package_pose"]}, "conveyor package_pose")
        errors += config_pose_errors(config, "conveyor")
        if float(config.get("roller_radius", 0.0275)) <= 0:
            errors.append("'roller_radius' must be > 0")
        for key in ("length", "width"):
            if key not in config:
                continue
            try:
                value = float(config[key])
            except (TypeError, ValueError):
                errors.append(f"'{key}' must be a number > 0")
                continue
            if value <= 0:
                errors.append(f"'{key}' must be > 0")
        if "ramp" in config:
            try:
                if float(config["ramp"]) <= 0:
                    errors.append("'ramp' must be > 0")
            except (TypeError, ValueError):
                errors.append("'ramp' must be a number > 0")
        if "length" in config:
            try:
                half = float(config["length"]) / 2
                if half + self._SLAB_OVER <= float(config.get("belt_wrap", 0.025)):
                    errors.append("'length' too short: drive-slab overhang <= belt_wrap")
            except (TypeError, ValueError):
                pass
        errors += super().validate_config(config)
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        asset = resolve_model(self.config.get("model", "conveyor"), base_dir=self.base_dir)
        child = mujoco.MjSpec.from_file(str(asset.path))
        apply_assets(child, asset)
        if self.length is not None or self.width is not None:
            self._resize(child)
        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    def _resize(self, child: mujoco.MjSpec) -> None:
        """Rescale the base belt geometry to the configured length/width (height fixed).

        Physics-critical parts (belt, drive slab, rollers, rails) are recomputed exactly with
        *additive* offsets, so the drive slab always overhangs the belt end (>= belt_wrap) no matter
        how short the belt is. The decorative feet scale *proportionally* so they track the belt
        footprint and never invert. Called on the freshly loaded child spec, so each conveyor
        instance sizes independently. The table under the belt is a separate prop
        (``industrial_table`` or any ~0.76 m top) and is NOT touched here.
        """
        L, W, hz = self._half_len, self._half_wid, self._HALF_H
        z = self._SURF_Z
        # Belt surfaces: exact.
        child.geom("belt_visual").size = [L, W, hz]
        child.geom("belt_surface").size = [L + self._SLAB_OVER, W, hz]  # slab overhangs belt end
        # Side rails: length spans the belt; offset just outside the belt edge.
        for rail, side in (("rail_near", -1.0), ("rail_far", 1.0)):
            g = child.geom(rail)
            g.size = [L, self._RAIL_OFF, 0.03]
            g.pos = [0.0, side * (W + self._RAIL_OFF), z]
        # End rollers: at each belt end (radius fixed), cylinder half-length spans the width.
        for roller, side in (("roller_a", -1.0), ("roller_b", 1.0)):
            child.body(roller).pos = [side * L, 0.0, z]
            child.geom(roller + "_geom").size = [self.roller_radius, W, 0.0]
        # Decorative feet: proportional scale (never inverts), so they stay under the belt corners.
        fx, fy = L / self._L0, W / self._W0
        for name in ("conv_leg_1", "conv_leg_2", "conv_leg_3", "conv_leg_4"):
            p = list(child.geom(name).pos)
            child.geom(name).pos = [p[0] * fx, p[1] * fy, p[2]]

    def configure(self, ctx: SimContext) -> None:
        m = ctx.model
        p = self.prefix
        self._ctx = ctx

        def jnt(n):
            return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p + n)

        belt_jid = jnt("belt_x")
        if belt_jid < 0:
            raise RuntimeError(f"conveyor: belt joint {p + 'belt_x'!r} not found")
        self._belt_dadr = m.jnt_dofadr[belt_jid]
        self._belt_qadr = m.jnt_qposadr[belt_jid]
        self._roller_dadr = [
            m.jnt_dofadr[jid] for jid in (jnt("roller_a_spin"), jnt("roller_b_spin")) if jid >= 0
        ]
        pkg_jid = jnt("package_free")
        self._pkg_qadr = m.jnt_qposadr[pkg_jid] if pkg_jid >= 0 else -1
        self._pkg_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p + "package")
        self._pkg_frame = p + "package"
        self._pair_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_PAIR, p + "belt_package")

        if "friction" in self.config and self._pair_id >= 0:
            mu = float(self.config["friction"])
            m.pair_friction[self._pair_id, 0] = mu
            m.pair_friction[self._pair_id, 1] = mu

        self.register_entity(ctx, body=p + "belt", model=self.config.get("model", "conveyor"))
        # Register the ridden package as a queryable entity so the sim_interfaces control plane can
        # report its ground-truth world pose (GetEntityState) and teleport it (SetEntityState, via
        # the free joint). This is the ground-truth object pose an off-board planner (e.g. MoveIt
        # Task Constructor) reads to sort the box; ``base_joint`` points at the free joint so a
        # set is applied to the box rather than silently rejected.
        self.has_package = self._pkg_qadr >= 0
        if self.has_package:
            self.object_name = self.config.get("object_name", "package")
            ctx.entities.add(
                Entity(
                    name=self.object_name,
                    kind="object",
                    body=p + "package",
                    meta={"prefix": p, "base_joint": p + "package_free"},
                )
            )
        super().configure(ctx)

    # -- programs ------------------------------------------------------------------------------
    @program
    def run(self, speed: float):
        """Bring the belt to a speed and end there; the belt runs on.

        Args:
            speed: belt surface speed (m/s); negative reverses
        """
        before = self.value("speed")
        speed = float(speed)
        if before == 0.0 and speed != 0.0:
            self.step("started", speed=speed)
        if self.ramp_rate:
            self.ramp("speed", speed, self.ramp_rate)
        else:
            self.set("speed", speed)
        yield from self.until_reached("speed")
        if before != 0.0 and speed == 0.0:
            self.step("stopped")

    @program
    def stop(self):
        """Bring the belt to rest."""
        yield from self.run(0.0)

    # The package's true world pose as a TF transform, so a viewer binds it to the scene body by name
    # (the child is the exported body name). This is ground truth: the package's pose is not a joint,
    # so nothing else publishes it. It belongs to the package, with no namespace, and the topic is
    # the relative `tf`, which resolves to the plain `/tf`.
    @endpoint.out(
        rate=30.0,
        when="has_package",
        owner=lambda self: self.object_name,
        namespace="",
        ros2={"topic": "tf", "frame_id": "world"},
    )
    def package_pose(self) -> Transform:
        """The package's world pose."""
        d = self._ctx.data
        return Transform("", self._pkg_frame, d.xpos[self._pkg_bid], d.xquat[self._pkg_bid])

    # A setpoint, so the latest one is applied once per step. Its namespace (own config, else none)
    # scopes the topic so several conveyors can share a world under one bridge.
    @endpoint.stream(ros2={"type": "std_msgs.msg.Float64"})
    def speed(self, data: Speed) -> None:
        """Belt speed; negative reverses. Replaces what the belt was doing.

        Args:
            data: belt surface speed
        """
        speed = float(data)
        if speed == self._goal and self.status_now.error == "":
            return  # running there, or there
        self.preempt("run", speed=speed)

    @property
    def _goal(self) -> float:
        """The speed the belt is at or ramping to."""
        return self.goal("speed")

    def on_reset(self, ctx: SimContext) -> None:
        # A speed set through the endpoint or the handle belongs to the episode that set it.
        super().on_reset(ctx)
        ctx.data.qpos[self._belt_qadr] = 0.0
        if self._pkg_qadr >= 0:
            ctx.data.qpos[self._pkg_qadr : self._pkg_qadr + 7] = self._package_pose_world
        mujoco.mj_forward(ctx.model, ctx.data)

    def pre_step(self, ctx: SimContext) -> None:
        super().pre_step(ctx)
        # Ideal constant-velocity belt motor: force the joint velocity every step.
        speed = self.value("speed")
        ctx.data.qvel[self._belt_dadr] = speed
        omega = speed / self.roller_radius
        for dadr in self._roller_dadr:
            ctx.data.qvel[dadr] = omega

    def post_step(self, ctx: SimContext) -> None:
        super().post_step(ctx)
        # Wrap the belt position so it can run indefinitely (objects keep their world pose).
        q = ctx.data.qpos
        if q[self._belt_qadr] > self.belt_wrap:
            q[self._belt_qadr] -= 2 * self.belt_wrap
        elif q[self._belt_qadr] < -self.belt_wrap:
            q[self._belt_qadr] += 2 * self.belt_wrap
