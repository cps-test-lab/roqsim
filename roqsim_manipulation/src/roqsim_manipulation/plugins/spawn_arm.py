"""Scene plugin: attach a manipulator MJCF into the world -- to the ground, or onto a mobile base.

The mobile analogue is :mod:`roqsim_mobile.plugins.spawn_robot`; this plugin only *places* an arm
(``build`` attaches the model at a frame; ``configure`` records the entity and applies the home
pose). The kinematics/joint-state publishing live in
:mod:`roqsim_manipulation.plugins.arm_controller`, which finds this arm via the entity registry and
its ``prefix``.

Config::

    spawn_arm:
      model: ur10e          # bundled model name, filename, or absolute path
      namespace: ur10e      # optional transport scope; the arm's endpoints inherit it (default: "")
      prefix: "ur10e_"      # MJCF name prefix (use distinct prefixes for >1 arm)
      base_body: base       # arm root body (ur10e -> 'base', panda -> 'link0')
      pos: [-0.41, 0.0, 0.76]
      rpy: [0.0, 0.0, 3.14159]   # mount orientation as roll/pitch/yaw (rad)
      home: [...]           # joint home pose (defaults per model); applied on reset
      pedestal: false       # add a static support box under the base (floor -> mount height); only
                            # has an effect when pos[2] > 0. Leave it off when the arm mounts on a
                            # table/desk that is already there (the usual case).
      pedestal_half_width: 0.1   # pedestal: half-width (m) of that box's square footprint
      rail:                 # OPTIONAL: carry the arm on a linear axis (gantry / ceiling track)
        axis: [1, 0, 0]     #   travel direction, in the MOUNT frame (after `rpy`)
        range: [-1.5, 1.5]  #   travel limits (m) about `pos`
        home: 0.0           #   carriage position at spawn/reset (m)
        joint: rail_joint   #   MJCF joint name, under the arm's prefix (default: 'rail_joint')
        kp: 20000           #   position-servo gain of the carriage drive
        damping: 200        #   carriage joint damping
      mount:                # OPTIONAL: weld the arm onto another entity's body instead of the world
        robot: robot        #   entity name of a spawn_robot in this world
        body: base_link     #   that robot's body to weld to (default: base_link)
      end_effector:         # OPTIONAL: weld a gripper onto the arm's tool flange
        model: robotiq_2f85 #   a gripper model (robotiq_2f85, schunk_pg70)
        site: attachment_site  # arm site to weld it to (default: attachment_site)
        prefix: ""          #   MJCF name prefix for the gripper's own names
        pos: [0, 0, 0.011]  #   offset in the SITE's frame (e.g. the UR->Robotiq adapter's 11 mm)
        rpy: [0, 0, 0]      #   extra rotation in the site's frame (rad)
        replaces: [ee_plate]  # bodies of the ARM model this tool supersedes, deleted before the
                            #   attach (the ur10e ships a conveyor pushing plate 60 mm past its
                            #   flange, which a gripper would be welded straight into)

``home``/``base_body`` fall back to per-model defaults so a bare ``{model: ur10e}`` works.

**Riding a linear axis (``rail:``)**
A gantry, a ceiling track or a seventh-axis floor rail is a *prismatic joint carrying the arm base*,
which is the one thing ``mount:`` cannot express: ``mount`` welds the arm to a body that already
exists, while a rail has to introduce the moving body itself. With ``rail:`` the plugin inserts a
carriage between the mount frame and the arm, gives it a slide joint along ``axis`` and a position
servo, and attaches the arm to the carriage.

The point of it is **kinematic redundancy**: a 6-DOF arm on a rail is a 7-DOF system, so a task pose
has a one-parameter family of solutions and a planner can trade base travel against arm posture. That
is the class of system this option exists for, and the reason it belongs here rather than in a
per-cell MJCF.

Two ordering facts that other code depends on, so they are guaranteed rather than incidental:

* **The rail joint is declared before the arm's joints**, and its actuator before the arm's. Both
  ``prefixed_joints`` and ``prefixed_actuators`` return model order, so ``arm_controller`` publishes
  and commands ``[rail, <arm joints...>]`` -- matching a URDF that puts the prismatic joint at the
  root of the chain, which is how MoveIt will see the same robot.
* **``home`` stays the ARM's joint vector**; the carriage's initial position is ``rail.home``. Folding
  the rail into ``home`` would silently invalidate every per-model default (a 6-value ur10e ``home``
  would land on ``[rail, j1..j5]`` and leave ``wrist_3`` unset).

The carriage is a real body with mass, so it needs geometry; the plugin draws a small box for it and a
thin beam spanning the travel. Both are **visual only** (``contype``/``conaffinity`` = 0). A ceiling
track that collides would trap the arm against its own support from the first step, and the collision
model a motion planner actually reasons about comes from the URDF/planning scene, not from these two
geoms -- so making them solid would add a contact the planner cannot see. Model the real structure as
scene geometry if the cell needs it.

**Mounting on a mobile base (``mount:``)**
This is what turns a base + an arm into a **mobile manipulator** without a per-combination MJCF: the
arm is attached to the base's body rather than to ``worldbody``, so it rides the base's free joint.
Both models stay untouched and any base pairs with any arm from the world YAML.

Two things it requires, both enforced in ``validate_config``/``build`` rather than left to fail
obscurely later:

* **The base must be declared before the arm** in the world's plugin list. ``build`` runs in
  declaration order, so the body the arm welds to has to exist already.
* **The arm needs a non-empty ``prefix``.** ``arm_controller`` and ``_apply_home`` select an arm's
  joints by prefix scan, and with an empty prefix that scan also claims the base's wheel joints --
  writing arm position targets into wheel actuators that another plugin owns. (An arm that names its
  ``joints:`` explicitly is safe either way, but the prefix is the cheap general guard.)

``name:`` is the entry's reserved SIBLING, not one of the keys above: it labels the entry and names
the entity this spawn registers (default: the plugin ref, i.e. ``spawn_arm``). Written *inside* the
config block it is silently inert -- the arm is then called ``spawn_arm``, and anything addressing
it by the name you chose (``arm_controller``'s ``arm:``, a sensor's ``robot:``) resolves to nothing.

**End effectors (``end_effector:``)**
A gripper is attached **into the arm's own spec before the arm is attached to the world**, so the
arm's prefix covers it. That is deliberate: ``arm_controller`` discovers a gripper as an actuator with
a non-joint (tendon) transmission sharing the arm's prefix, which is exactly how the pre-assembled
``gen3`` works -- so a bare arm plus a gripper reaches the same state as a factory-assembled one, and
the controller needed no change to gain interchangeable hands. The gripper's own
``<model>.manifest.yaml`` supplies the gripper half of ``arm_controller``'s config
(``gripper_joint``/``gripper_open``/``gripper_close``), merged by :meth:`SpawnArmPlugin.expand`.

The attach uses MuJoCo's site attachment, so the *site's* orientation defines the tool frame and
``pos``/``rpy`` are offsets within it -- matching how a real tool adapter is specified.
"""

from __future__ import annotations

import mujoco

from roqsim.config import parse_plugin_entry
from roqsim.context import Entity, SimContext
from roqsim.manifest import expand_manifest, load_manifest
from roqsim.models import ModelError, apply_assets, resolve_model
from roqsim.plugin import Plugin

from ._arm import prefixed_joints, rpy_to_quat

# Per-model defaults so a world only needs `{model: ...}`. Keyed by model file stem.
_DEFAULT_BASE_BODY = {
    "ur10e": "base",
    "ur5e": "base",
    "panda": "link0",
    "gen3": "base_link",
    "open_manipulator_x": "link1",
}
_DEFAULT_HOME = {
    "ur10e": [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
    # UR5e home from the reference implementation this model was ported from: elbow-up
    # over the workspace, which is the configuration its peg-in-hole runs start from.
    "ur5e": [0.0, -1.57079632, 1.57079632, -1.57079632, -1.57079632, 0.0],
    "panda": [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04, 0.04],
    # Gen3 arm home (7 joints); the gripper joints keep their rest (open) pose.
    "gen3": [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633],
    # OpenMANIPULATOR-X: the vendor's own `home` group state from open_manipulator_moveit_config's
    # SRDF (open_manipulator_x.srdf) -- folded back over the base, which is where ROBOTIS parks it.
    # An experiment that needs a different start pose sets `home:` in its world YAML.
    "open_manipulator_x": [0.0, -1.0, 0.7, 0.3],
}


class SpawnArmPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True

    @classmethod
    def expand(cls, spec, world, base_dir):
        """Inject the arm model's default plugins (its ``<model>.manifest.yaml``, e.g. arm_controller).

        Keeps arm-intrinsic plugins out of the world YAML: they ship with the model and become
        components of this arm. Off with ``default_plugins: false``; an entry the world nests under
        this arm with the same label overrides the injected default.

        An ``end_effector`` contributes ITS manifest the same way -- a second included document,
        merged in **underneath** whatever is already there. That is the general precedence rule
        (nearer wins), and it is what the gripper half of ``arm_controller``'s config needs: the
        world's value, then the gripper manifest's, then the arm manifest's. Matching is by label,
        like everywhere else, so the gripper's keys land on the controller that will actually run
        rather than starting a second one -- two ``arm_controller``s on one entity fight over the
        same actuators and refuse each other's blackboard keys.
        """
        injected = expand_manifest(spec, world, base_dir=base_dir)
        cfg = spec.config
        ee = cfg.get("end_effector") or {}
        if not (ee.get("model") and cfg.get("default_plugins", True)):
            return injected

        entity = spec.label
        declared = {s.label: s for s in (*world, *injected) if s.entity == entity}
        ee_file = resolve_model(ee["model"], base_dir=base_dir).path
        for entry in load_manifest(ee_file, base_dir=base_dir):
            base = parse_plugin_entry(entry, "end-effector manifest plugin")
            # The gripper's MJCF names carry the end-effector prefix once attached, but its manifest
            # cannot know that prefix -- it ships with the model, not with this world. Qualify the two
            # keys that ARE MJCF names so a prefixed gripper still resolves. (arm_controller adds the
            # ARM's prefix on top of whatever it is given.)
            ee_prefix = ee.get("prefix", "")
            if ee_prefix:
                for key in ("gripper_joint", "gripper_actuator"):
                    if key in base.config:
                        base.config[key] = f"{ee_prefix}{base.config[key]}"
            target = declared.get(base.label)
            if target is None:
                base.entity = entity
                base.config.setdefault("prefix", cfg.get("prefix", ""))
                injected.append(base)
                declared[base.label] = base
                continue
            for key, value in base.config.items():
                target.config.setdefault(key, value)
        return injected

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.arm_name = self.address
        self.prefix = self.config.get("prefix", "")
        stem = str(self.config.get("model", "")).rsplit("/", 1)[-1].removesuffix(".xml")
        self.base_body = self.config.get("base_body", _DEFAULT_BASE_BODY.get(stem, "base"))
        pos = self.config.get("pos", [0.0, 0.0, 0.0])
        self.pos = [float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)]
        rpy = self.config.get("rpy", [0.0, 0.0, 0.0])
        self.quat = rpy_to_quat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        self.home = list(self.config.get("home", _DEFAULT_HOME.get(stem, [])))
        # Linear axis carrying the arm base (gantry / ceiling track / seventh axis).
        rail = self.config.get("rail") or {}
        self.rail = rail or None
        if self.rail:
            axis = rail.get("axis", [1.0, 0.0, 0.0])
            self.rail_axis = [float(axis[0]), float(axis[1]), float(axis[2])]
            rng = rail.get("range", [-1.0, 1.0])
            self.rail_range = [float(rng[0]), float(rng[1])]
            self.rail_home = float(rail.get("home", 0.0))
            self.rail_joint = rail.get("joint", "rail_joint")
            self.rail_kp = float(rail.get("kp", 20000.0))
            self.rail_damping = float(rail.get("damping", 200.0))
        # Mobile-manipulator mount: weld onto a spawned robot's body instead of worldbody.
        mount = self.config.get("mount") or {}
        self.mount_robot = mount.get("robot")
        self.mount_body = mount.get("body", "base_link")
        # End effector welded at the arm's tool site.
        ee = self.config.get("end_effector") or {}
        self.ee_model = ee.get("model")
        self.ee_site = ee.get("site", "attachment_site")
        self.ee_prefix = ee.get("prefix", "")
        ee_pos = ee.get("pos", [0.0, 0.0, 0.0])
        self.ee_pos = [float(v) for v in (*ee_pos, 0.0, 0.0)][:3]
        ee_rpy = ee.get("rpy", [0.0, 0.0, 0.0])
        self.ee_quat = rpy_to_quat(float(ee_rpy[0]), float(ee_rpy[1]), float(ee_rpy[2]))
        self.ee_replaces = list(ee.get("replaces", []))

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if not config.get("model"):
            errors.append("'model' is required")
        else:
            try:
                resolve_model(config["model"], base_dir=self.base_dir)
            except ModelError as exc:
                errors.append(str(exc))
        if "rpy" in config and len(config["rpy"]) != 3:
            errors.append("'rpy' must be [roll, pitch, yaw] in radians")
        if len(config.get("pos", [0, 0, 0])) not in (2, 3):
            errors.append("'pos' must be [x, y] or [x, y, z]")

        rail = config.get("rail")
        if rail is not None:
            if not isinstance(rail, dict):
                errors.append("'rail' must be a mapping, e.g. {axis: [1,0,0], range: [-1.5, 1.5]}")
            else:
                axis = rail.get("axis", [1.0, 0.0, 0.0])
                if len(axis) != 3 or not any(float(v) for v in axis):
                    errors.append("rail 'axis' must be a non-zero [x, y, z] direction")
                rng = rail.get("range", [-1.0, 1.0])
                if len(rng) != 2 or float(rng[0]) >= float(rng[1]):
                    errors.append("rail 'range' must be [min, max] with min < max, in metres")
                elif not float(rng[0]) <= float(rail.get("home", 0.0)) <= float(rng[1]):
                    # Outside its own limits the carriage starts in violation and the servo fights the
                    # joint limit for the whole run -- a slow, silent corruption of every trial.
                    errors.append("rail 'home' must lie within 'range'")
                if config.get("mount"):
                    # Both want to decide what the arm base is attached to. Expressing a rail on a
                    # mobile base is a real thing to want, but it is a different mechanism (the
                    # carriage would have to ride the base's free joint) and pretending one of the two
                    # silently wins would be worse than refusing.
                    errors.append(
                        "'rail' and 'mount' are mutually exclusive: a rail introduces its own moving "
                        "carriage, while 'mount' welds the arm to a body that already exists"
                    )

        mount = config.get("mount")
        if mount is not None:
            if not isinstance(mount, dict):
                errors.append("'mount' must be a mapping, e.g. {robot: robot, body: base_link}")
            elif not mount.get("robot"):
                errors.append("'mount' requires 'robot' (the entity name of a spawn_robot)")
            elif not config.get("prefix"):
                # An empty prefix makes arm_controller's and _apply_home's prefix scan claim the
                # base's wheel joints as well -- position targets written into torque/velocity
                # actuators another plugin owns. Cheaper to refuse than to debug.
                errors.append(
                    "a mounted arm needs a non-empty 'prefix' so its joints stay distinct from the "
                    "base's (an empty prefix makes the arm's joint scan claim the wheels too)"
                )

        ee = config.get("end_effector")
        if ee is not None:
            if not isinstance(ee, dict):
                errors.append("'end_effector' must be a mapping, e.g. {model: robotiq_2f85}")
            elif not ee.get("model"):
                errors.append("'end_effector' requires 'model'")
            else:
                try:
                    resolve_model(ee["model"], base_dir=self.base_dir)
                except ModelError as exc:
                    errors.append(f"end_effector: {exc}")
                if "rpy" in ee and len(ee["rpy"]) != 3:
                    errors.append("end_effector 'rpy' must be [roll, pitch, yaw] in radians")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        asset = resolve_model(self.config["model"], base_dir=self.base_dir)
        child = mujoco.MjSpec.from_file(str(asset.path))
        # Resolve mesh/texture refs to absolute paths across the model's asset dirs (its own package
        # plus any borrowed via the manifest's `assets:`), so a model from another package -- or one
        # drawing meshes from several packages -- compiles regardless of CWD/attach order.
        apply_assets(child, asset)
        # The gripper goes on BEFORE the arm is attached to the world, so it ends up inside the arm's
        # subtree and under the arm's prefix -- which is what lets arm_controller find its tendon
        # actuator by the same prefix scan that works for the pre-assembled gen3.
        if self.ee_model:
            self._attach_end_effector(child)

        parent = self._mount_parent(spec)
        frame = parent.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        if self.rail:
            # The carriage replaces the mount frame as the arm's parent, so the arm attaches at the
            # carriage origin and `pos`/`rpy` keep meaning "where the axis sits, how it is oriented".
            frame = self._add_carriage(spec, frame)
        spec.attach(child, prefix=self.prefix, frame=frame)

        # A pedestal is a floor-standing support, so it is meaningless for an arm riding a base --
        # and actively wrong: it would be welded to the world under a robot that drives away.
        if self.config.get("pedestal", False) and self.pos[2] > 0.0 and not self.mount_robot:
            g = spec.worldbody.add_geom()
            g.name = f"{self.prefix}pedestal"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.pos = [self.pos[0], self.pos[1], self.pos[2] / 2.0]
            r = float(self.config.get("pedestal_half_width", 0.1))
            g.size = [r, r, self.pos[2] / 2.0]
            g.rgba = [0.25, 0.25, 0.27, 1.0]

    def _add_carriage(self, spec: mujoco.MjSpec, frame):
        """Insert the rail carriage at *frame* and return the frame the arm should attach to.

        Declared before ``spec.attach`` so the slide joint and its actuator precede the arm's in model
        order -- which is what makes ``arm_controller`` report the rail as joint 0, matching a URDF
        with the prismatic joint at the root of the chain.
        """
        carriage = frame.add_body()
        carriage.name = f"{self.prefix}rail_carriage"

        joint = carriage.add_joint()
        joint.name = f"{self.prefix}{self.rail_joint}"
        joint.type = mujoco.mjtJoint.mjJNT_SLIDE
        joint.axis = self.rail_axis
        joint.range = self.rail_range
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.damping = [self.rail_damping, 0.0, 0.0]  # per-axis in MjSpec; a slide uses only [0]

        # A body with a joint needs inertia, so the carriage geom is load-bearing, not decoration.
        # Visual-only (see the module docstring): a solid track would trap the arm against its own
        # support, and the planner's collision model does not come from here anyway.
        box = carriage.add_geom()
        box.name = f"{self.prefix}rail_carriage_geom"
        box.type = mujoco.mjtGeom.mjGEOM_BOX
        box.size = [0.12, 0.12, 0.06]
        box.contype = 0
        box.conaffinity = 0
        box.rgba = [0.30, 0.30, 0.33, 1.0]

        # The track itself, spanning the travel, drawn on the STATIC side so it does not ride along.
        # A geom in the mount frame rather than a body of its own: a second prefixed body hanging off
        # the world would give the arm two roots, and `roqsim export urdf` builds a tree from exactly one.
        span = (self.rail_range[1] - self.rail_range[0]) / 2.0
        mid = (self.rail_range[0] + self.rail_range[1]) / 2.0
        g = frame.add_geom()
        g.name = f"{self.prefix}rail_beam"
        g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        # A capsule's local +z is its length; `fromto` aims it along the travel axis without the
        # quaternion algebra (and without a special case when the axis is already z). With `fromto`
        # set, only size[0] (the radius) is read.
        g.size = [0.04, 0.0, 0.0]
        g.fromto = [
            *[(mid - span) * a for a in self.rail_axis],
            *[(mid + span) * a for a in self.rail_axis],
        ]
        g.contype = 0
        g.conaffinity = 0
        g.rgba = [0.55, 0.55, 0.58, 1.0]

        act = spec.add_actuator()
        act.name = f"{self.prefix}{self.rail_joint}_drive"
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = joint.name
        act.ctrlrange = self.rail_range
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        # Critically damped by default: an underdamped gantry carrying an arm oscillates for seconds
        # after every waypoint, which shows up as execution time the planner did not ask for.
        act.set_to_position(self.rail_kp, dampratio=1.0)

        return carriage.add_frame()

    def _mount_parent(self, spec: mujoco.MjSpec):
        """The body this arm welds to: a spawned robot's base, or ``worldbody``.

        The mount body is looked up by name in the already-built spec, so the base's spawn plugin must
        have run first. That is declaration order, and the error says so -- a bare MuJoCo KeyError
        here reads as a typo in the body name rather than as two plugins in the wrong order.
        """
        if not self.mount_robot:
            return spec.worldbody
        # `mount.body` is the base's MJCF body name. It carries the base's own prefix when it has one,
        # which is why this is the body name and not just "base_link".
        # MjSpec lookups return None for an unknown name rather than raising, so this is an explicit
        # check -- letting the None through surfaces as a TypeError deep inside add_frame().
        parent = spec.body(self.mount_body)
        if parent is None:
            raise RuntimeError(
                f"spawn_arm[{self.arm_name}]: mount body {self.mount_body!r} (robot "
                f"{self.mount_robot!r}) does not exist in the scene yet. Declare the base's "
                f"spawn_robot BEFORE this spawn_arm in the world's plugin list, and give "
                f"`mount.body` the base's prefixed body name if the base uses a prefix."
            )
        return parent

    def _attach_end_effector(self, child: mujoco.MjSpec) -> None:
        """Weld the gripper model into the arm's spec at the arm's tool site."""
        ee_asset = resolve_model(self.ee_model, base_dir=self.base_dir)
        ee = mujoco.MjSpec.from_file(str(ee_asset.path))
        apply_assets(ee, ee_asset)
        # Whatever tool the arm model ships with goes first, or the new one is welded into it. The
        # ur10e carries a `ee_plate` 60 mm past its flange for the conveyor demo; a gripper attached
        # at `attachment_site` (0.1 m) lands inside it, so the two collide from the first step.
        for body_name in self.ee_replaces:
            ee_victim = child.body(body_name)
            if ee_victim is None:
                raise RuntimeError(
                    f"spawn_arm[{self.arm_name}]: end_effector.replaces names body {body_name!r}, "
                    f"which arm model {self.config['model']!r} does not have. Remove it from "
                    f"`replaces` -- silently ignoring it would hide a rename in the arm model."
                )
            child.delete(ee_victim)
        site = child.site(self.ee_site)
        if site is None:
            raise RuntimeError(
                f"spawn_arm[{self.arm_name}]: arm model {self.config['model']!r} has no site "
                f"{self.ee_site!r} to mount an end effector on. Menagerie arms call the tool flange "
                f"'attachment_site'; set `end_effector.site` to the right name for this model."
            )
        # Attaching AT A SITE makes the site's orientation the tool frame, so `pos`/`rpy` below are
        # offsets within it -- the same way a real tool adapter is specified (the UR->Robotiq adapter
        # is 11 mm along the flange normal).
        frame = child.attach(ee, prefix=self.ee_prefix, site=site)
        frame.pos = self.ee_pos
        frame.quat = self.ee_quat

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.arm_name,
                kind="arm",
                body=self.prefix + self.base_body,
                meta={
                    "prefix": self.prefix,
                    "model": self.config["model"],
                    "home": self._home_vector(),
                    # Names the arm's extra DOF for consumers that must treat it differently from a
                    # revolute joint (a URDF/SRDF generator, a metric in metres rather than radians).
                    "rail_joint": (self.prefix + self.rail_joint) if self.rail else None,
                    # Inherited by the arm's endpoint-producing plugins (arm_controller), so the
                    # manifest-injected controller needs no namespace plumbing of its own.
                    "namespace": self.config.get("namespace", ""),
                },
            )
        )
        self._apply_home(ctx)

    def on_reset(self, ctx: SimContext) -> None:
        self._apply_home(ctx)
        mujoco.mj_forward(ctx.model, ctx.data)

    def _home_vector(self) -> list[float]:
        """The home pose in model joint order -- the carriage first when the arm rides a rail.

        `home` is configured as the ARM's joint vector so per-model defaults stay valid (module
        docstring). Everything downstream wants it in model order, and there is more than one such
        consumer: this plugin seeds ``qpos`` with it, and ``arm_controller`` seeds its held targets
        from the copy published on the entity. Those two must agree, or the arm spawns in one pose and
        is immediately servoed to another.
        """
        return ([self.rail_home] + self.home) if (self.rail and self.home) else self.home

    def _apply_home(self, ctx: SimContext) -> None:
        home = self._home_vector()
        if not home:
            return
        jids = prefixed_joints(ctx.model, self.prefix)
        for jid, q in zip(jids, home, strict=False):  # home may cover a joint subset
            ctx.data.qpos[ctx.model.jnt_qposadr[jid]] = float(q)
