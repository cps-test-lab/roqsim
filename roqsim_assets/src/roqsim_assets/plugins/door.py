"""Scene + controller plugin: a hinged (swing) door with a ROS-controllable opening angle.

A door is the union of the two existing prop patterns: it is a *parametric* prop like the
``shelf`` (a leaf built to the opening's width/height, either a primitive box or a textured mesh
model) **and** an *articulated, live-controllable* prop like the ``conveyor`` (a hinge joint driven
by a position actuator, whose target is set over ROS). It fills the gap the floorplan generator
leaves -- a door opening is otherwise just a 2 m hole with a lintel, no leaf.

Geometry. The door is placed by its **opening centre** (``pos``, like every other prop) and the wall
direction (``rpy`` yaw). ``hinge_side`` picks which vertical edge of the opening is *fixed* (the
hinge); the leaf spans from there across the opening. A ``hinge`` joint about +Z lets it swing;
``swing`` (+1/-1) chooses which side of the wall it opens toward. ``open`` is the initial openness
fraction (0 = closed, 1 = fully open at ``max_angle``) -- i.e. *how open* the door starts.

Actuation. A ``position`` actuator holds the hinge at the target angle every ``pre_step``. This one
mechanism unifies the two door kinds the user asked for: a **passive** door (``controllable: false``)
simply holds its ``open`` angle and resists being pushed (finite ``kp``); an **automatic** door
(``controllable: true``) additionally exposes ROS I/O so its target is commanded at runtime.

ROS interface (declared only when ``controllable``; wired by the generic ``ros2_bridge``, so no ROS is
imported here). Openness is a fraction in ``[0, 1]`` on every port:

* ``in``  ``<ns>/cmd``   -- ``std_msgs/Float64`` set-openness (fire-and-forget; the path an ``.osc``
  ``topic_publish`` drives).
* ``out`` ``<ns>/state`` -- ``std_msgs/Float64`` current openness, ~10 Hz.
* ``in``  action ``<ns>/door`` -- ``control_msgs/action/GripperCommand`` for open/close *with feedback*
  (``goal.command.position`` is the target fraction; the handler reports reached/stalled by watching
  the ``door:<name>`` state reader on the blackboard -- see ``roqsim_ros_bridge.actions``).

A :class:`DoorHandle` is also published on the blackboard under ``door:<name>`` for in-process
drivers (scripts, tests) that bypass any transport.

Config::

    door:
      name: door_1          # entity name (default 'door')
      prefix: door_1_        # MJCF name prefix (distinct per door)
      pos: [x, y, 0]         # opening CENTRE, [x, y] or [x, y, z] world placement
      rpy: [0, 0, yaw]       # orientation; yaw aligns the closed leaf along its wall
      width: 0.9             # opening / leaf width (m)
      height: 2.0            # leaf height (m)
      thickness: 0.04        # leaf thickness (m); box leaf only
      leaf: true             # false -> a cased opening: the casing is welded, no leaf is hung
      model: door            # optional leaf mesh model (door | door_glass | pkg:name); omit -> box
      color: [r, g, b, a]    # repaint the leaf (omit -> the model's own colours); alpha optional
      frame: true            # weld a static jamb/lintel casing around the opening (frame_model)
      frame_model: door_frame  # the static frame model to weld (visual-only)
      frame_color: [r,g,b,a]   # repaint the casing (default: follow `color`)
      mount_offset: 0.06     # hang the leaf this far proud of the wall (m), on the swing side, so it
                             #   clears the reveal and can open past 90 deg
      hinge_side: left       # 'left' | 'right' -- which edge of the opening is fixed (the hinge)
      swing: 1               # +1 / -1 -- which way the leaf opens about +Z
      max_angle: 120         # fully-open hinge angle (deg); a leaf can swing well past 90 in free
                             #   space -- raise/lower per door for what its surroundings allow
      open: 0.0              # initial openness fraction 0..1 ('how open'); held until commanded
      controllable: true     # expose the ROS endpoints/action; false = passive, held at 'open'
      namespace: ""          # transport scope -> /<ns>/cmd , /<ns>/state , /<ns>/door
      kp: 40.0               # position-actuator stiffness
      kv: 8.0                # position-actuator damping (velocity gain)

The leaf-mesh convention (``model:``) mirrors the rest of the asset library but with one addition: a
door model's origin is its **hinge (fixed) vertical edge** at floor level, the leaf extending along
+X and up +Z, so the plugin can hang it on the hinge with no per-model offset. A box leaf (no
``model``) is built to the same convention procedurally.

Frame. Unless ``frame: false``, a static jamb/lintel casing (the ``frame_model``, default
``door_frame``) is welded around the opening while the leaf hinges inside it. The frame is centred on
the opening and rescaled to width/height like the leaf; it is visual-only (the wall already collides),
so it never narrows the doorway. A missing frame model is a warning, not an error (bare opening).

Leafless openings. ``leaf: false`` welds the casing but hangs **no leaf** -- a cased opening (German
*Türblatt*-less door): no hinge joint, no actuator, no ROS surface, nothing to command. Use it where a
wall opening should read as a doorway rather than a hole, while staying a ``door`` in the floorplan so the
room loops it belongs to are unchanged. It is still registered as a ``door`` entity (on the casing body),
so anything enumerating the building's doors still finds it. ``leaf: false`` with ``frame: false`` would
place nothing at all and is refused.

Colour. ``color`` repaints the leaf and (unless ``frame_color`` overrides it) the casing, so a world can
match its doors to the rest of its trim without a recoloured copy of the model. Only the leaf's
**colliding** geometry is repainted: in this library a leaf's decoration is non-colliding by convention
(the chrome handle, ``contype``/``conaffinity`` 0), so it keeps its own finish instead of turning into
the door colour. That also means a glazed leaf should be left alone -- painting ``door_glass``'s pane
opaque would defeat it -- and only its ``frame_color`` set.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import mujoco

from roqsim.context import Endpoint, Entity, SimContext
from roqsim.models import ModelError, apply_assets, resolve_model
from roqsim.plugin import Plugin

logger = logging.getLogger("roqsim_assets.door")


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """(w, x, y, z) quaternion from roll/pitch/yaw (rad), fixed-axis XYZ (ROS/URDF convention)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


# Neutral tan-face colour for a box leaf when no mesh model is given (matches the wooden-door tone).
_LEAF_RGBA = [0.86, 0.82, 0.74, 1.0]

# Authored size (m) of a leaf mesh model (door / door_glass): a leaf runs 0..MESH_W along +X from the
# hinge and 0..MESH_H up +Z. The plugin rescales the loaded mesh to the configured width/height.
_MESH_W = 0.9
_MESH_H = 2.0

# Body name the casing arrives under (the frame model's own body, attached with a "frame_" prefix).
_FRAME_BODY = "frame_frame"


@dataclass
class DoorHandle:
    """Published by :class:`DoorPlugin`; consumed by in-process drivers (scripts, tests).

    ``set_openness``/``get_openness`` traffic in the openness fraction (0 = closed, 1 = fully open).
    ``read_state`` returns ``(openness, d_openness/dt)`` -- the reader the GripperCommand action
    handler watches to report reached/stalled. All run on the physics thread.
    """

    name: str
    set_openness: Callable[[float], None]
    get_openness: Callable[[], float]
    read_state: Callable[[], tuple[float, float]]


class DoorPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True
    _ROOT_BODY = "door_leaf"
    _JOINT = "hinge"
    _ACTUATOR = "hinge_pos"

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.door_name = self.address
        self.prefix = self.config.get("prefix", "")
        pos = self.config.get("pos", [0.0, 0.0, 0.0])
        if len(pos) in (2, 3):
            self.pos = [float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)]
        else:
            self.pos = [0.0, 0.0, 0.0]
        rpy = self.config.get("rpy", [0.0, 0.0, 0.0])
        self.quat = (
            _rpy_to_quat(*(float(v) for v in rpy)) if len(rpy) == 3 else [1.0, 0.0, 0.0, 0.0]
        )

        # Geometry. Bad values are tolerated here (kept as the default) so validate_config reports
        # them with a friendly message rather than crashing construction.
        self.width = self._float(self.config.get("width"), 0.9)
        self.height = self._float(self.config.get("height"), 2.0)
        self.thickness = self._float(self.config.get("thickness"), 0.04)
        # A hinged door hangs with a small clearance top and bottom; without a floor gap the leaf
        # scrapes the ground and the contact friction fights the actuator (the door never opens).
        self.floor_gap = self._float(self.config.get("floor_gap"), 0.01)
        # The leaf is hung on the FACE of the frame, proud of the wall by `mount_offset` (m) on the
        # side it swings toward -- not centred in the wall thickness. This is what lets it open well
        # past 90 deg: the leaf sweeps in free space in front of the wall instead of jamming into the
        # reveal. The frame lines the opening behind it. ~0.06 clears a standard 0.1 m wall.
        self.mount_offset = self._float(self.config.get("mount_offset"), 0.06)
        self.model = self.config.get("model")
        # A leafless door is a cased OPENING: casing only, no hinge/actuator/ROS surface.
        self.leaf = bool(self.config.get("leaf", True))
        # Repaint colours (None = keep the model's own). The casing follows the leaf unless given its
        # own colour, so `color` alone repaints the whole door unit.
        self.color = self._rgba(self.config.get("color"))
        self.frame_color = self._rgba(self.config.get("frame_color")) or self.color
        # Static frame (jamb + lintel casing) welded around the opening while the leaf hinges inside
        # it. On by default; `frame: false` (or an unresolvable frame_model) leaves the bare opening.
        self.frame = bool(self.config.get("frame", True))
        self.frame_model = self.config.get("frame_model", "door_frame")
        self.hinge_side = str(self.config.get("hinge_side", "left")).lower()
        self.swing = 1 if self._float(self.config.get("swing"), 1.0) >= 0 else -1
        self.max_angle = math.radians(self._float(self.config.get("max_angle"), 120.0))
        self.open0 = min(max(self._float(self.config.get("open"), 0.0), 0.0), 1.0)
        self.controllable = bool(self.config.get("controllable", True))
        self.kp = self._float(self.config.get("kp"), 40.0)
        self.kv = self._float(self.config.get("kv"), 8.0)
        # An automatic door only pushes *gently*: the actuator force is capped, so an obstacle
        # (a robot, a person) is nudged, never crushed. If the door stays blocked -- barely moving
        # while short of its target -- for `stall_timeout` seconds, it gives up and holds where it is
        # until commanded again (a real safety door stops pressing on an obstruction).
        self.max_torque = self._float(self.config.get("max_torque"), 15.0)
        self.stall_timeout = self._float(self.config.get("stall_timeout"), 3.0)
        self.stall_speed = self._float(self.config.get("stall_speed"), 0.02)  # fraction/s

        # Signed fully-open angle (rad); openness fraction maps linearly onto [0, _span].
        self._span = self.swing * self.max_angle
        self._target = self.open0  # commanded openness fraction (from ROS / the handle)
        self._effective = self.open0  # what we actually drive to (backed off when it gives up)
        self._stall_t = 0.0  # seconds spent blocked short of the target
        self._gave_up = False  # stopped pushing after a stall; reset on a new command
        self._reach_tol = 0.02  # openness within this of the target counts as reached
        # resolved in configure()
        self._ctx: SimContext | None = None
        self._aid = -1
        self._jid = -1
        self._qadr = -1
        self._dadr = -1

    @staticmethod
    def _float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _rgba(value) -> list[float] | None:
        """An ``[r, g, b]`` / ``[r, g, b, a]`` config colour as rgba; ``None`` if unset or malformed.

        Malformed values are dropped here (the model keeps its own colours) and reported by
        :meth:`validate_config`, so a typo never crashes construction.
        """
        if value is None:
            return None
        try:
            rgba = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        if len(rgba) == 3:
            rgba.append(1.0)
        return rgba if len(rgba) == 4 else None

    @staticmethod
    def _paint(spec: mujoco.MjSpec, rgba: list[float], *, colliding_only: bool) -> None:
        """Repaint ``spec``'s geoms via their materials (a geom without one gets its own rgba).

        ``colliding_only`` skips non-colliding geometry, which is this library's convention for
        decoration (a door's chrome handle) -- it keeps its finish while the panel takes the colour.
        """
        mats = {m.name: m for m in spec.materials}
        for geom in spec.geoms:
            if colliding_only and geom.contype == 0 and geom.conaffinity == 0:
                continue
            mat = mats.get(geom.material)
            if mat is not None:
                mat.rgba = rgba
            else:
                geom.rgba = rgba

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        for key in (
            "width",
            "height",
            "thickness",
            "max_angle",
            "kp",
            "kv",
            "max_torque",
            "stall_timeout",
        ):
            if key not in config:
                continue
            try:
                if float(config[key]) <= 0:
                    errors.append(f"'{key}' must be > 0")
            except (TypeError, ValueError):
                errors.append(f"'{key}' must be a number > 0")
        if "open" in config:
            try:
                if not 0.0 <= float(config["open"]) <= 1.0:
                    errors.append("'open' must be a fraction in [0, 1]")
            except (TypeError, ValueError):
                errors.append("'open' must be a number in [0, 1]")
        for key in ("color", "frame_color"):
            if key in config and self._rgba(config[key]) is None:
                errors.append(f"'{key}' must be [r, g, b] or [r, g, b, a] numbers")
        if not bool(config.get("leaf", True)) and not bool(config.get("frame", True)):
            errors.append("'leaf: false' with 'frame: false' would place nothing; keep the casing")
        if "hinge_side" in config and str(config["hinge_side"]).lower() not in ("left", "right"):
            errors.append("'hinge_side' must be 'left' or 'right'")
        if "swing" in config and float(config.get("swing", 1)) == 0:
            errors.append("'swing' must be non-zero (+1 or -1)")
        if "rpy" in config and len(config["rpy"]) != 3:
            errors.append("'rpy' must be [roll, pitch, yaw] in radians")
        if len(config.get("pos", [0, 0, 0])) not in (2, 3):
            errors.append("'pos' must be [x, y] or [x, y, z]")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        child = mujoco.MjSpec()
        child.compiler.degree = False  # joint range / axis below are in radians

        if not self.leaf:
            # A cased opening: the casing alone, welded. No body/joint/actuator -- there is nothing to
            # swing, so the hinge machinery below would only add an unused DOF to the model.
            self._add_frame(child)
            frame = spec.worldbody.add_frame()
            frame.pos = self.pos
            frame.quat = self.quat
            spec.attach(child, prefix=self.prefix, frame=frame)
            return

        # Hinge is at the fixed edge of the opening; the leaf spans from there to the far edge.
        # 'left'  -> hinge at the -x half-edge, leaf extends +x;
        # 'right' -> hinge at the +x half-edge, leaf extends -x.
        hinge_x = -self.width / 2 if self.hinge_side == "left" else self.width / 2
        leaf_dir = 1.0 if self.hinge_side == "left" else -1.0

        body = child.worldbody.add_body()
        body.name = self._ROOT_BODY
        # Hang the leaf on the frame face, proud of the wall on the side it opens toward, so it clears
        # the reveal and can swing past 90 deg. The side it sweeps toward is leaf_dir * swing (a
        # +x leaf rotating +theta goes +y; a -x leaf goes -y), so the mount offset follows that -- not
        # `swing` alone, which would put a right-hinge leaf on the wrong face.
        open_side = leaf_dir * self.swing
        body.pos = [hinge_x, open_side * self.mount_offset, 0.0]

        joint = body.add_joint()
        joint.name = self._JOINT
        joint.type = mujoco.mjtJoint.mjJNT_HINGE
        joint.axis = [0.0, 0.0, 1.0]
        joint.range = sorted([0.0, self._span])
        joint.limited = True

        self._add_leaf(child, body, leaf_dir)
        self._add_frame(child)

        act = child.add_actuator()
        act.name = self._ACTUATOR
        act.set_to_position(self.kp, self.kv)
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = self._JOINT
        act.ctrlrange = sorted([0.0, self._span])
        act.ctrllimited = True
        # Gentle push: cap the hinge torque so a blocked door can't force through an obstacle.
        act.forcerange = [-self.max_torque, self.max_torque]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    def _add_leaf(self, child: mujoco.MjSpec, body, leaf_dir: float) -> None:
        """Hang the leaf on the hinge body: a textured mesh model if ``model`` is set, else a box.

        Convention: a door mesh model's origin is its hinge edge at floor level, leaf along +X, up
        +Z. We attach it into the hinge body via a frame; ``leaf_dir < 0`` (a right-hand hinge)
        turns it 180 deg about +Z so it still spans into the opening. A box leaf reproduces the same
        convention procedurally, so both kinds hang identically. Both are lifted by ``floor_gap`` and
        fitted inside ``height`` with the same clearance top and bottom, so the leaf never scrapes.
        """
        leaf_h = max(self.height - 2 * self.floor_gap, 0.01)  # clearance top and bottom
        if self.model:
            try:
                asset = resolve_model(self.model)
            except ModelError as exc:
                raise RuntimeError(
                    f"door {self.door_name!r}: leaf model {self.model!r} could not be resolved ({exc})"
                ) from exc
            leaf = mujoco.MjSpec.from_file(str(asset.path))
            apply_assets(leaf, asset)
            if self.color:
                self._paint(leaf, self.color, colliding_only=True)
            # Rescale the authored 0.9 x 2.0 m leaf to this opening (thickness kept native). At the
            # default size the width factor is 1.0, so standard doors are undistorted in X.
            for mesh in leaf.meshes:
                s = list(mesh.scale)
                mesh.scale = [s[0] * self.width / _MESH_W, s[1], s[2] * leaf_h / _MESH_H]
            frame = body.add_frame()
            frame.pos = [0.0, 0.0, self.floor_gap]
            # +X points from the hinge toward the far edge; flip for a right-hand hinge.
            frame.quat = [1.0, 0.0, 0.0, 0.0] if leaf_dir > 0 else [0.0, 0.0, 0.0, 1.0]
            child.attach(leaf, prefix="leaf_", frame=frame)
            return

        mat = child.add_material()
        mat.name = "door_leaf"
        mat.rgba = self.color or _LEAF_RGBA
        g = body.add_geom()
        g.name = "leaf"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [self.width / 2, self.thickness / 2, leaf_h / 2]
        g.pos = [leaf_dir * self.width / 2, 0.0, self.floor_gap + leaf_h / 2]
        g.material = "door_leaf"

    def _add_frame(self, child: mujoco.MjSpec) -> None:
        """Weld a static frame (jamb + lintel casing) around the opening; the leaf hinges inside it.

        Centred on the opening (its own origin is the opening centre) and rescaled to width/height
        like the leaf, so leaf and frame stay consistent for any opening. Skipped -- with a warning,
        never fatal -- if ``frame`` is off or the frame model can't be resolved, leaving a bare
        opening. The frame is a separate static body (no joint), so ``attach`` welds it in place.
        """
        if not self.frame:
            return
        try:
            asset = resolve_model(self.frame_model)
        except ModelError as exc:
            logger.warning(
                "door %r: frame model %r unresolved (%s); leaving a bare opening",
                self.door_name,
                self.frame_model,
                exc,
            )
            return
        fr = mujoco.MjSpec.from_file(str(asset.path))
        apply_assets(fr, asset)
        if self.frame_color:
            # The casing is uniform trim and visual-only (contype 0), so every geom takes the colour.
            self._paint(fr, self.frame_color, colliding_only=False)
        for mesh in fr.meshes:
            s = list(mesh.scale)
            mesh.scale = [s[0] * self.width / _MESH_W, s[1], s[2] * self.height / _MESH_H]
        frame = child.worldbody.add_frame()
        frame.pos = [0.0, 0.0, 0.0]
        child.attach(fr, prefix="frame_", frame=frame)

    def configure(self, ctx: SimContext) -> None:
        self._ctx = ctx
        m = ctx.model
        p = self.prefix
        if not self.leaf:
            # A cased opening still registers as a door -- on the casing body, since there is no leaf --
            # so anything enumerating the building's doors finds it. No handle, no endpoints: there is
            # nothing to command.
            ctx.entities.add(
                Entity(
                    name=self.door_name,
                    kind="door",
                    body=p + _FRAME_BODY,
                    meta={"prefix": p, "leaf": False, "width": self.width, "height": self.height},
                )
            )
            # Only complain when a world explicitly asked for it -- `controllable` defaults to true, and
            # a leafless entry that never mentions it has nothing to be warned about.
            if self.config.get("controllable"):
                logger.warning(
                    "door %r: 'controllable' is ignored on a leafless door (nothing to command)",
                    self.door_name,
                )
            return
        self._jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p + self._JOINT)
        self._aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, p + self._ACTUATOR)
        if self._jid < 0 or self._aid < 0:
            raise RuntimeError(
                f"door {self.door_name!r}: hinge joint/actuator {p + self._JOINT!r} not found"
            )
        self._qadr = int(m.jnt_qposadr[self._jid])
        self._dadr = int(m.jnt_dofadr[self._jid])

        ctx.entities.add(
            Entity(
                name=self.door_name,
                kind="door",
                body=p + self._ROOT_BODY,
                meta={
                    "prefix": p,
                    "hinge_side": self.hinge_side,
                    "swing": self.swing,
                    "max_angle": self.max_angle,
                },
            )
        )

        # In-process handle + the state reader the GripperCommand action handler watches.
        ctx.blackboard.set(
            f"door:{self.door_name}",
            DoorHandle(
                name=self.door_name,
                set_openness=self.set_openness,
                get_openness=lambda: self._target,
                read_state=self.read_state,
            ),
        )

        if not self.controllable:
            return  # passive door: holds `open`, no ROS surface

        ns = self.config.get("namespace", "")
        # State reader keyed by the door's own name so the (generalized) GripperCommand handler finds
        # it without the door pretending to be a gripper (see roqsim_ros_bridge.actions).
        ctx.blackboard.set(f"door:{self.door_name}:state", self.read_state)
        ctx.interface.add(
            Endpoint(
                name="cmd",
                direction="in",
                owner=self.door_name,
                namespace=ns,
                write=self.set_openness,
                backend={
                    "ros2": {
                        "type": "std_msgs.msg.Float64",
                        "topic": self.topic_override("cmd") or "cmd",
                    }
                },
            )
        )
        ctx.interface.add(
            Endpoint(
                name="state",
                direction="out",
                owner=self.door_name,
                namespace=ns,
                read=lambda: self._target if self._ctx is None else self.read_state()[0],
                rate_hz=10.0,
                backend={
                    "ros2": {
                        "type": "std_msgs.msg.Float64",
                        "topic": self.topic_override("state") or "state",
                    }
                },
            )
        )
        ctx.interface.add(
            Endpoint(
                name="door",
                direction="in",
                owner=self.door_name,
                namespace=ns,
                write=self.set_openness,
                backend={
                    "ros2": {
                        "action": "control_msgs.action.GripperCommand",
                        "name": self.topic_override("door") or "door",
                        # Where the handler reads (position, velocity) for reached/stalled -- the
                        # door's own reader, not a gripper's (handler defaults to gripper:<owner>).
                        "state_key": f"door:{self.door_name}:state",
                    }
                },
            )
        )

    def set_openness(self, openness: float) -> None:
        """Set the target openness fraction (0 = closed, 1 = fully open); clamped.

        A fresh command clears any earlier give-up, so the door tries again toward the new target.
        """
        self._target = min(max(float(openness), 0.0), 1.0)
        self._stall_t = 0.0
        self._gave_up = False

    def read_state(self) -> tuple[float, float]:
        """(openness, d_openness/dt) as fractions -- computed on demand on the physics thread."""
        d = self._ctx.data
        if self._span == 0.0:
            return (0.0, 0.0)
        return (float(d.qpos[self._qadr]) / self._span, float(d.qvel[self._dadr]) / self._span)

    def on_reset(self, ctx: SimContext) -> None:
        if not self.leaf:
            return  # nothing to reset: a cased opening is static
        self._target = self._effective = self.open0
        self._stall_t = 0.0
        self._gave_up = False
        ctx.data.qpos[self._qadr] = self.open0 * self._span
        ctx.data.ctrl[self._aid] = self.open0 * self._span
        mujoco.mj_forward(ctx.model, ctx.data)

    def pre_step(self, ctx: SimContext) -> None:
        if not self.leaf:
            return  # a cased opening has no actuator
        if ctx.manual_control:
            return  # the viewer's slider owns the hinge actuator this run
        openness, speed = self.read_state()
        if not self._gave_up and abs(openness - self._target) > self._reach_tol:
            # Trying to reach the target: if the leaf is barely moving it is blocked -- count the
            # stall, and once it exceeds the timeout stop pushing and hold where it is.
            if abs(speed) < self.stall_speed:
                self._stall_t += ctx.dt
            else:
                self._stall_t = 0.0
            if self._stall_t >= self.stall_timeout:
                self._gave_up = True
                self._effective = openness  # hold at the obstruction, stop pressing
            else:
                self._effective = self._target
        elif not self._gave_up:
            self._stall_t = 0.0
            self._effective = self._target
        ctx.data.ctrl[self._aid] = self._effective * self._span
