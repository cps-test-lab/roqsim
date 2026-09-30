"""Scene plugin: add an ArUco / AprilTag fiducial marker to the world or onto a robot body.

A fiducial is a flat image mapped 1:1 onto a thin box geom so the camera plugins render it and a
real detector (``cv2.aruco.detectMarkers`` / an AprilTag detector) can decode it from the rendered
image. The marker bitmap is generated at *build* time with OpenCV (``cv2.aruco`` -- one dependency
covers both ArUco ``DICT_*`` dictionaries and the AprilTag families) and injected directly as raw
``MjSpec`` texture data, so nothing is written to disk. OpenCV is an optional dependency::

    pip install 'roqsim_sensors[markers]'

Placement is fixed (a static, non-colliding geom): either free-standing in the world or welded to a
named body of an already-spawned robot/arm (``attach_to`` + ``prefix``), at ``pose`` in that frame.
For the body-attached form this plugin must be listed *after* the spawn plugin in the world YAML, so
the prefixed body already exists in the spec when ``build`` runs.

**``emission`` trades detectability for pose accuracy, and defaults to 0 for that reason.** A raised
emission does keep a tag legible in a dim scene, but it also lifts the *black* texels toward grey and
saturates the white ones, which shifts the detector's threshold outward and makes the decoded square
larger than the geometry. That is not cosmetic: a pose estimator scales range by (declared size /
apparent size), so an apparently-larger tag is reconstructed closer. Measured against a world's own
projected marker corners (1280x720, 19 cm tag with the 5 mm quiet zone a real 20 cm cube leaves),
``emission: 0.4`` inflated the decoded tag by up to **9 %**, i.e. ~13 cm of range error at 1.2 m,
where ``emission: 0`` was stable to ~1.5 %. The effect is strongly coupled to the quiet zone: with a
generous margin it disappears, which is why it is easy to miss at the 0.15 default ``quiet_zone``.
Raise it only when a tag would otherwise not be found at all, and never in a run whose numbers are
poses.

Config::

    fiducial_marker:
      family: apriltag_36h11   # apriltag_36h11 | apriltag_25h9 | aruco_4x4_50 | aruco_5x5_100 | ...
      id: 0                    # marker id within the family
      size: 0.05               # side of the BLACK marker square (m) -- the length a detector estimates
      quiet_zone: 0.15         # white margin around the tag, as a fraction of `size` (default 0.15)
      emission: 0.0            # material emission; RAISE ONLY FOR VISIBILITY, NOT FOR DETECTION (see below)
      thickness: 0.002         # box half-thickness behind the marker face (m)
      vflip: false             # flip the texture rows / cols if the render comes out mirrored
      hflip: false             #   (a mirrored tag will NOT decode)
      pose:                    # the marker centre, in the world or in the `attach_to` body's frame:
        position: {x: 0.0, y: 0.0, z: 0.0}     #   a geometry_msgs/Pose, omitted components 0
        orientation: {roll: 0.0, pitch: 0.0, yaw: 0.0}   # or a quaternion; none faces +Z (up)
      attach_to: wrist_3_link  # OPTIONAL: weld to this body (without prefix) instead of the world
      prefix: "ur10e_"         # target robot/arm MJCF prefix (matches spawn_robot/spawn_arm `prefix`);
                               #   inherited automatically when this plugin ships in a model manifest

The texture, material and geom this builds are named after the entry's label (its ``name:``
sibling, else ``fiducial_marker``), so a world carrying several markers gives each entry a label.
"""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim.context import SimContext
from roqsim.plugin import Plugin
from roqsim.pose import config_pose, config_pose_errors, pose_spelling

#: Keys that would split a pose into a position and an orientation, per frame.
_SPLIT_POSE = ("quat", "rel_pose", "rel_quat", "rel_rpy")


def _pose_errors(config: dict) -> list[str]:
    """``pose`` as a mapping, and none of the keys that would split it in two."""
    pose = config.get("pose")
    if isinstance(pose, (list, tuple)):
        return [
            "'pose' is a mapping, a geometry_msgs/Pose relative to the world or the 'attach_to' "
            f"body: {pose_spelling(pose, config.get('rpy') or config.get('rel_rpy'))}"
        ]
    errors = [
        f"'{key}' is not read -- state the whole placement as 'pose', a geometry_msgs/Pose relative "
        "to the world or the 'attach_to' body: pose: {position: {x, y, z}, orientation: "
        "{roll, pitch, yaw} or {x, y, z, w}}, omitted components 0"
        for key in _SPLIT_POSE
        if key in config
    ]
    return errors + config_pose_errors(config, "fiducial_marker")


def _dict_name(family: str) -> str:
    """Map a friendly family name to a ``cv2.aruco.DICT_*`` attribute name.

    ``apriltag_36h11`` -> ``DICT_APRILTAG_36h11``; ``aruco_4x4_50`` -> ``DICT_4X4_50``; a raw
    ``DICT_...`` / ``dict_...`` name is accepted as-is.
    """
    f = family.strip()
    low = f.lower()
    if low.startswith("dict_"):
        return "DICT_" + f[5:].upper()
    if low.startswith("apriltag_"):
        return "DICT_APRILTAG_" + f[len("apriltag_") :]
    if low.startswith("aruco_"):
        return "DICT_" + f[len("aruco_") :].upper()
    return "DICT_" + f.upper()


def _import_cv2():
    try:
        import cv2  # noqa: PLC0415

        return cv2
    except ImportError as exc:  # pragma: no cover - exercised via validate_config
        raise ImportError(
            "fiducial_marker needs opencv-python: pip install 'roqsim_sensors[markers]'"
        ) from exc


class FiducialMarkerPlugin(Plugin):
    """Build-only plugin: generate a marker texture and add its geom to the scene."""

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.family = str(self.config.get("family", "apriltag_36h11"))
        self.marker_id = int(self.config.get("id", 0))
        self.size = float(self.config.get("size", 0.05))
        self.quiet_zone = float(self.config.get("quiet_zone", 0.15))
        self.emission = float(self.config.get("emission", 0.0))
        self.thickness = float(self.config.get("thickness", 0.002))
        self.vflip = bool(self.config.get("vflip", False))
        self.hflip = bool(self.config.get("hflip", False))
        base = self.label
        self.base_name = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(base))

    def validate_config(self, config: dict) -> list[str]:
        errors: list[str] = []
        if float(config.get("size", 0.05)) <= 0:
            errors.append("'size' must be > 0")
        if float(config.get("quiet_zone", 0.15)) < 0:
            errors.append("'quiet_zone' must be >= 0")
        errors += _pose_errors(config)
        # Validate the family/id against the actual OpenCV dictionary (also surfaces a missing cv2).
        try:
            cv2 = _import_cv2()
        except ImportError as exc:
            errors.append(str(exc))
            return errors
        name = _dict_name(str(config.get("family", "apriltag_36h11")))
        const = getattr(cv2.aruco, name, None)
        if const is None:
            errors.append(f"unknown fiducial family {config.get('family')!r} (-> cv2.aruco.{name})")
        else:
            dictionary = cv2.aruco.getPredefinedDictionary(const)
            n = int(dictionary.bytesList.shape[0])
            mid = int(config.get("id", 0))
            if not 0 <= mid < n:
                errors.append(f"'id' {mid} out of range for {config.get('family')} (0..{n - 1})")
        return errors

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        rgb = self._render_marker()
        h, w = rgb.shape[:2]

        tex = spec.add_texture()
        tex.name = self.base_name
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
        tex.width = w
        tex.height = h
        tex.nchannel = 3
        tex.data = rgb.tobytes()

        mat = spec.add_material()
        mat.name = f"{self.base_name}_mat"
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = self.base_name
        # Per-face mapping: one full copy of the texture on each box face (texrepeat is per-face when
        # texuniform is off; texuniform would instead scale the texture by geom size and clip the tag).
        mat.texrepeat = [1, 1]
        mat.specular = 0.0
        mat.shininess = 0.0
        mat.reflectance = 0.0
        mat.emission = self.emission

        # Square textured face + white quiet zone, so the printed `size` stays the black-square side.
        face = self.size * (1.0 + 2.0 * self.quiet_zone) / 2.0

        if "attach_to" in self.config:
            prefix = self.config.get("prefix", "")
            body_name = prefix + self.config["attach_to"]
            parent = spec.body(body_name)
            if parent is None:
                raise RuntimeError(
                    f"fiducial_marker: body {body_name!r} not found -- list this plugin after the "
                    f"spawn plugin, and check 'prefix'"
                )
            g = parent.add_geom()
        else:
            g = spec.worldbody.add_geom()
        pos, quat = config_pose(self.config)

        g.name = f"{self.base_name}_geom"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [face, face, self.thickness]
        g.pos = [float(v) for v in pos]
        g.quat = quat
        g.material = mat.name
        g.contype = 0  # a marker is purely visual: no collisions
        g.conaffinity = 0
        g.group = 2  # visual group, like the model meshes

    def _render_marker(self) -> np.ndarray:
        """Generate the marker as an (H, W, 3) uint8 RGB array: black tag on white + quiet zone."""
        cv2 = _import_cv2()
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _dict_name(self.family)))
        bits = int(dictionary.markerSize) + 2  # include the marker's own 1-bit black border
        px_per_cell = 24
        side = bits * px_per_cell
        # OpenCV 4.7+ renamed drawMarker -> generateImageMarker; support both.
        gen = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker
        img = gen(dictionary, self.marker_id, side)  # uint8, 0/255, black tag on white
        pad = int(round(self.quiet_zone * side))
        if pad > 0:
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        if self.vflip:
            img = img[::-1, :]
        if self.hflip:
            img = img[:, ::-1]
        return np.repeat(np.ascontiguousarray(img)[:, :, None], 3, axis=2).astype(np.uint8)
