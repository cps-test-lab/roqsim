"""The sensor catalog: which sensor types may be placed, their default FOV, cost, and mount limits.

A catalog entry pairs a sensor *type* (which selects the FOV adapter) with a default FOV config and a
:class:`MountConstraint` describing the realistic *constrained-mount* placement space -- where in the
room this sensor may go (wall/ceiling) and at what height. The placement search proposes poses inside
these constraints; the evaluator turns each into a :class:`~.adapters.PlacedSensor` and scores it.

**An entry states policy, not optics.** ``cost``, ``mount`` and ``description`` are judgements about
how a device may be deployed and are written here. The optics are *read from the model the entry
names*: ``fovy``/``resolution`` off its MJCF camera, ``near``/``far`` off its manifest's ``fov:``
block (see :func:`_model_optics`). Lidar templates stay empty for the same reason by another route --
their adapters instantiate the plugin and read its resolved defaults.

Restating those numbers here is what an entry must not do: a hand-written copy of a model's optics
cannot be kept true by attention, and a catalog that disagrees with the model it names is worse than
one that says nothing. There is deliberately no field to write one in.

Every camera entry derives; there is no exception. That includes ``oakd_camera``, which needs the
OAK-D Pro model in this package to derive from -- reading ``roqsim_mobile``'s turtlebot4 camera would
mean depending on a package this one cannot depend on.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import mujoco
import numpy as np

from roqsim.config import parse_plugin_entry
from roqsim.manifest import load_manifest, manifest_fov
from roqsim.models import resolve_model

from .adapters import PlacedSensor


@dataclass
class MountConstraint:
    surfaces: tuple[str, ...]  # e.g. ("wall", "ceiling")
    z_range: tuple[float, float]  # allowed mount height [m]
    can_tilt: bool


def _manifest_camera(model_file) -> str | None:
    """The camera name a model's manifest already gives its capture plugin, if it names one.

    The manifest says ``realsense_d435: {camera: d435_color}`` to wire the plugin, so the model has
    already stated which of its cameras is the imaging one. Reading that back is what lets this module
    stay out of the business of knowing device internals -- no plugin class is imported, and a model
    with two cameras is disambiguated by its own file rather than by a rule here.
    """
    for entry in load_manifest(model_file):
        camera = parse_plugin_entry(entry, "manifest plugin").config.get("camera")
        if camera:
            return str(camera)
    return None


@functools.cache
def _model_optics(model_ref: str) -> dict:
    """A sensor model's own optics: ``fovy``/``width``/``height`` from its MJCF, ``near``/``far`` from
    its manifest.

    Parses the MJCF without compiling it -- ``MjSpec.from_file`` does not need the meshes on disk
    (which is why :meth:`~roqsim_sensors.plugins.spawn_sensor.SpawnSensorPlugin.build` can call
    ``apply_assets`` *after* it), so this works for the three sensors whose meshes are generated and
    absent on a fresh clone. Parsing rather than scraping the XML also means MJCF ``<default>``
    classes are honoured.

    Raises rather than defaulting on anything missing. A silent ``{}`` here would fall through to
    :func:`~roqsim_sensors.coverage.adapters.camera_adapter`'s last-resort constants (``fovy 45``,
    ``far 10``) and quietly report coverage for a lens no device has.
    """
    asset = resolve_model(model_ref)
    spec = mujoco.MjSpec.from_file(str(asset.path))
    cameras = list(spec.cameras)
    if not cameras:
        raise ValueError(f"catalog: model {model_ref!r} has no camera to take optics from")
    wanted = _manifest_camera(asset.path)
    if wanted is not None:
        cams = [c for c in cameras if c.name == wanted]
        if not cams:
            raise ValueError(
                f"catalog: model {model_ref!r} manifest names camera {wanted!r}, which the MJCF "
                f"does not have. It has: {[c.name for c in cameras]}"
            )
        cam = cams[0]
    elif len(cameras) == 1:
        cam = cameras[0]
    else:
        raise ValueError(
            f"catalog: model {model_ref!r} has {len(cameras)} cameras "
            f"({[c.name for c in cameras]}) and its manifest names none -- add `camera:` to the "
            f"manifest's capture-plugin entry so the imaging one is stated by the model."
        )

    width, height = (int(v) for v in cam.resolution)
    if width <= 1 or height <= 1:
        raise ValueError(
            f"catalog: model {model_ref!r} camera {cam.name!r} has no `resolution` "
            f"(MuJoCo's unset sentinel is 1x1); the frustum's aspect ratio comes from it."
        )
    fov = manifest_fov(asset.path)
    missing = [k for k in ("near", "far") if k not in fov]
    if missing:
        raise ValueError(
            f"catalog: model {model_ref!r} manifest has no `fov:` {missing} -- a placement's "
            f"detection range is the model's to state, not this catalog's to invent."
        )
    return {
        "fovy": float(cam.fovy),
        "width": width,
        "height": height,
        "near": float(fov["near"]),
        "far": float(fov["far"]),
    }


@dataclass
class SensorSpec:
    type: str
    cost: float
    mount: MountConstraint
    #: Bundled model the optics are read from. Required for a camera; a lidar leaves it empty because
    #: its adapter instantiates the plugin and reads the defaults that resolved.
    model: str = ""
    #: What the model cannot state. ``far`` on a camera is an analysis assumption rather than a
    #: property of the device, so a study may pin it here; everything else comes from the model.
    fov_overrides: dict = field(default_factory=dict)
    description: str = ""

    @property
    def fov_template(self) -> dict:
        """Optics read from the model, overridden by anything the model cannot state.

        A fresh dict each call: the cache behind :func:`_model_optics` is process-wide, and
        :func:`placed_from_proposal` merges a proposal's config into what it gets back.
        """
        base = _model_optics(self.model) if self.model else {}
        return {**base, **self.fov_overrides}


#: Built-in catalog. ``far`` on cameras is an analysis detection range (not physics); tune per study.
CATALOG: dict[str, SensorSpec] = {
    "oakd_camera": SensorSpec(
        type="oakd_camera",
        model="roqsim_sensors:oakd",
        cost=1.0,
        mount=MountConstraint(surfaces=("wall", "ceiling"), z_range=(0.5, 3.5), can_tilt=True),
        description="OAK-D Pro RGB-D camera (narrow-ish FOV, medium range).",
    ),
    "realsense_d415": SensorSpec(
        type="realsense_d415",
        model="roqsim_sensors:d415",
        cost=0.8,
        mount=MountConstraint(surfaces=("wall", "ceiling"), z_range=(0.5, 3.5), can_tilt=True),
        description="RealSense D415 RGB camera (narrow FOV, 0.45 m min depth).",
    ),
    "realsense_d435": SensorSpec(
        type="realsense_d435",
        model="roqsim_sensors:d435",
        cost=0.8,
        mount=MountConstraint(surfaces=("wall", "ceiling"), z_range=(0.5, 3.5), can_tilt=True),
        description="RealSense D435 RGB camera.",
    ),
    "realsense_d455": SensorSpec(
        type="realsense_d455",
        model="roqsim_sensors:d455",
        cost=1.0,
        mount=MountConstraint(surfaces=("wall", "ceiling"), z_range=(0.5, 3.5), can_tilt=True),
        description="RealSense D455 RGB camera (wide 87x62 deg FOV, 0.6-6 m range).",
    ),
    "zivid": SensorSpec(
        type="zivid",
        model="roqsim_sensors:zivid",
        cost=2.0,
        mount=MountConstraint(surfaces=("wall", "ceiling"), z_range=(1.3, 3.0), can_tilt=True),
        description="Zivid 2+ structured-light RGB-D (narrow FOV, 1.3-5 m working range).",
    ),
    "livox_mid360": SensorSpec(
        type="livox_mid360",
        cost=3.0,  # adapter reads the Livox plugin defaults (360 x -7..52 deg, 40 m)
        mount=MountConstraint(surfaces=("ceiling",), z_range=(1.0, 4.0), can_tilt=True),
        description="Livox Mid-360 3D lidar (dome FOV, long range). Invert (roll=pi) to face down; "
        "inverted it has a blind disk directly beneath it of radius ~= (mount_z - target_z)*tan(90-v_max) "
        "(~2.6 m on the floor from a 3.3 m ceiling), so mount near walls/corners, not room centres.",
    ),
    "lidar": SensorSpec(
        type="lidar",
        cost=0.5,  # adapter reads the 2D lidar plugin defaults (planar fan)
        mount=MountConstraint(surfaces=("wall",), z_range=(0.2, 2.0), can_tilt=False),
        description="2D lidar (planar fan). Coverage is a thin horizontal slice at the mount height.",
    ),
}


def catalog_as_dict() -> dict:
    """JSON-serialisable view of the catalog (for the CLI ``catalog`` command / LLM planner)."""
    out = {}
    for name, spec in CATALOG.items():
        out[name] = {
            "type": spec.type,
            "fov_template": spec.fov_template,
            "cost": spec.cost,
            "mount": {
                "surfaces": list(spec.mount.surfaces),
                "z_range": list(spec.mount.z_range),
                "can_tilt": spec.mount.can_tilt,
            },
            "description": spec.description,
        }
    return out


def placed_from_proposal(proposal: dict, *, index: int = 0) -> PlacedSensor:
    """Build a hypothetical :class:`PlacedSensor` from a placement proposal dict.

    ``proposal`` = ``{"type": str, "pos": [x,y,z], "rpy": [r,p,y], "config": {..}}``. The catalog's
    ``fov_template`` supplies defaults; ``config`` overrides them per placement.
    """
    stype = proposal["type"]
    if stype not in CATALOG:
        raise KeyError(f"unknown sensor type {stype!r}; catalog has {sorted(CATALOG)}")
    config = dict(CATALOG[stype].fov_template)
    config.update(proposal.get("config") or {})
    return PlacedSensor(
        sensor_type=stype,
        pos=np.asarray(proposal["pos"], dtype=np.float64),
        rpy=np.asarray(proposal.get("rpy", [0.0, 0.0, 0.0]), dtype=np.float64),
        config=config,
        label=proposal.get("label", f"{stype}_{index}"),
    )
