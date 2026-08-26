"""Scene plugin: attach a standalone sensor MJCF (mesh + camera/site) into the world at a mount pose.

The generic, robot-free analogue of ``spawn_arm``/``spawn_robot``: for a sensor that isn't carried
by a robot -- a fixed overhead camera, a mast-mounted lidar -- this plugin only *places* the mount
(build + a fixed pose; there is no free joint to move it at runtime). It registers an
``Entity(kind='sensor')`` the same way a spawned robot/arm does, so a capture plugin
(``lidar``/``oakd_camera``/``realsense_d435``) resolves this mount's ``prefix``/``namespace``
via its own ``robot: <name>`` config -- no sensor-specific wiring needed here.

Config::

    - spawn_sensor:
        model: d435            # bundled model name, filename, or absolute path
        namespace: ""          # optional transport scope; the capture plugin's endpoints inherit it
        prefix: ""             # MJCF name prefix (use distinct prefixes for >1 mount of the model)
        pos: [0.0, 0.0, 0.0]
        rpy: [0.0, 0.0, 0.0]   # mount orientation as roll/pitch/yaw (rad)
        show_fov: false        # reveal / synthesise the sensor's FOV visualisation (see below)
        fov_alpha: 0.25        # per-cone translucency when show_fov is true (0..1); ~0.25 maximises the
                               #   darkness step between single- and multi-sensor overlap
        fov_range: <far>       # far plane of a synthesised camera frustum (m); default: model manifest
        fov_near: <near>       # near plane (m); >0 truncates the cone; default: model manifest 'fov:'
        fov_rays: [32, 24]     # ray grid [horizontal, vertical] used for the occlusion clip
        intrinsics:            # this UNIT's measured lens, rendered as well as published
          {fx: 1330.23, fy: 1329.37, cx: 974.25, cy: 538.99, width: 1920, height: 1080}
      name: camera_1           # the entry's OWN key, not the config's: this mount's entity

``model``'s ``<model>.manifest.yaml`` (e.g. ``d435.manifest.yaml``) ships the matching capture
plugin, injected automatically the same way a robot's manifest is (see
:func:`roqsim.manifest.expand_manifest`); off with ``default_plugins: false``.

**A measured lens, per placement.** ``intrinsics:`` writes ``fx``/``fy``/``cx``/``cy`` (pixels, at the
resolution they were measured at) onto the model's camera as ``sensorsize`` + ``focalpixel`` +
``principalpixel``, which is what MuJoCo *renders* through -- so the frame really has an off-centre
principal point and ``fx != fy``, and the capture plugin reads the same numbers back out of the compiled
model (:func:`~roqsim_sensors.plugins.camera_common.intrinsics_from_model`, path 1). Pixels and
``camera_info`` are then one claim rather than two.

It belongs to the placement and not to the model because **a calibration describes one physical unit**:
three D435s of one rig measure ``fx`` 1330 / 1344 / 1413 with principal points scattered up to 14 px off
centre, so a shared ``d435.xml`` has no single lens to carry, and a variant per unit would clone a mesh
in order to hold three numbers. Stating them here leaves one model and gives each mount its own optics.

The resolution is part of the measurement, so ``width``/``height`` are required rather than inferred --
the model camera's own ``resolution`` and the size the capture plugin renders at are both within reach
and neither is necessarily the frame the calibration was made in. The plugin may then render at any
size: the intrinsics scale with it. Distortion is a separate matter -- it cannot come out of a projection matrix --
and stays on the capture plugin's ``distortion:``, which warps the render to match.

**FOV visualisation.** ``show_fov: true`` makes the sensor's field of view visible. Three paths, tried
in order: (1) if the model has cameras (e.g. the RealSense/Zivid mounts) a translucent view **frustum**
is synthesised per camera from its ``fovy``/aspect spanning the valid detection band
``fov_near``..``fov_range``, **always clipped against world geometry** into a visibility volume that
stops at walls and objects (see *Occlusion* below); (2) otherwise (a camera-less model), if it ships
FOV geoms -- non-colliding, name ending :data:`FOV_GEOM_SUFFIX` (``_fov``), hidden at rgba alpha 0 --
they are made translucent (``fov_alpha``); (3) otherwise, if the model manifest
declares an angular ``fov:`` band (``h_min``/``h_max``/``v_min``/``v_max`` -- the camera-less lidars,
Mid-360 / Robin W1G) a translucent **sector** shell is synthesised from those datasheet angles between
radii ``fov_near``..``fov_range``, using the very direction convention the capture plugin casts with (so
the drawn shell matches the rays; a >= 2*pi azimuth span is a full 360deg dome). So every sensor can
show its FOV, whether or not it ships a bundled mesh. ``show_fov: true`` on a model with none of a
camera, an ``_fov`` geom, or an angular manifest band is a hard error, not a silent no-op.

The valid range is device knowledge: ``fov_near``/``fov_range`` default to the sensor model's own
``fov: {near, far}`` block in its ``<model>.manifest.yaml`` (a world overrides either per placement),
so ``show_fov: true`` alone draws the correct band without repeating device specs in every world.
``fov_near > 0`` sets the near cap of the synthesised visibility volume so the drawn cone starts at the
near plane -- the shape then *is* the valid range band, not a cone implying validity down to distance 0.
(A camera has no physical range, so these are display values.) A model that *has a camera* (e.g. the
RealSense or Zivid) always synthesises its frustum from that camera, even when it also ships a bundled
``_fov`` envelope (the Zivid does) -- the envelope is only revealed for a *camera-less* model that ships
one. A camera-less lidar (Mid-360, Robin W1G) draws a synthesised angular **sector** (see below), whose
~0.1 m near cutoff is negligible.

**Occlusion (always on for anything synthesised).** A synthesised FOV volume is never drawn as an
idealised cone that passes through walls: it is always clipped into a *visibility volume* that stops at
world geometry. A ray grid is cast from the sensor against the world built so far, each ray clamped at
its first hit, and the drawn mesh spans those hit points (a non-convex ``userface`` mesh). This covers
camera frustums (a ``fov_rays`` grid from the pinhole) and lidar sectors alike (the sector's own
azimuth x elevation grid from the scan site). Only a **bundled** ``_fov`` envelope draws un-clipped --
it is a baked mesh and not re-cuttable. The clip is a **static build-time snapshot**: it raycasts geom
groups 0/1/3 of the partial world, so list scene/floorplan plugins *before* the sensors (a sensor built
before the walls would see none); dynamic bodies (robots) occlude at their spawn pose; the volume never
updates at runtime. Costs one extra world compile per synthesising sensor at build time.

**Overlap reads as darkness.** The cones are translucent and MuJoCo alpha-blends them, so where several
sensors' cones overlap the region accumulates more layers and renders darker -- a visual cue for "how
many sensors see here". ``fov_alpha`` defaults to ``0.25``: the darkness *step* between single- and
double-coverage is largest near this alpha (a(1-a) is maximal around a=0.3) and vanishes at very low
alpha, which is why a barely-translucent cone makes single and double look identical. Raise it toward
opaque only if you want solid cones; lower it only if the cones obscure the scene -- but expect the
overlap cue to weaken. The cones are drawn **double-sided** so you also see the colour when standing
*inside* a field of view (MuJoCo back-face culls, so a single-sided shell vanishes from within); this
means a lone cone shows two layers (its near and far walls) and already reads as tinted, so overlap is
now a *further* darkening rather than the sole cue. This is still qualitative and view-dependent; for a
quantitative, unambiguous per-area count use the coverage density render (``sensor_coverage_probe`` /
``roqsim sensors coverage`` with ``palette: density``).

(Bundled ``_fov`` geoms are identified by name, not geom group, on purpose, and synthesised frustums
live in group :data:`FOV_GEOM_GROUP` (2): the MuJoCo 3.x offscreen renderer drops large geoms in group
4/5 once a scene has several geoms, so an FOV volume must live in a normally-rendered group.)
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim import raycast
from roqsim.context import Entity, SimContext
from roqsim.manifest import expand_manifest, manifest_fov
from roqsim.models import ModelError, apply_assets, resolve_model
from roqsim.plugin import Plugin

#: Name suffix marking a sensor model's FOV-visualisation geoms (non-colliding, hidden until
#: revealed). A name convention, not a geom group -- see the module docstring for why.
FOV_GEOM_SUFFIX = "_fov"

#: Geom group for FOV-visualisation meshes. Group 2, not 4/5: the MuJoCo 3.x offscreen renderer
#: drops large group-4/5 geoms once a scene has several geoms (see the zivid/mid360 models).
FOV_GEOM_GROUP = 2

#: Colour of synthesised camera FOV frustums (saturated blue; alpha set from ``fov_alpha``). Saturated
#: rather than pale so overlapping cones accumulate toward a clearly darker navy.
_FRUSTUM_RGB = (0.10, 0.40, 0.95)

#: Geom groups the FOV occlusion raycast treats as occluders: walls/furniture (0, 1) and collision
#: hulls (3). Group 2 is excluded so other sensors' FOV cones -- and robot *visual* meshes -- do not
#: block a ray; robots still occlude through their group-3 collision geoms.
_OCCLUDER_GROUPS = (0, 1, 3)

#: A ``fov_near`` of 0 would collapse the visibility volume's near cap to a point; draw it this far
#: (m) from the apex instead. Also the minimum near->far thickness kept per ray, so a ray blocked
#: before ``near`` yields a thin sliver rather than a zero-area (uncompilable) cell.
_FOV_EPS = 1e-3

_TWO_PI = 2.0 * math.pi

#: Angular sampling step (deg) of a synthesised lidar-sector shell in both azimuth and elevation. A
#: coarse mesh (5 deg -> a 360deg dome is 72 azimuth facets) is plenty for a translucent coverage cue
#: and keeps the vertex count modest; the drawn shape is a display volume, not a physics surface.
_SECTOR_STEP_DEG = 5.0


def _fov_half_extents(fovy_deg: float, aspect: float, depth: float) -> tuple[float, float]:
    """Half-width/height (hx, hy) of a pinhole camera's image rectangle at optical ``depth``.

    Single source of the FOV angular extent used by :func:`_visibility_grid` (the occlusion ray grid)
    so the grid's boundary coincides with the camera's true image rectangle. ``aspect`` = width/height."""
    t = math.tan(math.radians(fovy_deg) / 2.0)
    return t * aspect * depth, t * depth


def _double_sided(faces: np.ndarray) -> np.ndarray:
    """Each triangle plus its reverse-wound twin, so a translucent FOV volume renders from inside too.

    MuJoCo back-face culls mesh triangles, so an outward-wound FOV shell vanishes the moment the
    viewpoint is inside it -- exactly when you most want to notice you are standing in a sensor's field
    of view. The reversed twin gives every facet a front side from both directions. Doubling the facets
    with opposite winding makes the mesh non-manifold, so callers must set shell inertia (its volume is
    no longer well defined); the FOV geoms are non-colliding, so that inertia is never used."""
    return np.vstack([faces, faces[:, ::-1]])


def _visibility_grid(fovy_deg: float, aspect: float, nu: int, nv: int) -> np.ndarray:
    """Unnormalised optical-frame ray directions ``[u, v, -1]`` on an ``nu`` x ``nv`` grid.

    Camera looks along ``-z``; the grid spans the image rectangle with inclusive endpoints, so its
    corners are the four frustum edges. A point at optical depth ``d`` along a ray is ``d * dir``; the
    grid sheets are planar at ``z = -d`` (``near``/``far`` are plane depths, not radial distances).
    Row-major: v (row ``j``) outer, u (column ``i``) inner."""
    hx, hy = _fov_half_extents(fovy_deg, aspect, 1.0)
    u = np.linspace(-hx, hx, nu)
    v = np.linspace(-hy, hy, nv)
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu, vv, -np.ones_like(uu)], axis=-1).reshape(-1, 3)


def _visibility_mesh(
    raw_dirs: np.ndarray, depths: np.ndarray, near: float, nu: int, nv: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices + explicit triangle faces of an occlusion-clipped FOV volume, optical frame.

    Two grid sheets -- a near cap at ``near_eff = max(near, _FOV_EPS)`` and a far sheet at each ray's
    (clamped) optical depth -- joined by side-wall quads along the four boundary edges. Depths are
    floored to ``near_eff + _FOV_EPS`` so every cell keeps positive thickness (a ray blocked before
    ``near``, or all rays blocked, yields a thin plate rather than a zero-volume mesh that MuJoCo
    refuses to compile). Outward-consistent CCW winding; the caller adds reverse-wound twins via
    :func:`_double_sided` so the volume renders from inside as well as outside."""
    near_eff = max(near, _FOV_EPS)
    depths = np.clip(depths, near_eff + _FOV_EPS, None)
    verts = np.vstack([near_eff * raw_dirs, depths[:, None] * raw_dirs])
    o = nu * nv  # far-sheet vertex offset
    faces: list[list[int]] = []
    for j in range(nv - 1):
        for i in range(nu - 1):
            a, b = j * nu + i, j * nu + i + 1
            c, d = (j + 1) * nu + i + 1, (j + 1) * nu + i
            faces += [[a, b, c], [a, c, d]]  # near cap: outward toward the camera (+z)
            faces += [[o + a, o + d, o + c], [o + a, o + c, o + b]]  # far sheet: outward (-z)

    def band(n_a: int, n_b: int, flip: bool) -> None:
        f_a, f_b = o + n_a, o + n_b
        if flip:
            faces.extend([[n_a, f_b, f_a], [n_a, n_b, f_b]])
        else:
            faces.extend([[n_a, f_a, f_b], [n_a, f_b, n_b]])

    for i in range(nu - 1):
        band(i, i + 1, flip=False)  # bottom edge (j=0)
        band((nv - 1) * nu + i, (nv - 1) * nu + i + 1, flip=True)  # top edge
    for j in range(nv - 1):
        band(j * nu, (j + 1) * nu, flip=True)  # left edge (i=0)
        band(j * nu + nu - 1, (j + 1) * nu + nu - 1, flip=False)  # right edge
    return verts, np.asarray(faces, dtype=np.int32)


def _sector_grid(h_min: float, h_max: float, v_min: float, v_max: float) -> tuple[int, int]:
    """Azimuth/elevation sample counts (na, ne) for a lidar sector at :data:`_SECTOR_STEP_DEG`.

    Endpoints are inclusive in elevation (a band) and, for a bounded azimuth, in azimuth too; a
    wrapping 360deg azimuth drops the duplicate seam sample (see :func:`_lidar_sector_dirs`)."""
    na = max(8, round(math.degrees(h_max - h_min) / _SECTOR_STEP_DEG))
    ne = max(3, round(math.degrees(v_max - v_min) / _SECTOR_STEP_DEG) + 1)
    return na, ne


def _lidar_sector_dirs(
    h_min: float, h_max: float, v_min: float, v_max: float, na: int, ne: int, wraps: bool
) -> np.ndarray:
    """Unit ray directions on an ``ne`` x ``na`` (elevation x azimuth) grid, sensor frame.

    Mirrors ``livox_mid360._build_directions`` exactly (``dir = [cos(el)cos(az), cos(el)sin(az),
    sin(el)]``; forward ``+x``, azimuth about ``+z``, elevation off the xy-plane) so the drawn shell
    coincides with where the capture plugin casts. A wrapping 360deg azimuth samples ``[h_min, h_max)``
    (last sample one step short of the seam, no duplicate); a bounded band uses inclusive endpoints.
    Row-major: elevation (row ``j``) outer, azimuth (column ``i``) inner."""
    if wraps:
        az = h_min + np.arange(na) * ((h_max - h_min) / na)
    else:
        az = np.linspace(h_min, h_max, na)
    el = np.linspace(v_min, v_max, ne)
    EL, AZ = np.meshgrid(el, az, indexing="ij")  # (ne, na)
    cos_el = np.cos(EL)
    dirs = np.stack([cos_el * np.cos(AZ), cos_el * np.sin(AZ), np.sin(EL)], axis=-1)
    return dirs.reshape(-1, 3)


def _lidar_sector_mesh(
    h_min: float,
    h_max: float,
    v_min: float,
    v_max: float,
    near: float,
    far,
    na: int,
    ne: int,
    wraps: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices + explicit triangle faces of a lidar's angular-sector coverage volume, sensor frame.

    The closed solid between radius ``near`` and ``far`` within the azimuth band ``[h_min, h_max]``
    and elevation band ``[v_min, v_max]``: an inner shell (at ``max(near, _FOV_EPS)``) and an outer
    shell (at ``far``) on the :func:`_lidar_sector_dirs` grid, joined by elevation end caps and, for a
    bounded azimuth, two azimuth side caps. ``far`` is either one radius or, when the sector is
    clipped against the world, a per-ray array of them in the grid's row-major order -- floored like
    :func:`_visibility_mesh` does so a fully blocked ray leaves a thin cell rather than a
    zero-volume one MuJoCo would refuse to compile. A wrapping 360deg dome closes on itself (cells connect the
    last azimuth column back to the first, no seam) and has no side caps. Faces are wound
    outward-consistent (the caller adds reverse-wound twins via :func:`_double_sided` so the solid also
    renders from inside); explicit faces because the sector is non-convex, so the convex-hull path (a
    camera frustum) would fill in the dome."""
    dirs = _lidar_sector_dirs(h_min, h_max, v_min, v_max, na, ne, wraps)  # (ne*na, 3)
    near_eff = max(near, _FOV_EPS)
    outer = np.clip(
        np.broadcast_to(np.asarray(far, dtype=np.float64), (dirs.shape[0],)),
        near_eff + _FOV_EPS,
        None,
    )
    verts = np.vstack([near_eff * dirs, outer[:, None] * dirs])  # inner sheet, then outer sheet
    o = ne * na  # outer-sheet vertex offset
    ncol = na if wraps else na - 1  # azimuth cells (wrap closes the ring)
    faces: list[list[int]] = []

    def vid(j: int, i: int) -> int:
        return j * na + (i % na)

    for j in range(ne - 1):
        for i in range(ncol):
            a, b = vid(j, i), vid(j, i + 1)
            c, d = vid(j + 1, i + 1), vid(j + 1, i)
            faces += [
                [o + a, o + b, o + c],
                [o + a, o + c, o + d],
            ]  # outer shell: outward (+radial)
            faces += [[a, d, c], [a, c, b]]  # inner shell: outward (-radial)

    def band(n_a: int, n_b: int, flip: bool) -> None:
        """A near->far quad along the (n_a, n_b) boundary edge; ``flip`` reverses the winding."""
        f_a, f_b = o + n_a, o + n_b
        if flip:
            faces.extend([[n_a, f_b, f_a], [n_a, n_b, f_b]])
        else:
            faces.extend([[n_a, f_a, f_b], [n_a, f_b, n_b]])

    top = ne - 1
    for i in range(ncol):
        band(vid(0, i), vid(0, i + 1), flip=True)  # bottom edge (el = v_min), outward -el
        band(vid(top, i), vid(top, i + 1), flip=False)  # top edge (el = v_max), outward +el
    if not wraps:
        for j in range(ne - 1):
            band(vid(j, 0), vid(j + 1, 0), flip=False)  # az = h_min side cap
            band(vid(j, na - 1), vid(j + 1, na - 1), flip=True)  # az = h_max side cap
    return verts, np.asarray(faces, dtype=np.int32)


def _compile_world_snapshot(world_spec, *, plugin: str, model: str):
    """Compile a throwaway copy of the world built so far, for a build-time occlusion raycast.

    Compiling the *copy* leaves the real spec editable for the rest of the build (the engine compiles
    it once, later). Fails loudly: the raycast can only see what the plugins listed before this one
    produced, and that partial world must compile on its own."""
    try:
        wm = world_spec.copy().compile()
    except Exception as exc:
        raise RuntimeError(
            f"{plugin}: the FOV occlusion raycast uses the world built so far, but compiling that partial "
            f"world failed while placing model {model!r}: {exc}. The plugins listed before this "
            f"spawn_sensor must produce a compilable spec on their own (e.g. list scene/floorplan "
            f"plugins first)."
        ) from exc
    wd = mujoco.MjData(wm)
    mujoco.mj_forward(wm, wd)
    return wm, wd


def _raycast_depths(
    wm, wd, origin_w: np.ndarray, unit_dirs_w: np.ndarray, cutoff: float
) -> np.ndarray:
    """Euclidean hit distance along each world-frame ray (-1 = miss). Same seam as lidar/coverage.

    Occluders are :data:`_OCCLUDER_GROUPS`; ``flg_static=1`` so walls occlude; ``bodyexclude=-1`` (the
    sensor's own housing is not in the snapshot -- it is attached after this build step)."""
    geomgroup = np.zeros(6, dtype=np.uint8)
    geomgroup[list(_OCCLUDER_GROUPS)] = 1
    return raycast.cast(
        wm,
        wd,
        origin_w,
        unit_dirs_w,
        cutoff=cutoff,
        geomgroup=geomgroup,
        flg_static=True,
    ).dist


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


#: What a calibration measures, and the frame it measured them in. All six or none: a focal length in
#: pixels means nothing without its resolution, and two plausible ones are always in reach -- the
#: model's own ``resolution`` and the size the capture plugin renders at -- so neither is guessed.
_LENS_KEYS = ("fx", "fy", "cx", "cy", "width", "height")

#: Plus which camera they describe, for a model that carries more than one.
_INTRINSICS_KEYS = _LENS_KEYS + ("camera",)

#: Metres per pixel for a synthesised ``sensorsize``, used only when the model states none of its own.
#: The value is arbitrary and cannot affect anything: every consumer -- MuJoCo's projection and
#: :func:`~roqsim_sensors.plugins.camera_common.intrinsics_from_model` alike -- uses the focal and
#: principal lengths only as a RATIO to it. 1 um/px merely keeps a 1920-wide sensor a legible 1.92 mm.
_PIXEL_PITCH_M = 1e-6


def _intrinsics_errors(intr) -> list[str]:
    """Everything wrong with an ``intrinsics:`` block, as validation strings.

    Separate from the plugin so it reads as one closed vocabulary, and strict about unknown keys: a
    misspelled ``cy`` would otherwise render a centred principal point while the world says otherwise,
    and that is indistinguishable from an uncalibrated run in every artifact it produces.
    """
    if intr is None:
        return []
    if not isinstance(intr, dict):
        return ["'intrinsics' must be a mapping of fx/fy/cx/cy (plus width/height/camera)"]
    errors = []
    unknown = sorted(set(intr) - set(_INTRINSICS_KEYS))
    if unknown:
        errors.append(
            f"'intrinsics' has unknown key(s) {unknown}; it takes {list(_INTRINSICS_KEYS)}"
        )
    missing = [k for k in _LENS_KEYS if intr.get(k) is None]
    if missing:
        errors.append(
            f"'intrinsics' needs all of {list(_LENS_KEYS)} -- missing {missing}. A partial lens mixes "
            "a measured number with an assumed one and says nothing about which is which, and the "
            "resolution is part of the measurement rather than a default."
        )
    for key in _LENS_KEYS:
        if intr.get(key) is None:
            continue
        try:
            value = float(intr[key])
        except (TypeError, ValueError):
            errors.append(f"'intrinsics.{key}' must be a number, got {intr[key]!r}")
            continue
        if key in ("fx", "fy") and value <= 0:
            errors.append(f"'intrinsics.{key}' must be > 0")
        if key in ("width", "height") and value <= 1:
            errors.append(f"'intrinsics.{key}' must be > 1 (it is a pixel count)")
    return errors


class SpawnSensorPlugin(Plugin):
    #: Registers an entity, so its label names that entity and it may own a
    #: ``components:`` block of sensors, controllers and monitors that attach to it.
    provides_entity = True

    @classmethod
    def expand(cls, spec, world, base_dir):
        """Inject the sensor model's default capture plugin (its ``<model>.manifest.yaml``).

        Keeps the capture plugin out of the world YAML: it ships with the model and is wired to
        this mount via ``robot: <name>`` -- the same config key ``lidar``/``oakd_camera`` already
        use for a robot's own prefix/namespace, so a capture plugin needs no mount-specific config.
        """
        return expand_manifest(spec, world, base_dir=base_dir)

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.sensor_name = self.address
        self.prefix = self.config.get("prefix", "")
        pos = self.config.get("pos", [0.0, 0.0, 0.0])
        self.pos = [float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)]
        rpy = self.config.get("rpy", [0.0, 0.0, 0.0])
        self.quat = _rpy_to_quat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        # This UNIT's measured lens, written onto the model's camera at build time so MuJoCo renders
        # through it (see :meth:`_apply_intrinsics`). Empty is the historical path: the model's own
        # fovy, an ideal pinhole, a centred principal point.
        self._intrinsics = dict(self.config.get("intrinsics") or {})
        self.show_fov = bool(self.config.get("show_fov", False))
        self.fov_alpha = float(self.config.get("fov_alpha", 0.25))
        # Valid detection band (m) of the synthesised camera FOV frustum: ``fov_near`` is the near
        # plane, ``fov_range`` the far plane. Both DEFAULT to the sensor model's own manifest ``fov:``
        # block (device knowledge lives with the device, not each world); a world-YAML value overrides
        # per placement. ``fov_near > 0`` cuts off the apex so the drawn cone starts where the sensor
        # becomes valid. Resolved against the manifest at build time (see :meth:`_resolve_fov_range`).
        self._fov_near_cfg = self.config.get("fov_near")
        self._fov_far_cfg = self.config.get("fov_range")
        # A synthesised camera frustum is always clipped against the world at build time so it stops at
        # walls/objects (a visibility volume); ``fov_rays`` is that clip's ray grid [horizontal, vertical].
        rays = self.config.get("fov_rays", [32, 24])
        self.fov_rays = (int(rays[0]), int(rays[1]))

    def validate_config(self, config: dict) -> list[str]:
        errors = []
        if not config.get("model"):
            errors.append("'model' is required")
        else:
            try:
                resolve_model(config["model"])
            except ModelError as exc:
                errors.append(str(exc))
        if "rpy" in config and len(config["rpy"]) != 3:
            errors.append("'rpy' must be [roll, pitch, yaw] in radians")
        if len(config.get("pos", [0, 0, 0])) not in (2, 3):
            errors.append("'pos' must be [x, y] or [x, y, z]")
        if not 0.0 <= float(config.get("fov_alpha", 0.25)) <= 1.0:
            errors.append("'fov_alpha' must be in [0, 1]")
        # Only explicit world values are checked here (the manifest default is validated at build time,
        # in _resolve_fov_range, where the model is resolved).
        near = config.get("fov_near")
        far = config.get("fov_range")
        if near is not None and float(near) < 0.0:
            errors.append("'fov_near' must be >= 0")
        if near is not None and far is not None and float(near) >= float(far):
            errors.append(
                f"'fov_near' ({near}) must be < 'fov_range' ({far}); a near plane at or "
                "beyond the far plane leaves no volume to draw"
            )
        rays = config.get("fov_rays", [32, 24])
        if len(rays) != 2 or any(int(r) < 2 or int(r) > 256 for r in rays):
            errors.append("'fov_rays' must be [nu, nv] with each in 2..256")
        errors.extend(_intrinsics_errors(config.get("intrinsics")))
        return errors

    def _resolve_fov_range(self, asset) -> tuple[float, float]:
        """Effective (near, far) for the synthesised frustum: world config over the model's manifest.

        The sensor model owns its valid range via a ``fov: {near, far}`` block in its
        ``<model>.manifest.yaml``; a world may override either per placement. Falls back to
        ``near=0`` (apex pyramid) / ``far=2.0`` when neither declares one. Fails loudly on an
        empty band so a mis-authored manifest can't silently draw nothing."""
        meta = manifest_fov(asset.path)
        near = self._fov_near_cfg if self._fov_near_cfg is not None else meta.get("near", 0.0)
        far = self._fov_far_cfg if self._fov_far_cfg is not None else meta.get("far", 2.0)
        near, far = float(near), float(far)
        if near < 0.0 or near >= far:
            raise RuntimeError(
                f"spawn_sensor: invalid FOV range for model {self.config['model']!r}: "
                f"near={near} far={far} (need 0 <= near < far). Check the world config or the "
                f"model manifest 'fov:' block."
            )
        return near, far

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        asset = resolve_model(self.config["model"])
        child = mujoco.MjSpec.from_file(str(asset.path))
        # Resolve mesh/texture refs to absolute paths across the model's asset dirs (own package plus
        # any borrowed via the manifest's `assets:`), so compilation does not depend on CWD.
        apply_assets(child, asset)
        # BEFORE _show_fov, which reads the camera back: a frustum drawn from the model's nominal
        # fovy while the render uses a measured one would be a picture of the wrong lens.
        if self._intrinsics:
            self._apply_intrinsics(child)
        if self.show_fov:
            near, far = self._resolve_fov_range(asset)
            # A synthesised camera frustum is always clipped against the world built so far, so pass
            # the world spec every time; camera-less paths (bundled envelope, lidar sector) ignore it.
            self._show_fov(child, asset, near, far, world_spec=spec)
        frame = spec.worldbody.add_frame()
        frame.pos = self.pos
        frame.quat = self.quat
        spec.attach(child, prefix=self.prefix, frame=frame)

    def _apply_intrinsics(self, child: mujoco.MjSpec) -> None:
        """Write this placement's measured lens onto the model's camera, in pixels.

        MuJoCo renders through ``sensorsize`` + ``focalpixel`` + ``principalpixel`` whenever a sensor
        size is set, so the calibration goes HERE rather than into the capture plugin's config: the
        plugin then reads the very numbers the renderer used back out of the compiled model
        (:func:`~roqsim_sensors.plugins.camera_common.intrinsics_from_model`, path 1), and pixels and
        ``camera_info`` cannot drift apart.
        """
        cam = self._lens_camera(child)
        width, height = self._lens_resolution()
        fx, fy = float(self._intrinsics["fx"]), float(self._intrinsics["fy"])
        cx, cy = float(self._intrinsics["cx"]), float(self._intrinsics["cy"])
        cam.resolution = [width, height]
        # The sensor's physical size is a free scale -- everything downstream uses the focal and
        # principal lengths only as a ratio to it -- so it has to be non-zero (0 is MuJoCo's "unset",
        # which selects the fovy path) and square-pixelled, and nothing more. A model that states its
        # own keeps it: that one is a fact about the device.
        sw, sh = (float(v) for v in cam.sensor_size)
        if not (sw > 0.0 and sh > 0.0):
            cam.sensor_size = [width * _PIXEL_PITCH_M, height * _PIXEL_PITCH_M]
        cam.focal_pixel = [fx, fy]
        # MJCF's principal point is an OFFSET from the image centre with +y pointing UP the image,
        # while a calibration's cy counts DOWN from the top row. This flip is the one sign in the
        # file that a reader cannot check by eye, so a test pins it against the reader that undoes it.
        cam.principal_pixel = [cx - width / 2.0, height / 2.0 - cy]

    def _lens_camera(self, child: mujoco.MjSpec):
        """The camera ``intrinsics:`` describes: the named one, or the only one there is."""
        cameras = list(child.cameras)
        wanted = self._intrinsics.get("camera")
        if wanted:
            for cam in cameras:
                if cam.name == wanted:
                    return cam
            raise RuntimeError(
                f"spawn_sensor: 'intrinsics.camera' is {wanted!r}, which model "
                f"{self.config['model']!r} does not have. It has: {[c.name for c in cameras]}"
            )
        if len(cameras) == 1:
            return cameras[0]
        if not cameras:
            raise RuntimeError(
                f"spawn_sensor: 'intrinsics' states a lens but model {self.config['model']!r} has no "
                f"camera to give it to."
            )
        raise RuntimeError(
            f"spawn_sensor: model {self.config['model']!r} has {len(cameras)} cameras "
            f"({[c.name for c in cameras]}), which do not share a lens -- name the one this "
            f"calibration measured with 'intrinsics.camera'."
        )

    def _lens_resolution(self) -> tuple[int, int]:
        """The resolution the stated lens was measured at -- required, never inferred.

        Two other resolutions are always within reach: the model camera's own ``resolution``, and the
        size the capture plugin renders at. Either would be a plausible guess, and a wrong one scales
        every number in the block by the ratio between two frames, silently. So the block carries its
        own, and :func:`_intrinsics_errors` refuses one without it.
        """
        return int(self._intrinsics["width"]), int(self._intrinsics["height"])

    def _show_fov(self, child: mujoco.MjSpec, asset, near: float, far: float, world_spec) -> None:
        """Make the sensor's field of view visible before attach.

        A model with cameras (the RealSense/Zivid mounts) always draws a synthesised **frustum** per
        camera, clipped against ``world_spec`` (the world built so far) into a *visibility volume* that
        stops at walls and objects -- occlusion is unconditional, not a per-placement opt-in.

        A camera-less model falls back to revealing its bundled ``_fov`` envelope (if it ships one) or,
        for a lidar that does not (Mid-360, Robin W1G), synthesising the azimuth x elevation **sector**
        from the manifest's angular ``fov:`` band -- and that sector is clipped against ``world_spec``
        too, casting its own direction grid from the scan site. Only the bundled envelope draws
        un-clipped, being a baked mesh; ``near``/``far`` still set the radii the clip works within.

        Raises only when there is nothing at all to show, so ``show_fov: true`` never silently does
        nothing."""
        if self._add_camera_frustums(child, asset, near, far, world_spec=world_spec) > 0:
            return
        revealed = self._reveal_fov(child) or self._add_lidar_sectors(
            child, asset, near, far, world_spec=world_spec
        )
        if revealed == 0:
            raise RuntimeError(
                f"spawn_sensor: show_fov is set but model {self.config['model']!r} ships no FOV geom "
                f"(none ending {FOV_GEOM_SUFFIX!r}), has no camera to synthesise a frustum from, and "
                f"declares no angular 'fov:' band in its manifest to draw a lidar sector"
            )

    def _reveal_fov(self, child: mujoco.MjSpec) -> int:
        """Set the alpha of the model's hidden ``_fov`` geoms to ``fov_alpha``. Returns how many."""
        revealed = 0
        for geom in child.geoms:
            if geom.name.endswith(FOV_GEOM_SUFFIX):
                rgba = list(geom.rgba)
                rgba[3] = self.fov_alpha
                geom.rgba = rgba
                revealed += 1
        return revealed

    def _add_camera_frustums(
        self, child: mujoco.MjSpec, asset, near: float, far: float, *, world_spec
    ) -> int:
        """Synthesise a translucent, occlusion-clipped FOV mesh at each camera. Returns how many.

        The camera's orientation is stored as an alternative (``xyaxes``) form, so we read its resolved
        body-frame pose (``cam_pos``/``cam_quat``) off a throwaway compile of the model -- the same
        pattern the lidar plugins use to derive a static mount transform -- and place the vertices
        (built in the camera's -z-looking optical frame) into the mount body's frame.

        A ray grid (:func:`_visibility_grid`) is cast from the camera against a snapshot of ``world_spec``
        (the world built so far) and each ray clamped at its hit, so the drawn mesh is a *visibility
        volume* that stops at walls and objects (:func:`_visibility_mesh`)."""
        cameras = list(child.cameras)
        if not cameras:
            return 0
        # A copy of CHILD rather than a re-read of the model file: a placement may have written its
        # own lens onto the camera (:meth:`_apply_intrinsics`), and MuJoCo derives ``cam_fovy`` from
        # a focal length, so copying is also what makes the drawn cone follow the measured optics.
        probe = child.copy()
        pm = probe.compile()
        pd = mujoco.MjData(pm)
        mujoco.mj_forward(pm, pd)  # populate cam_xpos/cam_xmat (child-root frame)
        wm, wd = _compile_world_snapshot(
            world_spec, plugin=self.name or "spawn_sensor", model=self.config["model"]
        )
        r_mount = np.zeros(9)  # world <- child-root: the attach frame's rotation
        mujoco.mju_quat2Mat(r_mount, np.asarray(self.quat, dtype=np.float64))
        r_mount = r_mount.reshape(3, 3)
        mount_pos = np.asarray(self.pos, dtype=np.float64)
        added = 0
        for cam in cameras:
            cid = mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_CAMERA, cam.name)
            if cid < 0:
                continue
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, np.asarray(pm.cam_quat[cid], dtype=np.float64))
            rot = rot.reshape(3, 3)
            pos = np.asarray(pm.cam_pos[cid], dtype=np.float64)
            w, h = (int(v) for v in pm.cam_resolution[cid])
            aspect = (w / h) if (w > 1 and h > 1) else 1.0
            fovy = float(pm.cam_fovy[cid])
            nu, nv = self.fov_rays
            raw = _visibility_grid(fovy, aspect, nu, nv)  # (nv*nu, 3) optical-frame dirs
            norms = np.linalg.norm(raw, axis=1)
            origin_w = r_mount @ pd.cam_xpos[cid] + mount_pos
            r_cam_w = r_mount @ pd.cam_xmat[cid].reshape(3, 3)  # world <- optical
            unit_dirs_w = (raw / norms[:, None]) @ r_cam_w.T
            dist = _raycast_depths(wm, wd, origin_w, unit_dirs_w, cutoff=far * float(norms.max()))
            # Euclidean ray distance -> optical-axis (plane) depth; miss (-1) -> far. cutoff is a
            # culling hint, not a clamp, so clamp to far manually (see lidar.py).
            depth = np.where(dist >= 0.0, dist / norms, np.inf)
            depth = np.minimum(depth, far)
            verts_cam, faces = _visibility_mesh(raw, depth, near, nu, nv)
            verts_body = verts_cam @ rot.T + pos  # optical frame -> mount body frame
            mesh = child.add_mesh()
            mesh.name = f"{cam.name}{FOV_GEOM_SUFFIX}"
            mesh.uservert = verts_body.reshape(-1).tolist()
            mesh.userface = _double_sided(faces).reshape(-1).tolist()
            # Double-sided (non-manifold) faces have no well-defined volume, and the clipped visibility
            # volume can be arbitrarily thin (all rays blocked near the sensor); shell inertia lets both
            # compile ("mesh volume is too small" / degenerate volume otherwise). The geom is non-colliding.
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            geom = cam.parent.add_geom()
            geom.name = f"{cam.name}{FOV_GEOM_SUFFIX}"
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = mesh.name
            geom.contype = 0
            geom.conaffinity = 0
            geom.group = FOV_GEOM_GROUP
            geom.rgba = [*_FRUSTUM_RGB, self.fov_alpha]
            added += 1
        return added

    def _add_lidar_sectors(
        self, child: mujoco.MjSpec, asset, near: float, far: float, *, world_spec
    ) -> int:
        """Synthesise a translucent angular-sector FOV volume for a lidar. Returns how many (0 or 1).

        A lidar's field of view is an azimuth x elevation band between ``range_min`` and ``range_max``,
        not a pinhole frustum -- so, the same way :meth:`_add_camera_frustums` synthesises a cone from
        a model's ``<camera>`` intrinsics, this synthesises the sector from the model manifest's
        angular ``fov:`` block (``h_min``/``h_max``/``v_min``/``v_max``, radians; a >= 2*pi azimuth
        span is a full 360deg dome). The shell is built in the scan site's frame with the same
        direction convention the capture plugin casts with, so the drawn volume coincides with the
        rays. ``near``/``far`` are the shell's inner/outer radii (the manifest's ``fov: {near, far}``).

        Clipped against ``world_spec`` exactly as a camera frustum is: the sector's own direction grid
        is cast from the scan site and each ray clamped at its first hit, so the drawn volume stops at
        walls instead of passing through them. A lidar was long exempted here for having "no pinhole to
        raycast from", but :func:`_lidar_sector_dirs` *is* an origin plus a direction grid -- the same
        two things :meth:`_add_camera_frustums` casts with. Un-clipped, a long-range lidar drew its
        full physical reach through the building: the Robin W1G's 200 m cone and the Mid-360's 40 m
        dome bounded an otherwise 10 m room, and MuJoCo's model-derived default camera framed *that*,
        so every render of such a world came out as a few dark pixels in an empty frame.

        Returns 0 when the manifest declares no angular band (the model is not a lidar), so the caller
        falls through to the 'nothing to show' error rather than this silently doing nothing."""
        meta = manifest_fov(asset.path)
        if "h_min" not in meta:
            return 0
        h_min, h_max = float(meta["h_min"]), float(meta["h_max"])
        v_min, v_max = float(meta["v_min"]), float(meta["v_max"])
        wraps = (h_max - h_min) >= _TWO_PI - 1e-9
        na, ne = _sector_grid(h_min, h_max, v_min, v_max)

        site = self._lidar_site(child)
        radii = self._clipped_sector_radii(
            child,
            asset,
            site,
            _lidar_sector_dirs(h_min, h_max, v_min, v_max, na, ne, wraps),
            far,
            world_spec=world_spec,
        )
        verts_site, faces = _lidar_sector_mesh(
            h_min, h_max, v_min, v_max, near, radii, na, ne, wraps
        )

        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(site.quat, dtype=np.float64))
        verts_body = verts_site @ rot.reshape(3, 3).T + np.asarray(site.pos, dtype=np.float64)

        mesh = child.add_mesh()
        mesh.name = f"{site.name}{FOV_GEOM_SUFFIX}"
        mesh.uservert = verts_body.reshape(-1).tolist()
        mesh.userface = _double_sided(faces).reshape(-1).tolist()
        # Double-sided faces (visible from inside the dome too) are non-manifold, and a thin shell
        # (near ~= far, or a degenerate band) would trip "mesh volume is too small"; shell inertia
        # sidesteps both, and the geom is non-colliding so its inertia is never used.
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        geom = site.parent.add_geom()
        geom.name = f"{site.name}{FOV_GEOM_SUFFIX}"
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh.name
        geom.contype = 0
        geom.conaffinity = 0
        geom.group = FOV_GEOM_GROUP
        geom.rgba = [*_FRUSTUM_RGB, self.fov_alpha]
        return 1

    def _clipped_sector_radii(
        self, child, asset, site, dirs_site: np.ndarray, far: float, *, world_spec
    ):
        """Per-ray outer radii for the sector: each of ``dirs_site`` clamped at its first world hit.

        The site's pose is read off a throwaway compile of the model rather than its spec ``pos``/
        ``quat``, for the reason :meth:`_add_camera_frustums` does the same for a camera: those are
        stated relative to the site's parent body, and a mount is free to nest one. ``site_xpos``/
        ``site_xmat`` are already resolved into the child-root frame, which the attach transform then
        carries to world."""
        probe = mujoco.MjSpec.from_file(str(asset.path))
        apply_assets(probe, asset)
        pm = probe.compile()
        pd = mujoco.MjData(pm)
        mujoco.mj_forward(pm, pd)  # populate site_xpos/site_xmat (child-root frame)
        sid = mujoco.mj_name2id(pm, mujoco.mjtObj.mjOBJ_SITE, site.name)
        if (
            sid < 0
        ):  # the site exists in the spec but not the compiled probe -- nothing to cast from
            return far
        wm, wd = _compile_world_snapshot(
            world_spec, plugin=self.name or "spawn_sensor", model=self.config["model"]
        )
        r_mount = np.zeros(9)  # world <- child-root: the attach frame's rotation
        mujoco.mju_quat2Mat(r_mount, np.asarray(self.quat, dtype=np.float64))
        r_mount = r_mount.reshape(3, 3)
        origin_w = r_mount @ pd.site_xpos[sid] + np.asarray(self.pos, dtype=np.float64)
        r_site_w = r_mount @ pd.site_xmat[sid].reshape(3, 3)  # world <- site
        dist = _raycast_depths(wm, wd, origin_w, dirs_site @ r_site_w.T, cutoff=far)
        # cutoff is a culling hint rather than a clamp, and a miss is -1; both mean "no closer than
        # far" (see the identical handling in _add_camera_frustums).
        return np.minimum(np.where(dist >= 0.0, dist, np.inf), far)

    def _lidar_site(self, child: mujoco.MjSpec):
        """The scan site the sector is centred on: the model's sole site (the ray origin).

        A standalone lidar mount carries exactly one site -- the laser's optical origin the capture
        plugin casts from. Fails loudly on none/several rather than guessing which is the scan site."""
        sites = list(child.sites)
        if len(sites) != 1:
            raise RuntimeError(
                f"spawn_sensor: lidar FOV synthesis for model {self.config['model']!r} expects the "
                f"mount to carry exactly one site (the scan origin), found {len(sites)}"
            )
        return sites[0]

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.sensor_name,
                kind="sensor",
                body=self.prefix + "mount",
                meta={
                    "prefix": self.prefix,
                    "model": self.config["model"],
                    # Inherited by the mount's capture plugin (manifest-injected or explicit), so
                    # it needs no namespace plumbing of its own.
                    "namespace": self.config.get("namespace", ""),
                },
            )
        )
