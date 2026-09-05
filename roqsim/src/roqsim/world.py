"""Built-in *world definitions*: the static ground + lighting a fixed cell sits in.

A world is the fixed environment the robots/props are placed into -- distinct from the robot-family
scene plugins. It is chosen with the ``sim.world`` key in the world YAML::

    sim:
      world: empty_room      # a named built-in world definition

When ``sim.world`` is unset the engine builds :data:`DEFAULT_WORLD` (``empty_room``): a checker
ground plane named ``floor``, a ceiling light, a gradient sky and four perimeter walls -- a bounded
room lit to daylight brightness, light enough that a robot in it reads without a lamp of its own.

``sim.world`` may instead be a **path to an MJCF file** (see :func:`world_file`), loaded as the base
scene -- e.g. a baked scene like ``depot/depot.xml`` -- or a **package-qualified reference**
``<package>:<world>`` (e.g. ``roqsim_scenes:depot``), resolved against a registered
``roqsim.worlds`` provider (mirroring how ``model:`` refs resolve via ``roqsim.models``; see
:func:`world_file`). Anything that is neither the built-in ``empty_room`` nor a resolvable
file/package ref is a fail-fast error.

A world provider is a package whose ``pyproject.toml`` registers::

    [project.entry-points."roqsim.worlds"]
    my_pkg = "my_pkg"          # a module exposing WORLDS_DIR

and whose module exposes ``WORLDS_DIR`` (a dir of ``<world>/<world>.xml`` baked scenes, or flat
``<world>.xml`` files).

A scene plugin that builds its own ground + lighting (it sets ``Plugin.provides_world = True``, e.g.
the mobile ``floorplan``, which loads a floorplan mesh + walls) *overrides* this: when such a plugin is
present the engine skips the world definition. Setting both ``sim.world`` and such a plugin is allowed
but the engine warns and lets the plugin win. Keeping the ground+light here (not baked into the engine)
means fixed-cell packages never need to depend on the mobile ``floorplan``.

Add a world definition by registering a builder in :data:`_WORLDS`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module, metadata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco

#: World definition used when ``sim.world`` is not set.
DEFAULT_WORLD = "empty_room"

#: Entry-point group for ``<package>:<world>`` providers (mirrors ``roqsim.models``).
WORLDS_ENTRY_POINT_GROUP = "roqsim.worlds"

#: Half-extent (m) of the default ground plane; a generous square that covers a tabletop-scale cell.
_DEFAULT_GROUND_HALF_EXTENT = 5.0

#: Height (m) of the room's overhead light -- above the walls, and high enough that its shadow cone
#: still covers their top edge (see :func:`_add_ground_and_light`).
_LIGHT_HEIGHT = 5.0

#: The room's palette: the two greys of the checker floor, and the sky gradient over it (horizon
#: first). Public because a preview that stands a model on its own ground (a thumbnail) is meant to be
#: the same picture as that model standing in this room.
FLOOR_RGB1 = [0.70, 0.71, 0.72]
FLOOR_RGB2 = [0.60, 0.61, 0.63]
SKY_RGB1 = [0.62, 0.74, 0.9]
SKY_RGB2 = [0.95, 0.96, 0.98]


def _add_ground_and_light(spec: mujoco.MjSpec, size: float) -> None:
    """Add a checker ground plane named ``floor``, a bright sky, and a top-down ceiling light."""
    import mujoco

    # A gradient skybox, so what is above the walls is daylight rather than the black void a
    # skybox-less model renders. It is backdrop only: MuJoCo's skybox lights nothing, so the room's
    # brightness comes from the light and the global ambient below.
    sky = spec.add_texture()
    sky.name = "sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.width = sky.height = 512
    sky.rgb1 = SKY_RGB1
    sky.rgb2 = SKY_RGB2

    # A light floor in a light room: a dark ground absorbs the light and reads as dim however hard
    # it is lit, so the checker is a pale grey pair that keeps the scale cue without the darkness.
    tex = spec.add_texture()
    tex.name = "grid"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.width = tex.height = 512
    tex.rgb1 = FLOOR_RGB1
    tex.rgb2 = FLOOR_RGB2

    mat = spec.add_material()
    mat.name = "floor_mat"
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [max(1, int(size * 4)), max(1, int(size * 4))]
    # Not reflective: on a light floor even a little reflectance mirrors the ceiling light into a
    # white hotspot around whatever stands on it.
    mat.reflectance = 0.0

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [size, size, 0.05]
    floor.material = "floor_mat"
    floor.friction = [2.0, 0.005, 0.0001]

    # Daylight, not a single lamp in the dark. MuJoCo's headlight ambient is the only truly global
    # term (a light's own ambient is added per light, so it scales with how many a scene has), and it
    # is what keeps a surface facing away from the ceiling out of the near-black MuJoCo defaults it
    # to. The ceiling light then carries the direction: enough diffuse to model shape and cast a
    # readable contact shadow on top of that fill.
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]

    # A wide, flat overhead spot rather than MuJoCo's default cone: the default 45-degree cutoff with
    # exponent 10 burns a bright pool into the middle of the floor and leaves the corners dim, which
    # is the opposite of an evenly daylit room. Straight down, as `roqsim.render` documents -- it
    # keys the model-preview tilt on a light pointing that way.
    #
    # Height and cutoff are set by what the SHADOWS need, not by where a lamp would hang. A spot's
    # shadow map only covers `shadowscale` x its cutoff cone, so geometry outside that cone is lit
    # but casts nothing: with the frustum too tight, a shadow thrown against a wall stops dead partway
    # up it. At full scale, the cone from `_LIGHT_HEIGHT` has to still reach the far top corner of the
    # room -- `(height - wall) * tan(cutoff) >= half-diagonal` -- which is what these numbers satisfy
    # for a room of `_DEFAULT_GROUND_HALF_EXTENT`.
    spec.visual.map.shadowscale = 1.0

    light = spec.worldbody.add_light()
    light.pos = [0, 0, _LIGHT_HEIGHT]
    light.dir = [0, 0, -1]
    light.cutoff = 70.0
    light.exponent = 0.0
    light.ambient = [0.12, 0.12, 0.12]
    light.diffuse = [0.48, 0.48, 0.48]
    light.specular = [0.1, 0.1, 0.1]


#: Wall thickness (m) and height (m). Walls are tall enough to sit above a standing humanoid's lidar
#: (~1.1 m) so the scan actually returns the walls rather than passing over them.
_WALL_THICKNESS = 0.1
_WALL_HEIGHT = 2.0


def _add_walls(spec: mujoco.MjSpec, half: float) -> None:
    """Add four perimeter walls enclosing a ``2*half`` square room, inner faces at ``+-half``."""
    import mujoco

    t = _WALL_THICKNESS / 2.0
    hz = _WALL_HEIGHT / 2.0
    # Each wall is centred just outside +-half so its inner face is on the room boundary; the long
    # half-extent runs the full side plus the thickness so the corners close.
    walls = [
        ("wall_px", [half + t, 0.0, hz], [t, half + 2 * t, hz]),
        ("wall_nx", [-half - t, 0.0, hz], [t, half + 2 * t, hz]),
        ("wall_py", [0.0, half + t, hz], [half + 2 * t, t, hz]),
        ("wall_ny", [0.0, -half - t, hz], [half + 2 * t, t, hz]),
    ]
    for name, pos, size in walls:
        wall = spec.worldbody.add_geom()
        wall.name = name
        wall.type = mujoco.mjtGeom.mjGEOM_BOX
        wall.pos = pos
        wall.size = size
        wall.rgba = [0.76, 0.76, 0.78, 1.0]


def _empty_room(spec: mujoco.MjSpec) -> None:
    """The default world: a daylit ground plane + light, enclosed by four perimeter walls.

    The walls sit at the floor's edge (same half-extent) so the room is fully enclosed, giving robots
    a bounded room and the lidar something to see, so a plugin-less world isn't an infinite empty plane.
    """
    _add_ground_and_light(spec, _DEFAULT_GROUND_HALF_EXTENT)
    _add_walls(spec, _DEFAULT_GROUND_HALF_EXTENT)


#: Registry of world definitions: name -> builder that mutates the MjSpec in the build phase.
#: The only built-in is ``empty_room`` (a walled room); ``sim.world`` may otherwise be an MJCF file
#: path (see :func:`world_file`).
_WORLDS: dict[str, Callable[[mujoco.MjSpec], None]] = {
    "empty_room": _empty_room,
}


def available_worlds() -> list[str]:
    """Names of the registered built-in world definitions."""
    return sorted(_WORLDS)


def _world_entry_points():
    """Entry points for the ``roqsim.worlds`` group across Python versions (see models._entry_points)."""
    eps = metadata.entry_points()
    if hasattr(eps, "select"):  # Python 3.10+
        return list(eps.select(group=WORLDS_ENTRY_POINT_GROUP))
    return list(eps.get(WORLDS_ENTRY_POINT_GROUP, []))  # pragma: no cover - legacy


def _find_world(worlds_dir: Path, world: str) -> Path | None:
    """An MJCF for ``world`` under ``worlds_dir``: ``<world>/<world>.xml`` (baked-scene layout), a flat
    ``<world>.xml``/``.mjcf``, or the sole ``*.xml`` in ``<world>/``."""
    for cand in (
        worlds_dir / world / f"{world}.xml",
        worlds_dir / world / f"{world}.mjcf",
        worlds_dir / f"{world}.xml",
        worlds_dir / f"{world}.mjcf",
    ):
        if cand.is_file():
            return cand
    sub = worlds_dir / world
    if sub.is_dir():
        xmls = sorted(sub.glob("*.xml"))
        if len(xmls) == 1:
            return xmls[0]
    return None


def _find_world_yaml(worlds_dir: Path, world: str) -> Path | None:
    """A world YAML for ``world`` under ``worlds_dir``: flat ``<world>.yaml``/``.yml`` or
    ``<world>/<world>.yaml``/``.yml`` (mirrors :func:`_find_world` for the MJCF side)."""
    for cand in (
        worlds_dir / f"{world}.yaml",
        worlds_dir / f"{world}.yml",
        worlds_dir / world / f"{world}.yaml",
        worlds_dir / world / f"{world}.yml",
    ):
        if cand.is_file():
            return cand
    return None


def _resolve_provider_ref(
    name: str, finder: Callable[[Path, str], Path | None], kind: str
) -> str | None:
    """Resolve a ``<package>:<world>`` ref via a registered ``roqsim.worlds`` provider (or an
    importable module exposing ``WORLDS_DIR``), locating the target with ``finder``. ``None`` if the
    left side names no provider/module; ``FileNotFoundError`` if it does but the target is missing.

    ``finder`` is :func:`_find_world` (baked MJCF, for ``sim.world``) or :func:`_find_world_yaml`
    (a world YAML, for ``extends``); ``kind`` is a word used in the not-found message.
    """
    left, _, world = name.partition(":")
    for ep in _world_entry_points():
        if ep.name == left:
            worlds_dir = Path(ep.load().WORLDS_DIR)
            found = finder(worlds_dir, world)
            if found is None:
                raise FileNotFoundError(
                    f"{kind} {world!r} not found in provider {left!r} ({worlds_dir})"
                )
            return str(found)
    # Fall back to treating the left side as an importable module exposing WORLDS_DIR.
    try:
        module = import_module(left)
    except ImportError:
        return None
    worlds_dir = getattr(module, "WORLDS_DIR", None)
    if worlds_dir is None:
        return None
    found = finder(Path(worlds_dir), world)
    if found is None:
        raise FileNotFoundError(f"{kind} {world!r} not found in module {left!r} ({worlds_dir})")
    return str(found)


def _resolve_world_ref(name: str) -> str | None:
    """Resolve a ``<package>:<world>`` ref to an absolute MJCF path, else ``None``.

    Mirrors :func:`roqsim.models.resolve_model`: a registered ``roqsim.worlds`` provider named by
    the left side, else the left side treated as an importable module exposing ``WORLDS_DIR``.
    """
    return _resolve_provider_ref(name, _find_world, "sim.world")


def resolve_world_yaml_ref(name: str) -> str | None:
    """Resolve a ``<package>:<world>`` ref to an absolute world-YAML path, else ``None``.

    The YAML counterpart of :func:`_resolve_world_ref`, used by ``extends:`` (see
    :func:`roqsim.config.load_config`) so a world can inherit another package's world YAML by
    reference, e.g. ``extends: roqsim_scenes:depot`` -> ``.../worlds/depot.yaml``.
    """
    return _resolve_provider_ref(name, _find_world_yaml, "extends world")


def world_file(name: str | None, base_dir: str | Path) -> str | None:
    """If ``sim.world`` names an **MJCF file** or **package ref** (not a built-in), return its absolute
    path, else ``None``.

    Resolution order (mirrors ``model:`` refs):

    1. A **package-qualified ref** ``<package>:<world>`` (e.g. ``roqsim_scenes:depot``) resolved
       against a registered ``roqsim.worlds`` provider (or an importable module exposing
       ``WORLDS_DIR``). Its baked meshes/textures resolve relative to the returned MJCF's own dir, so
       package-installed scenes just work.
    2. A **file path**: ends in ``.xml``/``.mjcf`` or contains a path separator; resolved absolute or
       relative to ``base_dir`` (the world YAML's dir) -- e.g. ``depot/depot.xml``.

    Raises ``FileNotFoundError`` if it looks like a file/package ref but doesn't exist.
    """
    if not name or name in _WORLDS:
        return None
    # 1) package-qualified "<package>:<world>" (only when it isn't itself an existing path).
    if ":" in name and not Path(name).exists():
        resolved = _resolve_world_ref(name)
        if resolved is not None:
            return resolved
    # 2) file reference.
    if not (name.endswith((".xml", ".mjcf")) or "/" in name or os.sep in name):
        return None
    path = Path(name)
    if not path.is_absolute():
        path = Path(base_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"sim.world file not found: {path}")
    return str(path)


def build_world(spec: mujoco.MjSpec, name: str | None) -> str:
    """Build the named world definition into ``spec`` (``None`` -> :data:`DEFAULT_WORLD`).

    Returns the resolved world name. Raises ``KeyError`` for an unknown name.
    """
    resolved = name or DEFAULT_WORLD
    try:
        builder = _WORLDS[resolved]
    except KeyError:
        raise KeyError(
            f"unknown sim.world {resolved!r}; available: {', '.join(available_worlds())}"
        ) from None
    builder(spec)
    return resolved
