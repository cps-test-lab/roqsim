"""Human-facing visualisations of a coverage result.

* :func:`render_coverage_3d` -- an offscreen MuJoCo render of the world with the sample points drawn as
  spheres coloured by how many sensors see them (red = 0 -> green = many). Reuses the ``mujoco.Renderer``
  idiom from ``roqsim_assets``' ``render-thumbnails``; needs only mujoco + numpy, and works headless
  on whatever backend ``import roqsim`` selected for the machine.
* :func:`render_heatmap_2d` -- a top-down coverage-count heatmap over the floor (matplotlib). Needs the
  optional ``coverage`` extra (``matplotlib``); it raises a clear error if matplotlib is unavailable.

Both are offline analysis helpers, not plugins -- they never mutate the ``MjSpec``.
"""

from __future__ import annotations

import numpy as np

from .engine import CoverageResult

# Discrete colour ramp by coverage count. Two palettes, same shape (index = coverage count, clamped):
#
# * ``coverage`` (default) -- 0 sensors -> red, then warm -> cool green as coverage grows. Answers
#   "is this covered?" (green = safe); the historical ramp the coverage-search workflow expects.
# * ``density`` -- 0 sensors -> lightest, then progressively DARKER as more sensors overlap. Answers
#   "how densely is this covered?" -- overlap reads as darkness (what the user asked to see).
_RAMP = np.array(
    [
        [0.85, 0.15, 0.15, 1.0],  # 0
        [0.95, 0.55, 0.15, 1.0],  # 1
        [0.95, 0.85, 0.20, 1.0],  # 2
        [0.55, 0.80, 0.25, 1.0],  # 3
        [0.20, 0.70, 0.35, 1.0],  # >=4
    ]
)

_DENSITY_RAMP = np.array(
    [
        [0.94, 0.94, 0.97, 1.0],  # 0  -- lightest (near white)
        [0.70, 0.78, 0.92, 1.0],  # 1
        [0.42, 0.55, 0.83, 1.0],  # 2
        [0.20, 0.30, 0.62, 1.0],  # 3
        [0.05, 0.10, 0.35, 1.0],  # >=4 -- darkest (navy)
    ]
)

_PALETTES = {"coverage": _RAMP, "density": _DENSITY_RAMP}


def _resolve_palette(palette: str) -> np.ndarray:
    """Ramp for a palette name; fail loudly on an unknown one rather than silently defaulting."""
    try:
        return _PALETTES[palette]
    except KeyError:
        raise ValueError(
            f"unknown palette {palette!r}; choose one of {sorted(_PALETTES)}"
        ) from None


def _count_color(count: int, ramp: np.ndarray = _RAMP) -> np.ndarray:
    return ramp[min(int(count), len(ramp) - 1)]


def _framed_camera(model, data):
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c = data.geom_xpos[g]
        r = float(model.geom_rbound[g]) or 0.0
        lo = np.minimum(lo, c - r)
        hi = np.maximum(hi, c + r)
    if not np.isfinite(lo).all():
        lo, hi = np.array([-5, -5, 0.0]), np.array([5, 5, 3.0])
    center = (lo + hi) / 2.0
    cam.lookat = center
    cam.distance = 1.4 * float(np.linalg.norm(hi - lo))
    cam.azimuth = 90.0
    cam.elevation = -70.0  # near top-down, matches the "room coverage" mental model
    return cam


def render_coverage_3d(
    model,
    data,
    result: CoverageResult,
    out_path: str,
    *,
    size: int = 1024,
    cam=None,
    marker_radius: float = 0.12,
    max_markers: int = 4000,
    palette: str = "coverage",
) -> str:
    """Render the world with sample points coloured by coverage count; write a PNG to ``out_path``.

    ``palette`` selects the colour encoding: ``coverage`` (red 0 -> green many) or ``density``
    (light 0 -> dark many, so overlapping-sensor regions read darker)."""
    import mujoco

    ramp = _resolve_palette(palette)

    points = result.points
    counts = result.counts
    if (
        len(points) > max_markers
    ):  # deterministic subsample to stay under the renderer's geom budget
        sel = np.linspace(0, len(points) - 1, max_markers).astype(int)
        points, counts = points[sel], counts[sel]

    # The offscreen framebuffer is fixed at compile time (default 640x480); asking for more raises.
    height = min(size, int(model.vis.global_.offheight))
    width = min(size, int(model.vis.global_.offwidth))
    max_geom = len(points) + 2000
    renderer = mujoco.Renderer(model, height=height, width=width, max_geom=max_geom)
    try:
        renderer.update_scene(data, camera=cam if cam is not None else _framed_camera(model, data))
        scene = renderer.scene
        eye = np.eye(3, dtype=np.float64).reshape(-1)
        for p, c in zip(points, counts, strict=False):
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([marker_radius, 0, 0], dtype=np.float64),
                np.asarray(p, dtype=np.float64),
                eye,
                _count_color(c, ramp).astype(np.float32),
            )
            scene.ngeom += 1
        rgb = renderer.render()
    finally:
        renderer.close()

    _write_png(out_path, rgb)
    return out_path


def _write_png(path: str, rgb: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # PIL is a mujoco extra; fail loudly rather than silently skip
        raise RuntimeError("render_coverage_3d needs Pillow (pip install pillow)") from exc
    Image.fromarray(rgb).save(path)


def render_heatmap_2d(
    result: CoverageResult,
    out_path: str,
    *,
    resolution: float = 0.25,
    per_type: bool = False,
    title: str = "sensor coverage",
    palette: str = "coverage",
) -> str:
    """Top-down coverage-count heatmap over the floor plane; write a PNG to ``out_path``.

    Bins sample points into an x/y grid and shows the maximum coverage count per cell. ``palette``
    selects the colour encoding: ``coverage`` (red 0 -> green many) or ``density`` (light 0 -> dark
    many, so overlapping-sensor regions read darker).
    """
    ramp = _resolve_palette(palette)
    try:
        import warnings

        with warnings.catch_warnings():
            # A --system-site-packages venv can see two matplotlibs (system + pip); the older one's
            # optional Axes3D import then warns. We only draw 2D here, so silence that specific noise.
            warnings.filterwarnings("ignore", message="Unable to import Axes3D")
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError as exc:
        raise RuntimeError(
            "render_heatmap_2d needs matplotlib -- install the coverage extra: "
            "pip install 'roqsim_sensors[coverage]'"
        ) from exc

    pts = result.points
    if len(pts) == 0:
        raise ValueError("no sample points to plot")
    counts = result.counts
    lo = pts[:, :2].min(axis=0)
    hi = pts[:, :2].max(axis=0)
    nx = max(1, int(np.ceil((hi[0] - lo[0]) / resolution)) + 1)
    ny = max(1, int(np.ceil((hi[1] - lo[1]) / resolution)) + 1)
    grid = np.full((ny, nx), -1.0)  # -1 = no sample here
    ix = np.clip(((pts[:, 0] - lo[0]) / resolution).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - lo[1]) / resolution).astype(int), 0, ny - 1)
    for cx, cy, c in zip(ix, iy, counts, strict=False):
        grid[cy, cx] = max(grid[cy, cx], c)

    cmap = ListedColormap(ramp[:, :3])
    # No-sample cells must be visually distinct from "0 sensors". The density palette's 0 is already
    # near-white, so use a neutral grey there; the coverage palette's 0 is red, so light grey reads
    # as "no data" against it.
    cmap.set_under("#b8b8b8" if palette == "density" else "#dddddd")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 100], cmap.N)
    fig, ax = plt.subplots(figsize=(8, 8 * ny / max(nx, 1)))
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[lo[0], hi[0], lo[1], hi[1]],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    # Draw sensor positions.
    for fov in result.fovs:
        ax.plot(fov.origin[0], fov.origin[1], marker="*", color="black", markersize=12)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbar = fig.colorbar(
        im, ax=ax, ticks=[0, 1, 2, 3, 4], boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 100]
    )
    cbar.set_label("# sensors")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
