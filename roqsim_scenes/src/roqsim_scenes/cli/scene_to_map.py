"""Generate a Nav2 occupancy grid (map.pgm + map.yaml) from an imported scene.

Stage 3 for navigation experiments: a 3D scene alone does not run Nav2 — AMCL and the global costmap
need a 2D grid **co-registered with the scene**::

    scene.json + meshes/*.obj --[ scene_to_map.py ]--> map.pgm + map.yaml

Run::

    roqsim scenes scene-to-map --scene <scene_dir> --scan-height 0.51 --resolution 0.03 \
        --origin -15.1 -25 --free-from 0 0 --out <scene_dir>/map

What it does: slices every scene triangle with the horizontal plane at ``--scan-height``, rasterises
the resulting segments as occupied cells, then flood-fills free space from a known-free world point
(``--free-from``, normally the robot's start). Cells reached by the fill are free, cells crossed by
geometry are occupied, and everything else stays **unknown** — which is what a real SLAM map looks
like and what AMCL expects, rather than a naive free/occupied binary that would claim knowledge of
the inside of walls.

Why the scan height is a parameter and not a constant: a planar scan at height h is *exactly* what a
2D costmap sees, so h decides what counts as an obstacle. Take it from the robot model's lidar mount
(husky_a200: 0.51 m). On non-planar terrain, a slope projects into spurious occupancy at some heights
and not others — frequently the phenomenon under study, so never round it for convenience.

``--resolution`` and ``--origin`` exist to honour a *published* map's metadata even when the grid
itself is missing: matching them keeps goal coordinates in the paper's map frame comparable to ours.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

_OCC, _FREE, _UNKNOWN = 0, 254, 205


def _load_scene(scene_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    manifest = json.loads((scene_dir / "scene.json").read_text())
    out = []
    for obj in manifest["objects"]:
        if not obj.get("collide", True):
            continue  # visual-only geometry is not an obstacle
        verts, faces = [], []
        for line in (scene_dir / obj["mesh"]).read_text().splitlines():
            s = line.split()
            if not s:
                continue
            if s[0] == "v":
                verts.append([float(x) for x in s[1:4]])
            elif s[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in s[1:4]]
                faces.append(idx)
        if faces:
            out.append((np.array(verts), np.array(faces)))
    return out


def _robot_geoms(model, ctx) -> set[int]:
    """Geom ids belonging to any ``kind="robot"`` entity's kinematic subtree.

    The subtree, not just the base body: for a mobile base that is the chassis *plus its wheels*, and a
    map with four wheel-sized obstacles at the start pose is no better than one with a chassis there.
    Same rule ``contact_monitor`` uses to decide what counts as the robot.

    A world with no robot yields an empty set, so an environment-only world is unaffected.
    """
    import mujoco

    roots: list[int] = []
    for entity in ctx.entities.all():
        if entity.kind != "robot" or not entity.body:
            continue
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entity.body))
        if bid >= 0:
            roots.append(bid)
    if not roots:
        return set()

    def in_subtree(body: int) -> bool:
        while body > 0:
            if body in roots:
                return True
            body = int(model.body_parentid[body])
        return False

    return {g for g in range(model.ngeom) if in_subtree(int(model.geom_bodyid[g]))}


def _load_world(
    world: str, hull_at: float | None = None
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, list]:
    """World-space triangles of every geom in a compiled roqsim world, plus its XY bounds.

    The scene-dir path above reads an *imported* scene, so it cannot see anything a world YAML adds on
    top of it -- and for this repo that is exactly the interesting geometry: the tables, shelves and
    props a task places with `spawn_model`. A map built from the scene dir would omit them, which does
    not fail, it just produces a global costmap that plans straight through a table and leaves the local
    costmap to notice. Compiling the world is the only way to map what will actually be simulated.

    Every geom counts EXCEPT the robot's (see ``_robot_geoms``): a map is the environment, and a robot
    compiled in at its spawn pose would leave permanent occupancy exactly where every trial starts.

    Otherwise every geom counts, not just collidable ones. A 2D costmap is built from what the LIDAR returns, and
    roqsim's raycaster tests real triangles while ignoring contype/conaffinity -- so this substrate's
    convention of visual-only walls (a floorplan mesh, the Depot's shell, a table's legs) is precisely
    the geometry a scan plane hits. Filtering by `collide` here would delete the walls from the map.

    Primitives are tessellated because they have to be: `industrial_table`'s legs are boxes, and a map
    without them has no table in it.

    ``hull_at`` adds a second, DIFFERENT kind of obstacle: the convex-hull footprint of any COLLIDABLE
    mesh geom whose hull spans that height. It exists because triangles and collisions disagree for a
    mesh, and only for a mesh -- MuJoCo collides one by its CONVEX HULL but raycasts its real triangles.
    An `industrial_table` is the worked example: a scan plane at 0.209 m passes between its legs and the map
    gets four isolated cells, while the physics engine has a solid block from floor to tabletop. Measured
    in the pick world, rays fired at the table from 1.2 m away pass straight through and hit the wall
    3 m beyond it, so the robot's own lidar cannot see something it will certainly collide with, and a
    global planner routes through it. Off by default: it is not what a lidar returns, so a map built for
    perception fidelity should not have it. Turn it on for a map a planner will avoid obstacles with.
    """
    import mujoco  # local: roqsim_scenes' other tools do not need MuJoCo

    from roqsim.engine import Engine
    from roqsim.runner import config_for_input

    engine = Engine(config_for_input(world))
    engine.setup()
    engine.reset()
    model, data = engine.ctx.model, engine.ctx.data
    mujoco.mj_forward(model, data)

    # ...but NOT the robot. A map is the environment; the robot is what moves through it. Compiling the
    # world puts the robot at its spawn pose, so without this its chassis and wheels rasterise into a
    # blob of permanent occupancy exactly where every trial begins -- AMCL then localises against a
    # phantom obstacle at the start, and a planner refuses to leave. Worse, `--free-from` is documented
    # as "normally the robot's start", so the flood fill is seeded ON the blob and the damage looks like
    # a slightly small free area rather than a bug.
    skip = _robot_geoms(model, engine.ctx)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    hulls: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for g in range(model.ngeom):
        if g in skip:
            continue
        gtype = model.geom_type[g]
        size = model.geom_size[g]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            continue  # the floor: a scan plane above it never intersects it
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[g])
            adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            fadr, fnum = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            local = model.mesh_vert[adr : adr + num].reshape(-1, 3)
            faces = model.mesh_face[fadr : fadr + fnum].reshape(-1, 3)
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = (float(v) for v in size[:3])
            local = np.array(
                [(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)], dtype=float
            )
            faces = np.array(
                [
                    (0, 1, 3),
                    (0, 3, 2),
                    (4, 6, 7),
                    (4, 7, 5),
                    (0, 4, 5),
                    (0, 5, 1),
                    (2, 3, 7),
                    (2, 7, 6),
                    (0, 2, 6),
                    (0, 6, 4),
                    (1, 5, 7),
                    (1, 7, 3),
                ]
            )
        elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            r, hz = float(size[0]), float(size[1])
            n = 16
            ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            ring = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
            local = np.vstack(
                [np.column_stack([ring, np.full(n, -hz)]), np.column_stack([ring, np.full(n, hz)])]
            )
            faces = np.array(
                [(i, (i + 1) % n, n + i) for i in range(n)]
                + [((i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
            )
        else:
            continue  # spheres/ellipsoids/hfields: no obstacle in this substrate's worlds
        world_verts = (data.geom_xmat[g].reshape(3, 3) @ local.T).T + data.geom_xpos[g]
        out.append((world_verts, faces))
        if (
            hull_at is not None
            and gtype == mujoco.mjtGeom.mjGEOM_MESH
            and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
            and world_verts[:, 2].min() <= hull_at <= world_verts[:, 2].max()
        ):
            hulls.append(_hull_edges(world_verts[:, :2]))

    allv = np.vstack([v for v, _ in out])
    return out, allv[:, :2].min(axis=0), allv[:, :2].max(axis=0), [e for h in hulls for e in h]


def _hull_edges(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Edges of the 2D convex hull of *points*, as segments the rasteriser can consume.

    Monotone chain, so this needs no scipy.
    """
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) < 3:
        return []

    def half(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                out.pop()
            out.append(p)
        return out[:-1]

    ring = half(pts) + half(list(reversed(pts)))
    return [(np.array(ring[i]), np.array(ring[(i + 1) % len(ring)])) for i in range(len(ring))]


def _slice_segments(
    verts: np.ndarray, faces: np.ndarray, h: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Intersect triangles with the plane z=h -> 2D segments."""
    segs = []
    tri = verts[faces]  # (M,3,3)
    z = tri[:, :, 2]
    crosses = (z.min(axis=1) <= h) & (z.max(axis=1) >= h)
    for t in tri[crosses]:
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            za, zb = t[a, 2], t[b, 2]
            if (za - h) * (zb - h) > 0 or za == zb:
                continue
            u = (h - za) / (zb - za)
            pts.append(t[a, :2] + u * (t[b, :2] - t[a, :2]))
        if len(pts) >= 2:
            segs.append((pts[0], pts[1]))
    return segs


def _raster(segs, origin, res, w, hgt) -> np.ndarray:
    grid = np.full((hgt, w), _UNKNOWN, dtype=np.uint8)
    for p, q in segs:  # Bresenham-ish: sample along the segment at sub-cell spacing
        n = max(2, int(np.linalg.norm(q - p) / (res * 0.5)) + 1)
        for t in np.linspace(0, 1, n):
            x, y = p + t * (q - p)
            cx = int((x - origin[0]) / res)
            cy = int((y - origin[1]) / res)
            if 0 <= cx < w and 0 <= cy < hgt:
                grid[cy, cx] = _OCC
    return grid


def _flood(grid: np.ndarray, start_cell: tuple[int, int]) -> np.ndarray:
    hgt, w = grid.shape
    cx, cy = start_cell
    if not (0 <= cx < w and 0 <= cy < hgt) or grid[cy, cx] == _OCC:
        raise SystemExit(
            f"--free-from lands on {'an occupied cell' if 0 <= cx < w and 0 <= cy < hgt else 'outside the map'}. "
            "Pick a point you know is free (normally the robot's start pose)."
        )
    q = deque([(cx, cy)])
    grid[cy, cx] = _FREE
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < hgt and grid[ny, nx] == _UNKNOWN:
                grid[ny, nx] = _FREE
                q.append((nx, ny))
    return grid


def _write_pgm(path: Path, grid: np.ndarray) -> None:
    # ROS map_server: row 0 of the PGM is the TOP row = max y, so flip.
    img = np.flipud(grid)
    with open(path, "wb") as fh:
        fh.write(b"P5\n" + f"{img.shape[1]} {img.shape[0]}\n255\n".encode())
        fh.write(img.tobytes())


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", type=Path, help="scene dir (holds scene.json + meshes/)")
    src.add_argument(
        "--world",
        help="an roqsim world (YAML path or '<pkg>:<world>' ref), compiled through the plugin pipeline. "
        "Use this when the obstacles are added by the world rather than baked into the scene",
    )
    ap.add_argument("--out", type=Path, required=True, help="output basename (writes .pgm + .yaml)")
    ap.add_argument(
        "--scan-height", type=float, required=True, help="lidar scan plane, metres above z=0"
    )
    ap.add_argument(
        "--resolution", type=float, default=0.05, help="m/cell (match the published map)"
    )
    ap.add_argument(
        "--origin", type=float, nargs=2, help="map origin x y (match the published map)"
    )
    ap.add_argument(
        "--free-from", type=float, nargs=2, default=[0.0, 0.0], help="a known-free world point"
    )
    ap.add_argument(
        "--pad", type=float, default=1.0, help="metres of margin around the scene bounds"
    )
    ap.add_argument(
        "--collision-hulls",
        action="store_true",
        help="also mark the convex-hull FOOTPRINT of collidable mesh geoms spanning the scan height. "
        "MuJoCo collides a mesh by its convex hull but raycasts its real triangles, so a table whose "
        "scan plane passes between its legs is invisible to the lidar AND to this map while still being "
        "a solid block to the physics -- a planner then routes straight through it. Use for a map a "
        "planner navigates by; leave off for one that reproduces what a scan returns (--world only)",
    )
    args = ap.parse_args(argv)

    hull_segs: list = []
    if args.world:
        geometry, lo, hi, hull_segs = _load_world(
            args.world, hull_at=args.scan_height if args.collision_hulls else None
        )
    else:
        manifest = json.loads((args.scene / "scene.json").read_text())
        lo = np.array(manifest["bounds_min"][:2])
        hi = np.array(manifest["bounds_max"][:2])
        geometry = _load_scene(args.scene)
    origin = np.array(args.origin) if args.origin else lo - args.pad
    far = hi + args.pad
    w = int(np.ceil((far[0] - origin[0]) / args.resolution))
    hgt = int(np.ceil((far[1] - origin[1]) / args.resolution))

    segs = []
    for verts, faces in geometry:
        segs += _slice_segments(verts, faces, args.scan_height)
    segs += hull_segs
    if not segs:
        raise SystemExit(
            f"nothing intersects z={args.scan_height} m. Either the scan height is wrong or the scene "
            "is empty at that plane -- do not ship an all-unknown map."
        )

    grid = _raster(segs, origin, args.resolution, w, hgt)
    occ_before = int((grid == _OCC).sum())
    grid = _flood(
        grid,
        (
            int((args.free_from[0] - origin[0]) / args.resolution),
            int((args.free_from[1] - origin[1]) / args.resolution),
        ),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pgm = args.out.with_suffix(".pgm")
    _write_pgm(pgm, grid)
    args.out.with_suffix(".yaml").write_text(
        f"image: {pgm.name}\n"
        f"mode: trinary\n"
        f"resolution: {args.resolution}\n"
        f"origin: [{origin[0]}, {origin[1]}, 0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        # 0.196, not 0.25. nav2_map_server classifies a pixel by occ = (255 - value)/255 and calls it
        # free when occ < free_thresh. The unknown shade written above is 205, whose occ is exactly
        # 0.19607 -- so free_thresh 0.196 leaves it UNKNOWN (0.19607 is not < 0.196) while 0.25 makes
        # every unknown cell load as free space. This file writes a trinary map; declaring 0.25 threw
        # the third state away at load time, letting the planner route through the region outside the
        # walls and feeding AMCL a likelihood field that claims knowledge it does not have. 0.196 is
        # the canonical ROS value and it exists for exactly this reason.
        f"free_thresh: 0.196\n"
    )
    free = int((grid == _FREE).sum())
    unk = int((grid == _UNKNOWN).sum())
    print(
        f"MAP_OK {w}x{hgt} @ {args.resolution} m/cell origin={list(np.round(origin, 3))}\n"
        f"  segments={len(segs)} occupied={occ_before} free={free} unknown={unk} "
        f"({100 * free / (w * hgt):.1f}% free)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
