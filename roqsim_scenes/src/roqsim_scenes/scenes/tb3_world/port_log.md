# tb3_world scene — port log

**Source**: ROBOTIS `turtlebot3_simulations`, `turtlebot3_gazebo/worlds/turtlebot3_world.world` and
the `turtlebot3_world` model it includes, at the commit pinned in `CREDITS.txt` (Apache-2.0). The
world file carries no geometry of its own: two Fuel `<include>`s (ground plane, sun) and one
`<include>` of that model, whose single link declares 9 `<cylinder>` pillars, a hexagonal wall from
`meshes/wall.dae` and five ROS-logo blocks from `meshes/hexagon.dae`, each with its own `<collision>`
and `<visual>`.

**Pipeline** (`external/convert/build_tb3_world_scene.py` runs both stages with these arguments):

```
worlds/turtlebot3_world.world  (+ models/turtlebot3_world, via --model-path)
  --[ roqsim scenes sdf-to-scene --ignore-up-axis wall.dae --ignore-up-axis hexagon.dae ]-->
      scene.json + meshes/*.obj
      34 collidable objects: 9 pillar cylinders, 5 logo blocks, 20 wall prisms (render: false)
      15 visual objects, the source meshes and primitives unchanged (collide: false)
  --[ roqsim scenes scene-to-mjcf ]--> ../../worlds/tb3_world/tb3_world.xml
```

## Two decisions, both of which change the world if made the other way

**The wall's `<up_axis>` contradicts its own geometry.** `wall.dae` and `hexagon.dae` declare
`<up_axis>Y_UP</up_axis>` (and `<unit name="inch" meter="0.0254"/>`) but are authored Z-up. Honouring
the declaration — which is what the COLLADA spec says to do, and what this repository's reader does
for every other asset — rotates the hexagon into the x–z plane: a fence standing on edge, 6.6 m tall
and 1.27 m thick, extruded sideways. Read as authored it is a room: a hexagonal fence 1.27 m tall on
a footprint 6.599 m across the vertices. Three independent things say the second reading is the one
the source means, so the port asserts it per file (`--ignore-up-axis`, which the importer announces
on every mesh it applies to) rather than editing the asset:

| evidence | value |
| --- | --- |
| the SDF places this mesh at `<pose>0 0 -0.3 0 0 -1.5708</pose>` | a fence sunk 0.30 m into the floor, 0.97 m of it above — coherent only for a Z-up footprint |
| the pillars beside it are `<cylinder>` primitives, 0.50 m tall, on a 3×3 grid at ±1.1 m | they sit *inside* the footprint of the Z-up reading and nowhere near the Y_UP one |
| the occupancy grid ROBOTIS publishes for this world (`turtlebot3_navigation2/map/map.pgm`) | walls at 5.15 m flat-to-flat, against 5.080 m for the Z-up reading and nothing like it for Y_UP |

**The wall is a closed ring, so it cannot be collided as a mesh.** MuJoCo collides a mesh by its
convex hull, and the hull of a hexagonal ring is the filled room (hull 35.9 m³ against 7.5 m³ of
actual wall, ratio 4.8): a robot spawned inside would sit in solid geometry, unable to move, while
every trial still ran to completion and reported a plausible result. The importer refuses that, and
its two existing cuts cannot help — the ring is one connected component, and a reflex-edge cut at the
inner corners leaves the outer faces connected the long way round. What does cut it is the wall's own
footprint: `scene_mesh_io.split_extruded_shell` decomposes an extrusion of a 2D footprint into
convex prisms over a trapezoidal decomposition of it, here **20 prisms**, each convex by construction
(hull/mesh ratio 1.00). The decomposition is exact — the prisms' total volume is the ring's to six
decimals — so no material is shaved off and none is invented. The *visual* stays the single source
mesh, so what is drawn is the source's geometry, not the colliders.

## Verified

Against the source's own declarations, which are authored independently of the meshes:

| check | result |
| --- | --- |
| 9 pillars, from the SDF's `<cylinder>` primitives | r = 0.150 m, 0.50 m tall, centres exactly at x, y ∈ {−1.1, 0, 1.1}, z ∈ [0, 0.5] |
| wall footprint, inner surface | vertex-to-vertex 5.866 m, flat-to-flat 5.080 m (= 461.88 in and 400 in at `meter="0.0254"` × the SDF's `<scale>0.25`) |
| wall thickness and height | 0.3175 m, z ∈ [−0.30, 0.97] — the source's `<pose>` and mesh, unaltered |
| MJCF compiles and loads | 51 geoms, 49 meshes, no MuJoCo warnings |

Against the occupancy grid ROBOTIS publishes for this world — an independent artifact, produced by
SLAM rather than from the model, so it is evidence and not a tolerance. Both grids are sampled onto
one 5 cm grid, eroded by the robot radius, and flood-filled from inside the room:

| erosion radius | this scene | published grid | IoU |
| --- | --- | --- | --- |
| 0.175 m | 15.625 m² | 15.920 m² | 0.949 |
| 0.220 m (nav2's TurtleBot 3 footprint radius) | 14.088 m² | 14.453 m² | 0.940 |

The published grid is slightly the more generous of the two, by about one 5 cm cell along each wall,
which is what a SLAM map of a wall looks like. Its corners fall ~0.28 m short of the geometry's
(vertex-to-vertex 5.30 m measured against 5.866 m authored) — a hexagon's corner is the part a lidar
sees most obliquely and maps worst, and it is why the grid, not the model, is the thing being checked
here.

## Assumptions and artifacts, none of them the source's

- **`--ignore-up-axis` is an assertion about two files**, argued above. If a future upstream commit
  re-exports those meshes Z-up, the flag becomes wrong and the sha256 pin is what will catch it.
- **The wall collides as 20 prisms rather than 6 mitred plates.** A trapezoidal decomposition cuts at
  every vertex's x, so a rotated hexagon yields more pieces than it has sides. Exactness and
  convexity are what matter for physics; the piece count only costs geoms.
- **The five ROS-logo blocks are imported, collidable, and outside the wall**, as in the source. A
  robot cannot reach them. The largest is buried 0.5 m, which is why the drawn floor's depth rule had
  to learn the difference between a prop standing in the floor and a floor modelled below the ground
  (`scene_to_mjcf._lowest_renderable_z`): the source hides that block's underside, and so does this.
- **No 2D map is committed here.** `roqsim scenes scene-to-map --world roqsim_scenes:tb3_world
  --scan-height 0.2` projects one, and the numbers above come from it, but a map belongs to the
  experiment that fixes a scan height and a map frame, not to the scene.
