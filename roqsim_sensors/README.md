# roqsim_sensors

Generic, robot-family-agnostic sensor plugins for [roqsim](../README.md): a robot doesn't need
to be a mobile base or an arm to carry a lidar or a camera, so these live in their own sibling
package rather than `roqsim_mobile`/`roqsim_manipulation`. ROS-free; ROS coupling lives in
`roqsim_ros_bridge`.

## Plugins

| plugin | role |
|--------|------|
| `lidar` | 2D lidar via `mj_multiRay` (no GL). Declares a `scan` (`LaserScan`) output endpoint. Ported from our earlier in-house nav prototype's `Lidar`. |
| `livox_mid360` | Livox Mid-360 3D lidar via `mj_multiRay` (no GL). Casts a spherical ray grid over the device's 360°×59° (−7°..+52°) FoV and declares a `cloud` (`PointCloud2`) output endpoint. Sibling of `lidar` (point cloud, not planar scan). Bundled with the standalone `mid360` mount below. Uniform-grid sampling approximates the device's non-repetitive scan pattern — a documented substrate limit. |
| `seyond_robin_w1g` | Seyond Robin W1G forward-facing solid-state 3D lidar via `mj_multiRay` (no GL). A thin subclass of `livox_mid360` for a **bounded** FoV: azimuth is a −60°..+60° band with inclusive endpoints (not a wrapping 360° sweep, `AZIMUTH_WRAPS = False`), elevation −35°..+35°, boresight +x; declares a `cloud` (`PointCloud2`) endpoint on `seyond/points`. Defaults follow the datasheet (120°×70° FoV, 0.1–70 m range, 10 Hz). Uniform-grid sampling approximates the device's proprietary scan pattern/point rate — a documented substrate limit. Bundled with the standalone `robin_w1g` mount below. |
| `oakd_camera` | OAK-D Pro RGB-D camera rendered via `mujoco.Renderer` (GL). Declares `image`, `depth`, and a `camera_info` for each stream; depth in `32FC1` metres or `16UC1` millimetres (`depth_encoding`). Ported from our earlier in-house nav prototype's `Camera`/`CameraFrame`. Bundled as a default TurtleBot 4 sensor (`roqsim_mobile`'s `turtlebot4.manifest.yaml`). |
| `realsense_d435` | Intel RealSense D435(i) colour stream, plus an **opt-in** depth image (`depth: true`) and `PointCloud2` (`points: true`, implies depth). Declares `image`/`camera_info` and, when enabled, `depth`/`points` output endpoints, topic/frame naming following `realsense-ros` (`camera/depth/image_rect_raw`, `camera/depth/color/points`). The cloud is reprojected into the ROS optical frame; it is opt-in because a 640×480 frame is up to 307k points. Depth is `32FC1` metres unless `depth_encoding: 16UC1` asks for the driver's own millimetres. Bundled with the standalone `d435` mount below, or attach one to your own robot's MJCF (the `open_manipulator_x` arm ships an eye-in-hand one). |
| `realsense_d415` | Intel RealSense D415 colour stream (RGB only). Same renderer as `realsense_d435`; a separate plugin/model so a robot carrying a real D415 is faithful. Bundled with the `d415` mount below. |
| `realsense_d455` | Intel RealSense D455 colour stream, plus the same **opt-in** depth/`PointCloud2` path as `realsense_d435` (which it subclasses, with the D455's own 0.6–6 m clip range). A separate plugin/model so a robot carrying a real D455 is faithful — the D455 is the wide-FOV, longer-range member (87°×62° colour, 95 mm stereo baseline, 0.6–6 m range). Bundled with the `d455` mount below. |
| `zivid` | Zivid 3 XL250 industrial structured-light 3D camera, rendered via `mujoco.Renderer` (GL). RGB-D (a `depth_camera` sibling of `oakd_camera`): declares `image`, `depth`, and a `camera_info` per stream, standing in for the sensor's coloured point cloud. Datasheet defaults: 39° square FOV, ~5 Hz (250–1500 ms typical 3D capture), depth clipped to the 1.3–5 m working range; topic/frame naming approximates `zivid-ros`. Bundled with the standalone `zivid` mount below. |
| `spawn_sensor` | Attach a standalone sensor MJCF (mesh + camera/site) at a fixed mount pose — the robot-free analogue of `spawn_robot`/`spawn_arm` for a sensor that isn't carried by anything (an overhead camera, a mast lidar). Pulls in the model's default capture plugin from its `<model>.manifest.yaml` manifest, e.g. `{model: d435}` brings in `realsense_d435`. |
| `fiducial_marker` | Add an ArUco/AprilTag fiducial to the scene as a flat, non-colliding geom so cameras render it and a detector can decode it. The marker image is generated at build time with OpenCV (`cv2.aruco`, one dep covers both families) and injected as raw texture data. Fixed placement: free-standing (`pose`) or welded to a robot body (`attach_to` + `prefix`). Needs the optional `markers` extra: `pip install 'roqsim_sensors[markers]'`. |
| `force_torque` | Six-axis force/torque sensor at a site — the one sensor here that reports **contact force** rather than geometry, which is the whole measurement for a contact-rich manipulation task (insertion, polishing, compliant assembly). Adds MuJoCo's `<force>`/`<torque>` pair on a site (yielding to a vendor MJCF that already ships one), reports the wrench in the `sensor`, `base` or `world` frame, and publishes both a `wrench` (`WrenchStamped`) endpoint and a `WrenchReader` on the blackboard under `ft:<name>` for in-process controllers. **Where the sensor cuts matters**: a site sensor measures the wrench transmitted *through* that site from its children, so the tool must hang below it — a tool attached above the site reads identically zero. |
| `ground_truth_pose` | Publish a body's *true* world pose (no odometry drift, no localisation), as a TF-shaped endpoint — the reference signal an evaluation compares a stack's estimate against. See `docs/ground_truth.rst`. |
| `sensor_coverage_probe` | Report the **sensor coverage** of a world (a world-YAML toggle for the `coverage` subpackage below). Computes once at `configure` how much of the room / which objects the world's sensors observe, and by how many (0..N), then writes an agent-digestible `report.json` + a render. `sensors: auto` evaluates every MuJoCo camera; give an explicit list for lidars/Livox. Rendering needs a GL backend, which `import roqsim` selects for this machine — set `MUJOCO_GL` only to override. |

Every camera plugin reads its resolution/FOV from the named MuJoCo `<camera>` element (`resolution`,
`fovy`) rather than duplicating them in plugin config, and skips the (expensive) render while a
transport reports zero subscribers — gated on **every** output endpoint fed by that render pass
(`image`, `image_compressed`, *and* a subclass's `depth`/`points`), not on `image` alone, so a consumer
that wants only depth, or only the compressed stream, still gets frames. `camera_info` is deliberately
not a gate: it needs no render, so a lone info subscriber must not switch the renderer on. When
subscriber counts aren't knowable (no bridge loaded) the render is always on.

Colour goes out in both formats a real driver offers: raw `sensor_msgs/Image` and
`sensor_msgs/CompressedImage` on `<image topic>/compressed`, `image_transport`'s convention — so a
stack written against a hardware driver finds the topic it expects. Both come off the same array (the
ROS bridge owns the codec; no camera plugin imports one) and both skip the *publish* as well as the
render when unsubscribed, so the second stream costs nothing until something asks for it. Hence
on by default: `compressed: false` opts out, `jpeg_quality` (default 95, `image_transport`'s own)
sets the quality. `compressed: false` opts out of the compressed companions of both streams.

Depth goes out as **float32 metres** (`32FC1`, `inf` for "no return") by default, which is lossless in
the unit the renderer produces. `depth_encoding: 16UC1` publishes **millimetres with 0 for invalid**
instead — what a real RealSense driver puts on `depth/image_rect_raw`, so a stack or a bag comparison
written against hardware sees the format it expects, at half the bytes. It quantises to a millimetre
and cannot reach past 65.535 m, so `clip_far` is checked against that ceiling at load time rather than
saturating (an OAK-D's 100 m default therefore has to come down to opt in). A point cloud is
unaffected either way: it reprojects the float metres, whatever the depth topic advertises.

Every depth camera publishes intrinsics for the depth stream as well as the colour one, on the
`camera_info` beside its depth image — a consumer that rectifies or reprojects depth subscribes to
that topic, and given only the colour stream's it waits forever. Both carry the same numbers, because
one MuJoCo camera renders both streams; real hardware images depth through separate optics, which a
one-camera plugin cannot reproduce.

A `16UC1` camera also offers `<depth topic>/compressedDepth` — `image_transport`'s depth transport,
RVL-coded (lossless, about 5:1) — so a driver-shaped consumer finds that topic too. It is absent under
`32FC1` because the codec is 16-bit, which is the format's constraint rather than a policy: asking for
the topic anyway is an error, not a silent no-op. Its encoder drops returns past 10 m
(`image_transport`'s own `depth_max` default, mirrored so the bytes match a driver's), so a camera
that sees further has to lower `clip_far` or switch the companion off rather than publish two depth
topics that disagree. The encode is not cheap — 35 ms for a 1280×720 frame against JPEG colour's
2.9 ms, since RVL has no C library behind it — but like the colour companion it is skipped entirely
while nothing subscribes.

### Conventions for standalone sensor models

Two rules bind every camera model bundled here (spawned via `spawn_sensor`); both are locked by
tests in `tests/test_spawn_sensor.py`:

- **Look direction.** A standalone mount looks along its local **+y** (out the lens), **+z up**, at
  `rpy [0,0,0]` — so a mount placed with no rotation looks *horizontally* toward a wall, not up. Aim
  it with the `spawn_sensor` `rpy` yaw (e.g. yaw −90° → faces +x). A device whose mesh puts the lens
  on another axis carries a reorient quat on its `mount` body to satisfy this: the d435/d415/d455
  meshes have the lens on mesh-local +z, so their mount body sets `quat="0 0 0.70710678 0.70710678"`
  (maps mesh +z→+y); the `zivid` mesh is authored lens=+y already, so it needs none. This is *not*
  the eye-in-hand flange convention (+z out of flange); it is the agreed standalone-mount convention.
- **Resolution cap.** A camera rendered through a `spawn_sensor` mount cannot use a `<camera
  resolution>` larger than **640×480** — MuJoCo's default offscreen framebuffer. Raising the sensor
  model's own `<visual><global offwidth/offheight>` does **not** help: `MjSpec.attach` drops the
  child's visual-global settings, so the compiled world keeps the parent's 640×480 and the camera
  plugin's `mujoco.Renderer(model, h, w)` raises `ValueError` above it. A square high-res sensor
  (e.g. the Zivid XL250, native 704/1408/2816) must render a downscaled square that fits — `zivid`
  uses 480×480, documenting its native modes in the XML. To exceed 640×480, the *world* MJCF must
  raise `offwidth`/`offheight` and the plugin config must override `width`/`height`.

## Bundled models

**One folder per model**: `models/<name>/<name>.xml` with that model's meshes in its own `meshes/`
subdir (reached bare through `<compiler meshdir="meshes">`) and its manifest, licence and thumbnail
beside it — the same layout `roqsim_manipulation_assets` and `roqsim_assets` use, and one
`roqsim.models.resolve_model` accepts directly. Adding or removing a sensor is one `mv`, and its licence
sidecar travels with the mesh it covers. Three of the six have meshes that are **generated, not
committed** (`mid360`, `zivid`, `robin_w1g` — vendor CAD with unclear redistribution terms; see
`external/external_assets.yaml` and run `make external-convert`), which is the other reason the files
are grouped per device rather than pooled: nothing in a shared `meshes/` said which files a fresh
clone would be missing. `tests/test_sensor_model_layout.py` locks the layout, the mesh resolution and the
`package-data` globs.

| model | role |
|-------|------|
| `d435` | A standalone Intel RealSense D435 mount: visual mesh + a `d435_color` camera, sized/posed from the real device. Spawn with `spawn_sensor: {model: d435, pos: ..., rpy: ...}`; its manifest (`d435.manifest.yaml`) auto-attaches `realsense_d435`. The mesh is converted (COLLADA → OBJ, joined + decimated) from the official [realsense-ros](https://github.com/IntelRealSense/realsense-ros)'s `realsense2_description` package (Apache-2.0 — see `models/d435/D435_MESH_LICENSE`); the camera's pose/orientation is carried over from that package's `_d435.urdf.xacro`. |
| `d415` | A standalone Intel RealSense D415 mount: visual mesh + a `d415_color` camera (looks along +y). Spawn with `spawn_sensor: {model: d415, ...}`; its manifest auto-attaches `realsense_d415`. Mesh from `realsense2_description`'s `d415.stl` (decimated to ~13k tris via Blender, Apache-2.0 — see `models/d415/D415_MESH_LICENSE`). |
| `d455` | A standalone Intel RealSense D455 mount: visual mesh + a `d455_color` camera (looks along +y) with the D455's wide 87°×62° colour FOV baked in (fovy 62° + the 640×400 / 1.6-aspect resolution ⇒ ~87° horizontal). Spawn with `spawn_sensor: {model: d455, ...}`; its manifest auto-attaches `realsense_d455`. Mesh from `realsense2_description`'s `d455.stl` (decimated to ~6k tris via Blender, Apache-2.0 — see `models/d455/D455_MESH_LICENSE`); the camera pose is mapped from that package's `_d455.urdf.xacro`. |
| `mid360` | A standalone Livox Mid-360 3D-lidar mount: visual meshes (grey housing + blue laser dome) + a `mid360` scan site at the dome's optical centre, plus a hidden FOV coverage envelope mesh. Spawn with `spawn_sensor: {model: mid360, pos: ..., rpy: ...}`; its manifest (`mid360.manifest.yaml`) auto-attaches `livox_mid360` (frame `livox_frame`). Add `show_fov: true` to draw the translucent 360°×59° coverage volume. The meshes are tessellated (Open CASCADE) and decimated from Livox's own Mid-360 STEP assemblies (housing + FOV) — see `models/mid360/MID360_MESH_LICENSE` for provenance. |
| `zivid` | A standalone Zivid 3 XL250 mount: visual mesh (dark housing, glass strip + two lens windows) + a `zivid_color` camera at the +x optical module (looks along +y), plus a hidden 39° square FOV frustum. Spawn with `spawn_sensor: {model: zivid, pos: ..., rpy: ...}`; its manifest (`zivid.manifest.yaml`) auto-attaches `zivid`. Add `show_fov: true` to draw the translucent coverage volume out to the 2.5 m focus distance. The housing mesh is decimated (~8k tris via Blender) from Zivid's own Zivid 3 STL; optical parameters (baseline, FOV, focus) come from the XL250 datasheet — see `models/zivid/ZIVID_MESH_LICENSE` for provenance. |
| `robin_w1g` | A standalone Seyond Robin W1G solid-state-lidar mount: visual mesh (dark wedge housing, window on +x, rear connector) + a `robin_w1g` scan site just inside the window, plus a hidden rectangular FOV frustum. Spawn with `spawn_sensor: {model: robin_w1g, pos: ..., rpy: ...}`; its manifest (`robin_w1g.manifest.yaml`) auto-attaches `seyond_robin_w1g` (frame `seyond_lidar`). Unlike the 360°-dome Mid-360 it looks *forward* along +x (`rpy [0,0,0]` faces a wall). Add `show_fov: true` to draw the translucent 120°×70° coverage volume (a representative 5 m shell, not the 70/150 m range). The mesh is tessellated (Open CASCADE) and decimated from Seyond's own Robin W1G STEP; the FOV frustum is authored from the datasheet — see `models/robin_w1g/ROBIN_W1G_MESH_LICENSE` for provenance. |

## Demo world

`worlds/all_sensors_demo.yaml` places every standalone-capable sensor in the default empty room —
the `mid360`, `d435`, `d415`, `d455`, and `zivid` mounts (bringing `livox_mid360`, `realsense_d435`,
`realsense_d415`, `realsense_d455`, `zivid`), the 2D `lidar` sharing the Mid-360 mount, and two `fiducial_marker` targets
— as a robot-free showcase and smoke test (`oakd_camera` is robot-carried; see `roqsim_mobile`'s
TurtleBot world). Every sensor draws its **field of view** (`show_fov`): the Mid-360 and Zivid reveal
their bundled coverage meshes, the RealSense cameras get a synthesised frustum. A
`sensor_coverage_probe` then writes a coverage report + heatmap + 3D render to `./coverage`. Cameras and
the coverage render need a GL backend and the markers need the `markers` extra:

```bash
roqsim sim roqsim_sensors/src/roqsim_sensors/worlds/all_sensors_demo.yaml
```

## Sensor coverage estimation

`roqsim_sensors.coverage` answers, for a **fixed** world: *how much of the room and which objects are
observed, and by how many sensors (0..N)?* — and helps search for a sensor layout that reaches a target
coverage. It adds no new hard dependency (numpy + mujoco); the 2D heatmap needs the optional `coverage`
extra (`pip install 'roqsim_sensors[coverage]'`), the 3D render needs only mujoco.

- One shared FOV definition (`coverage/fov.py`: `SensorFov`, a posed angular sector — camera `FRUSTUM`
  or lidar `CONE_BAND`) with a per-type **adapter registry** (`coverage/adapters.py`) that extracts it
  from each sensor's own parameters. A new/special sensor gets a new adapter there; the sensor plugins
  are never edited.
- The engine (`coverage/engine.py`) gates each point by range → angular FOV → line-of-sight
  (`mj_multiRay`, masking group 4 so an absent entity cannot occlude; an FOV-visualisation mesh
  is excluded by its alpha 0 instead, not by its group). Sampling (`coverage/sampling.py`) covers a
  3D room volume and object surfaces.
- Two outputs: an agent-digestible `report.json` (achieved coverage, per-object, uncovered regions,
  per-sensor contribution) and a human render (a top-down 2D heatmap and/or a 3D marker render).

Two front doors: the `sensor_coverage_probe` plugin (above) reports a world's coverage from its YAML;
the **`roqsim sensors coverage` CLI** searches for placements with the agent in the loop:

```bash
roqsim sensors coverage catalog                              # sensor types, FOV, cost, mount constraints
roqsim sensors coverage estimate \             # evaluate a placement set
    --world <mjcf-or-world-yaml> --placements p.json --target k=1,frac=0.95 --render both --out run/
roqsim sensors coverage greedy \               # deterministic max-coverage baseline
    --world <w> --target k=1,frac=0.9 --types livox_mid360,oakd_camera --mount-z 3.0 --out run/
```

The agent workflow (evaluate → read `report.json` gaps → refine placements → repeat) is described in the
`sensor-coverage` skill.

## Test

```bash
python -m pytest roqsim_sensors/tests
```
