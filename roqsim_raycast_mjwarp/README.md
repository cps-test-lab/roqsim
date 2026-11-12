# roqsim_raycast_mjwarp

A **GPU ray-cast backend** for roqsim's lidars, on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).
It stands behind `roqsim.raycast` like the CPU cast, so every sensor built on roqsim's ray-cast base
(`lidar`, `livox_mid360`, `seyond_robin_w1g`) takes it without knowing Warp exists.

Installed, it needs no line in a world. roqsim's default, `raycaster: auto`, casts a lidar here
where this backend can run and the lidar casts at least `auto_min_rays` rays per sweep (2,000 by
default: a 3D lidar), and on the CPU otherwise. Each lidar's choice and its reason are logged at load
and recorded as a `raycaster` condition, which a recording and a campaign's run directory carry.

```yaml
sim:
  raycaster: auto              # the default; cpu or mjwarp force a backend for every lidar
  raycasters:
    mjwarp: {device: "cuda:0", auto_min_rays: 2000}
components:
  - spawn_sensor: {model: mid360, frame_id: livox_frame}
    components:
      - livox_mid360: {raycaster: mjwarp}   # or this lidar only
```

The user guide is roqsim's `docs/rendering.rst`, "Choosing a ray caster"; how it works is the
docstring of `roqsim_raycast_mjwarp.backend`.

## Installing it

```bash
pip install 'roqsim_raycast_mjwarp[mjwarp]'   # MuJoCo Warp 4.0.1, Warp 1.19.0, MuJoCo on roqsim's line
```

PyPI's Warp 1.19.0 is the CUDA 13 build and needs a driver of CUDA 13.0 or later. For an older
driver, install the CUDA 12 build of the same release from Warp's release page over it:

```bash
pip install https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl
```

## What it gives, and what it refuses

- **The CPU's hits**, but for float32's few rays: a ray grazing an edge, a tie between two visible
  geoms at the same distance. The lidars draw their noise per ray, so such a ray changes its own
  point and no other. Distances agree to float32's precision (the parity test's tolerance
  is 50 µm). `tests/test_parity.py` counts and bounds each kind, for a 2D lidar, a VLP-16 and a
  Mid-360, in the default room and the depot, with an obstacle present and absent.
- **A mesh hit from either side**, as MuJoCo's `mj_ray` hits it. MuJoCo Warp's own `mjw.rays` culls
  a triangle met from behind, so a lidar inside a mesh would see through it; the backend's kernel
  calls MuJoCo Warp's private ray functions to avoid that, which is why the extra pins its version.
- **What the CPU cast honours**: the visible geom groups, `flg_static`, the excluded body, an absent
  entity (`roqsim.presence`) and roqsim's rule for hits beyond the cutoff. The lidars' range window
  and noise are applied after the cast, the same on either backend.
- **One device scene per world**, shared by every lidar that chose it, its BVH refitted once per
  tick. The same state cast twice gives the same bytes.
- **No fallback when named.** Without MuJoCo Warp, without a CUDA device, with a Warp build the
  driver cannot run, or with a model `mjw.put_model` refuses, a world whose lidar names `mjwarp`
  does not load, and the error names the lidar, the backend and the reason. Under `auto` the lidar
  casts on the CPU instead, and its recorded choice gives that reason.

It pays for a 3D lidar and a mesh-heavy world. A 2D lidar in a small room gains nothing: each
sweep costs a fixed hand-over to the device, about what a 2D sweep costs on the CPU. That is why
`auto` leaves a lidar below `auto_min_rays` on the CPU; in a mesh-heavy world, where even a 2D
scanner is cheaper here, lower it.

## Tests

`make test` in roqsim runs them. Without a CUDA device, or without Warp, the GPU tests skip and say
why; the tests of the failures run everywhere.
