# roqsim_aerial

Aerial-vehicle plugins and models for [roqsim](../README.md).

| | |
| --- | --- |
| Models | `crazyflie_2` — Bitcraze Crazyflie 2 nano quadrotor (MIT, from MuJoCo Menagerie) |
| Plugins | `quadrotor_controller` — cascaded position + attitude control over collective thrust and three body moments<br>`wind_field` — steady flow, 1-cosine gust, Dryden turbulence |
| Worlds | `crazyflie_2_demo.yaml` — takes off from the floor and holds 1 m |

```bash
roqsim sim roqsim_aerial/src/roqsim_aerial/worlds/crazyflie_2_demo.yaml
```

## Two things an aerial world needs that a ground world does not

**Air.** MuJoCo defaults `density` and `viscosity` to 0 — a vacuum. A drone still hovers there, but
nothing damps it, so a lateral step rings forever and the run reads as badly tuned gains rather than
as missing air. Every aerial world should carry:

```yaml
sim:
  integrator: rk4     # a quadrotor's attitude loop is stiff relative to the timestep
  density: 1.225      # kg/m^3, air at sea level
  viscosity: 1.8e-5   # Pa*s
```

`quadrotor_controller` warns rather than silently flying in vacuum.

**A controller, always.** A quadrotor MJCF has no stabiliser: an uncommanded drone is not a robot
standing still, it is a falling brick. `crazyflie_2`'s manifest therefore pulls
`quadrotor_controller` in automatically, and unlike a ground robot's manifest that entry is not a
convenience.

## The flight envelope is the experiment

The Crazyflie carries 0.35 N of collective thrust against a 27 g airframe — a thrust-to-weight ratio
of **1.32**. That margin, not the controller, is what an aerial campaign is usually about. The two
two plugins that vary it are `wind_field`, below, and core roqsim's `payload` (carried mass — see
[docs/plugins.rst](../docs/plugins.rst)).

```yaml
components:
  - spawn_robot: {model: crazyflie_2, prefix: "cf2_", pose: {position: {x: 0.0, y: 0.0}}}
    name: drone
    components:
      - quadrotor_controller: {target: [0.0, 0.0, 1.0]}
      - payload: {mass: 0.005}        # 5 g -> T/W 1.11
  - wind_field:
      steady: [3.0, 0.0, 0.0]
      gust: {magnitude: 6.0, onset: 4.0, duration: 1.5}
      turbulence: {intensity: 0.8, length_scale: 4.0}
```

Measured against a 1 m altitude hold, the boundary sits where physics says it does:

| payload | total | T/W | altitude held |
| --- | --- | --- | --- |
| 0 g | 27 g | 1.32 | 1.00 m |
| 3 g | 30 g | 1.19 | 0.91 m |
| 6 g | 33 g | 1.08 | 0.82 m |
| 9 g | 36 g | 0.99 | never leaves the floor |

The graded sag before the collapse is worth knowing: `quadrotor_controller` has no integral term, so
added weight buys steady-state altitude error rather than a cliff. The cliff arrives at T/W = 1.

## `wind_field`

`sim.wind` states a *constant*, which is the one wind a controller never has to reject: it trims
against it once and the error goes to zero. Rejection is tested by wind that changes, so this plugin
writes `model.opt.wind` each tick — steady flow, a MIL-F-8785C 1-cosine gust, and Dryden turbulence.
MuJoCo's own drag terms do the work; nothing here applies a force of its own. Three consequences:

- **One owner per knob.** Declaring both `sim.wind` and `wind_field` is refused rather than merged:
  the compiled model would say one thing and the first tick another, and the run's provenance would
  record the value that was immediately overwritten. State the mean flow in `steady:`.
- **Turbulence draws from the run's seed** through `ctx.rng_for`, so there is no `seed` key here and
  the whole world reproduces together. Because the episode is part of the key, repetitions of one
  configuration see *different* turbulence — samples of the weather, not copies of it.
- **Wind is inert in a vacuum**, for the same reason the drone is undamped in one: it acts through
  the density and viscosity drag terms. With both at 0 the plugin has no effect at all, and warns.

## Licensing

Package code is Apache-2.0. The Crazyflie 2 model is MIT — see
`src/roqsim_aerial/models/crazyflie_2/CRAZYFLIE_2_LICENSE`, and the substrate's `THIRD_PARTY`
convention in the root README.
