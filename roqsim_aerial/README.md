# roqsim_aerial

Aerial-vehicle plugins and models for [roqsim](../README.md).

| | |
| --- | --- |
| Models | `crazyflie_2` — Bitcraze Crazyflie 2 nano quadrotor (MIT, from MuJoCo Menagerie) |
| Plugins | `quadrotor_controller` — cascaded position + attitude control over collective thrust and three body moments |
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

## Licensing

Package code is Apache-2.0. The Crazyflie 2 model is MIT — see
`src/roqsim_aerial/models/crazyflie_2/CRAZYFLIE_2_LICENSE`, and the substrate's `THIRD_PARTY`
convention in the root README.
