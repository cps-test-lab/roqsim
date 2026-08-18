# external/train — training locomotion policies

Build tooling, not runtime: `external/` is where this repo keeps the things that *produce* vendored
artifacts (see `external/convert/README.md`). A trained policy is a build artifact like a converted
MJCF, so the trainer lives here and the checkpoint it emits is committed under the owning family
package — `roqsim_humanoid/src/roqsim_humanoid/policy/<name>/`.

**Nothing under `roqsim_*` imports this**, and nothing here is installed into the sim venv. That is not
tidiness, it is a hard requirement: `make venv` installs every `roqsim_*` package editable, so declaring
`jax[cuda]`/`brax` in one would drag multi-GB CUDA wheels into a venv that is deliberately CPU-only
(the Makefile says exactly this about torch). Training gets its own disposable venv, `.venv-train`,
mirroring how `external/.venv-tools` isolates the mesh-conversion toolchain.

Training is **explicitly invoked** and never wired into `external_assets.yaml`: that manifest is driven
by `make venv` and is fail-soft for *network* errors, whereas a training run is a long GPU job that must
only happen when asked.

## Hardware

Needs an NVIDIA GPU. MJX parallelises across GPU, not CPU threads, so the 96-thread cluster is the wrong
shape for training — it is the right shape for *evaluating* the result (a headless roqsim trial runs at
~30x realtime on one core, so a thousand-cell sweep is minutes; see the campaign in
a downstream experiment package).

Reference point: a humanoid locomotion policy trains in ~56 min at 8192 envs on an RTX 4090 (24 GB).
A stand is a much smaller problem than the joystick task it derives from — no gait to discover, no
velocity curriculum, no command sampling — so expect **20–40 min per run** there, and 3–6 runs while the
reward is tuned. On 6 GB expect ~1024–2048 envs instead of 8192 and roughly 2–3x longer.

## Layout

```
external/train/
  README.md          this file
  requirements.txt   pinned trainer deps (jax, brax, mujoco-playground)
  g1_stand/          the G1 standing task: env, rewards, arm disturbance, export
```

## What a trainer must get right

Two things that silently invalidate a run, both learned the hard way here:

1. **Drive non-actuated joints through their position actuators with the deployment gains — never by
   writing `qpos`.** Setting `qpos` makes a limb infinitely stiff: it forces the joints regardless of
   reaction torque, so the base feels a different disturbance than a real PD-servo'd arm transmits. At
   deployment the G1's arm is a `<position>` actuator per joint that *lags* its MoveIt target under
   load, and that lag is part of what the policy must learn to reject.
2. **Emit a spec next to the checkpoint, and verify the export numerically.** The spec
   (`roqsim.policy.PolicySpec`) is what the runtime uses to assemble the observation, so trainer and
   runtime cannot disagree about the layout — an observation mismatch does not error, it just makes the
   robot twitch. After export, feed the same 200 random observations through the JAX policy and the
   TorchScript one and assert they agree to <1e-5; a subtly wrong export is indistinguishable from a
   badly trained policy, and that confusion is expensive.
