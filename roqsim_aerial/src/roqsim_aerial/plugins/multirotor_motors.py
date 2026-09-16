"""Actuation plugin: normalized per-rotor commands -> rotor thrusts and their reaction torques.

**Why per-rotor actuation exists at all.** A flight stack commands *motors*, not collective thrust:
PX4's simulator interface hands over `HIL_ACTUATOR_CONTROLS`, up to 16 normalized 0..1 outputs, of
which a quad-X uses the first four. The Crazyflie's MJCF exposes collective thrust plus three body
moments instead -- which is the right abstraction for a *controller*, because it is exactly what a
controller's output looks like before the mixer. That is also why it cannot be the interface to a
flight stack: the mixer's output is four numbers and there is nowhere to put them. Inverting the
mixer to recover collective-and-moments would be re-deriving inside the simulator the very mapping
the flight stack under test is responsible for getting right. So the airframe carries four force
actuators and this plugin owns them.

This is roqsim's counterpart of gz-sim's ``MulticopterMotorModel``, and matches it on the parts that
change the flight dynamics: a normalized command scaled to a maximum thrust, a first-order
motor+ESC lag, and a yaw reaction torque proportional to thrust through a moment constant, applied
with the rotor's spin direction.

**What it deliberately does NOT model** -- stated because an omission a reader has to discover by
being surprised is a trap:

* **blade flapping** -- the rotor disc tilting under translational inflow, which shows up as a
  pitch/roll moment proportional to airspeed. It matters for fast forward flight; it does not
  change hover, station-keeping or the low-speed manoeuvres this substrate is used for.
* **rotor drag / induced-drag terms** -- the lateral force proportional to rotor speed times body
  velocity that makes a real quad's velocity dynamics first-order rather than pure double
  integrator. MuJoCo's medium drag (``sim.physics`` density/viscosity) supplies body drag, but it
  is not the same term.
* **ground effect** -- the thrust increase within roughly one rotor diameter of the floor. An
  experiment about landing or low hover would notice.
* **battery sag** -- maximum thrust does not fall as the pack drains, so an endurance or
  payload-margin campaign gets a constant ceiling rather than a decaying one.

Config::

    multirotor_motors:
      robot: drone                # entity name registered by spawn_robot
      namespace: ""               # transport scope (default: inherited from spawn_robot)
      body: x500                  # root body the reaction torque acts on (default: entity's root)
      rotors: [rotor0_thrust, rotor1_thrust, rotor2_thrust, rotor3_thrust]
      spin: [1, 1, -1, -1]        # +1 = CCW seen from above, -1 = CW; PX4 quad-X order
      max_thrust: null            # N per rotor; default: read from each actuator's ctrlrange
      moment_constant: 0.05       # m, yaw torque per newton of thrust (PX4 CA_ROTOR*_KM)
      time_constant: 0.02         # s, first-order motor + ESC lag

**``max_thrust`` defaults to the model's own ``ctrlrange``**, not to a constant here. The MJCF is
the authority on its actuator limits; a hardcoded default would let a model and its actuation
plugin disagree about what "1.0" means, and the disagreement would show up as a thrust-to-weight
ratio nobody chose.

**``moment_constant`` is k_m/k_f**, the torque a rotor drags per newton of thrust it makes, in
metres. The default 0.05 is not a generic propeller estimate: it is PX4's published value for this
airframe, ``CA_ROTOR*_KM`` in ``4001_gz_x500``. **This number must agree between the airframe and
the flight stack's mixer.** PX4's control allocator inverts its own KM to decide how much
differential thrust a commanded yaw moment needs; if the simulator drags a different amount per
newton, every yaw command is scaled wrong -- sluggish or oscillatory yaw that reads as a badly tuned
rate loop rather than as two halves of the system disagreeing about a propeller. It is config here
rather than a gear in the MJCF because it is a motor/propeller property, so a re-propped airframe
changes it without touching the frame.

**The spin sign convention, which is the easy thing to get backwards.** A rotor spinning CCW (+1)
pushes air down and, by reaction, drags the *airframe* CW -- i.e. in -z. So the yaw torque a rotor
applies to the body is ``-spin_i * k_m * T_i`` about body z, and the contracted pattern
(+1, +1, -1, -1) sums to zero at equal thrust: two rotors of each handedness, which is the whole
reason a quad has two of each. Yaw is commanded by spinning up the pair of one handedness and down
the pair of the other, which is also why yaw authority is the weakest axis on a multirotor.

**The reaction torque is applied in the WORLD frame.** ``data.xfrc_applied`` is a cartesian
force/torque about the body CoM expressed in world coordinates, so the body-z torque is rotated by
the body's rotation before it is written. Writing a body-frame torque straight in is correct only
while the drone is level -- and it is wrong precisely when the drone is tilted, which is when yaw
control is being exercised.

**Air matters.** ``density``/``viscosity`` default to 0 in MuJoCo, so a world that does not set
them flies this drone through a vacuum: full rotor authority, no aerodynamic damping, and a
disturbance that never settles. The plugin warns rather than silently flying in vacuum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "rotors": ["rotor0_thrust", "rotor1_thrust", "rotor2_thrust", "rotor3_thrust"],
    #: PX4 quad-X: rotor0 front-right CCW, rotor1 rear-left CCW, rotor2 front-left CW,
    #: rotor3 rear-right CW. Diagonally opposite rotors share a handedness.
    "spin": [1, 1, -1, -1],
    "max_thrust": None,  # None => read the model's ctrlrange, per rotor
    "moment_constant": 0.05,
    "time_constant": 0.02,
}


@dataclass(frozen=True)
class MotorsHandle:
    """In-process handle other plugins find on the blackboard at ``motors:<robot>``.

    Deliberately narrow: a flight-stack bridge needs to *command* normalized outputs and to read
    back what the motors are actually doing after the lag, and nothing else. Passing the plugin
    itself would hand every consumer the whole lifecycle surface as well.
    """

    name: str
    count: int
    max_thrust: tuple[float, ...]
    set_normalized: Callable[[object], None]
    read_normalized: Callable[[], np.ndarray]


class MultirotorMotorsPlugin(Plugin):
    #: Drives an entity's actuators, so it belongs inside that entity's ``components:`` block.
    requires_owner = True

    def __init__(self, config=None, *, name=None, entity=None, label=None):
        super().__init__(config, name=name, entity=entity, label=label)
        self.robot = self.entity
        self._aids: list[int] = []
        self._bid = -1
        self._max_thrust = np.zeros(0)
        self._spin = np.array(self.cfg("spin"), dtype=float)
        self._cmd = np.zeros(len(self.cfg("rotors")))
        self._state = np.zeros(len(self.cfg("rotors")))

    def cfg(self, key):
        return self.config.get(key, _DEFAULTS[key])

    # -- config ----------------------------------------------------------------------------------

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        rotors = config.get("rotors", _DEFAULTS["rotors"])
        spin = config.get("spin", _DEFAULTS["spin"])
        if len(spin) != len(rotors):
            errors.append(
                f"'spin' has {len(spin)} entries but 'rotors' has {len(rotors)}: every rotor needs "
                f"a handedness, and a defaulted one would be a silently wrong yaw mixer"
            )
        if any(float(s) not in (1.0, -1.0) for s in spin):
            errors.append("'spin' entries must be +1 (CCW) or -1 (CW)")
        if float(config.get("time_constant", _DEFAULTS["time_constant"])) < 0:
            errors.append("'time_constant' must be >= 0 s")
        if float(config.get("moment_constant", _DEFAULTS["moment_constant"])) < 0:
            errors.append("'moment_constant' must be >= 0 m")
        if config.get("max_thrust") is not None and float(config["max_thrust"]) <= 0:
            errors.append("'max_thrust' must be > 0 N")
        return errors

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        model = ctx.model

        rotors = list(self.cfg("rotors"))
        self._spin = np.array([float(s) for s in self.cfg("spin")])
        if len(self._spin) != len(rotors):
            raise RuntimeError(
                f"multirotor_motors ({self.robot}): 'spin' has {len(self._spin)} entries for "
                f"{len(rotors)} rotors"
            )

        self._aids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + n) for n in rotors
        ]
        missing = [n for n, aid in zip(rotors, self._aids, strict=True) if aid < 0]
        if missing:
            raise RuntimeError(
                f"multirotor_motors ({self.robot}): could not resolve rotor actuators {missing}. "
                f"The model must expose one force actuator per rotor, in PX4 motor order."
            )

        body = self.config.get("body") or (entity.meta.get("root_body") if entity else None)
        self._bid = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + body) if body else -1
        )
        if self._bid < 0:
            # Fall back to the body owning rotor 0's site: that is where the thrust is applied and
            # therefore the body being flown, by construction.
            site = model.actuator_trnid[self._aids[0], 0]
            self._bid = int(model.site_bodyid[site]) if site >= 0 else -1
        if self._bid < 0:
            raise RuntimeError(
                f"multirotor_motors ({self.robot}): could not resolve the airframe body to apply "
                f"the rotor reaction torque to"
            )

        configured = self.config.get("max_thrust")
        if configured is None:
            # The model is the authority on its own actuator limits.
            self._max_thrust = np.array(
                [float(model.actuator_ctrlrange[a][1]) for a in self._aids]
            )
            if not np.all(self._max_thrust > 0):
                raise RuntimeError(
                    f"multirotor_motors ({self.robot}): the model's rotor actuators have no upper "
                    f"ctrlrange, so a normalized command has no scale. Give each rotor a ctrlrange "
                    f"in newtons, or set 'max_thrust' explicitly."
                )
        else:
            self._max_thrust = np.full(len(rotors), float(configured))

        self._cmd = np.zeros(len(rotors))
        self._state = np.zeros(len(rotors))

        # A vacuum is a silent, plausible-looking failure mode: the rotors still lift, but nothing
        # damps the airframe, so a disturbance rings forever and the run reads as bad flight-stack
        # gains rather than as a world with no air in it.
        if float(model.opt.density) == 0.0 and float(model.opt.viscosity) == 0.0:
            logger.warning(
                "multirotor_motors (%s): the world has no medium (density and viscosity are 0), "
                "so this drone is flying in a vacuum and has no aerodynamic damping. Set "
                "sim: {density: 1.225, viscosity: 1.8e-5} for air.",
                self.robot,
            )

        ctx.blackboard.set(
            f"motors:{self.robot}",
            MotorsHandle(
                name=self.robot,
                count=len(rotors),
                max_thrust=tuple(float(v) for v in self._max_thrust),
                set_normalized=self.set_normalized,
                read_normalized=self.read_normalized,
            ),
        )
        ctx.interface.add(
            Endpoint(
                name="motor_cmd",
                direction="in",
                owner=self.robot,
                namespace=ns,
                write=lambda msg: self.set_normalized(getattr(msg, "data", msg)),
                backend={
                    "ros2": {"type": "std_msgs.msg.Float32MultiArray", "topic": "motor_cmd"}
                },
            )
        )

    # -- commands --------------------------------------------------------------------------------

    def set_normalized(self, values) -> None:
        """Command the rotors with normalized 0..1 outputs, in the model's rotor order.

        This is PX4's unit (`HIL_ACTUATOR_CONTROLS`), and the reason the command is stored rather
        than written straight to ``data.ctrl``: a motor does not reach a new thrust in one tick, so
        the write happens in ``pre_step`` through the lag.
        """
        values = np.asarray(list(values), dtype=float)
        if values.size != self._cmd.size:
            raise ValueError(
                f"multirotor_motors ({self.robot}): expected {self._cmd.size} normalized commands, "
                f"got {values.size}"
            )
        self._cmd = np.clip(values, 0.0, 1.0)

    def read_normalized(self) -> np.ndarray:
        """The *lagged* rotor state, 0..1 -- what the motors are doing, not what was asked for."""
        return self._state.copy()

    # -- lifecycle -------------------------------------------------------------------------------

    def on_reset(self, ctx: SimContext) -> None:
        self._cmd = np.zeros_like(self._cmd)
        self._state = np.zeros_like(self._state)
        if ctx.data is not None and self._bid >= 0:
            # A stale external force surviving a reset is a real bug class: the next episode starts
            # with a torque nobody commanded, and it looks like a physics difference between
            # repetitions rather than like leftover state.
            ctx.data.xfrc_applied[self._bid] = 0.0
            for aid in self._aids:
                ctx.data.ctrl[aid] = 0.0

    def pre_step(self, ctx: SimContext) -> None:
        model, data = ctx.model, ctx.data
        dt = float(ctx.dt)
        tau = float(self.cfg("time_constant"))

        # Exponential-style first-order lag: alpha = dt / (tau + dt), not the naive dt/tau. The
        # naive form is the forward-Euler discretisation and is only stable for dt < 2*tau -- with
        # tau = 0.02 s a world stepping at 10 ms is already marginal and one stepping at 50 ms
        # oscillates and diverges. dt/(tau+dt) is the backward-Euler form: it stays in (0, 1] for
        # every positive dt, so a coarse world degrades to "the motors respond immediately", which
        # is wrong but bounded, instead of exploding.
        alpha = 1.0 if tau <= 0.0 else dt / (tau + dt)
        self._state = self._state + alpha * (self._cmd - self._state)

        thrust = self._state * self._max_thrust
        for i, aid in enumerate(self._aids):
            lo, hi = model.actuator_ctrlrange[aid]
            data.ctrl[aid] = float(np.clip(thrust[i], lo, hi))

        # Reaction torque: a CCW (+1) rotor drags the airframe CW, so the body torque is
        # -spin * k_m * T. Summed over rotors, then rotated into the world frame, because
        # xfrc_applied is expressed there (see the module docstring).
        tau_z = float(-np.sum(self._spin * float(self.cfg("moment_constant")) * thrust))
        rot = np.array(data.xmat[self._bid]).reshape(3, 3)
        data.xfrc_applied[self._bid, 3:6] = rot @ np.array([0.0, 0.0, tau_z])
