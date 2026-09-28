"""What a compiled flex is, and how it will behave: the analysis ``roqsim check`` prints.

Everything here reads a compiled ``MjModel`` and steps nothing. It answers the questions an author
of a ``<flexcomp>`` otherwise answers by running the world and watching it: what the flex compiled
into, how fast it rings, how much of its damping is the material's and how much the integrator's,
and whether its contact stiffness is the one MuJoCo will use.

Three answers are rules of MuJoCo's integrator rather than of the model. Only this analysis reads
them, so they are stated where they are used rather than among :mod:`roqsim.flex`'s numbered rules:
**numerical damping under** ``discrete`` and **the resolution limit** in :func:`explain_flex`, and
**the** ``solref`` **floor** in :func:`solref_floor`. Each was measured on MuJoCo 3.14.0 and is
pinned by ``tests/test_flex_modes.py``, which measures it on a stepped model.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from .flex import flex_dof_body_ids

#: ``flex_interp`` -> the ``dof`` attribute a ``<flexcomp>`` states it with.
DOF_MODES = {0: "full", 1: "trilinear", 2: "quadratic"}

#: Above this many degrees of freedom :func:`first_modes` refuses rather than spend the time. The
#: stiffness costs two passive-force evaluations of the model per degree of freedom and the eigen
#: solve grows with the cube of the count; at the cap the analysis takes seconds, and a cubic cost is
#: not something ``roqsim check`` should spend unasked on a fine grid.
MODES_DOF_CAP = 2400

#: A generalized eigenvalue below this fraction of the largest is a rigid-body mode, not an elastic
#: one. Measured: a free block's six rigid modes land at 1e-16 (translation) to 1e-10 (rotation,
#: the finite-difference step) of the largest, its first elastic mode at 1e-3, and the gap widens
#: only slowly with a finer grid.
_RIGID_TOL = 1e-8

#: Finite-difference step for the stiffness, in the DOF's own unit (metres for a vertex slide joint).
_FD_STEP = 1e-6

#: The largest ``omega * timestep`` at which ``roqsim check``'s figures for a mode are the ones a
#: ``discrete`` run shows: at it, a mode damped at ``zeta`` up to 0.3 rings within 5 % of the
#: reported damping ratio and frequency. Above it the mode is under-resolved by the timestep --
#: the resolution limit in :func:`explain_flex`, where the measurement is.
MAX_OMEGA_DT = 0.3


def _round_down(value: float, digits: int = 3) -> float:
    """*value* rounded down to *digits* significant figures, so a stated bound is never above it."""
    import math

    exponent = math.floor(math.log10(value)) - digits + 1
    return round(math.floor(value / 10.0**exponent) * 10.0**exponent, -exponent)


class FlexTooLarge(ValueError):
    """:func:`first_modes` was asked about a flex above :data:`MODES_DOF_CAP` degrees of freedom."""

    def __init__(self, name: str, ndof: int):
        self.hint = (
            'use a coarser grid (fewer vertices in the <flexcomp> count), dof="quadratic" (27 '
            'nodes) or dof="trilinear" (8 nodes), or measure the modes from a free-vibration run'
        )
        super().__init__(
            f"flex {name!r} has {ndof} degrees of freedom, above the {MODES_DOF_CAP} its modes are "
            "computed for"
        )


@dataclass(frozen=True)
class FlexModes:
    """The lowest elastic modes of one flex, with every other degree of freedom held."""

    #: Angular frequencies, rad/s, ascending.
    omega: tuple[float, ...]
    #: Degrees of freedom the flex has of its own (its unpinned vertices', or nodes').
    ndof: int
    #: Modes dropped as rigid-body motion (a free flex has six).
    rigid: int
    #: With ``shapes=True``: the model DOF indices the shapes are over, and the mode shapes as the
    #: columns of an ``(ndof, n)`` array, mass-normalised (``v.T @ M @ v == 1``). Otherwise ``None``.
    dofs: object = None
    shapes: object = None

    @property
    def hz(self) -> tuple[float, ...]:
        import math

        return tuple(w / (2 * math.pi) for w in self.omega)


def _slice(values, addresses, index: int):
    """Flex *index*'s rows of a per-flex array. An address of -1 means the flex has none."""
    start = int(addresses[index])
    if start < 0:
        return values[:0]
    later = [int(a) for a in addresses[index + 1 :] if int(a) > start]
    return values[start : min(later) if later else len(values)]


def flex_name(model: mujoco.MjModel, flex_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, flex_id) or f"#{flex_id}"


def flex_parent(model: mujoco.MjModel, flex_id: int) -> tuple[int, int]:
    """``(parent body id, pinned count)``: the body the flex was declared in, and how many anchors sit on it.

    A ``<flexcomp>`` gives each free vertex (node) a body of its own under the body it is declared
    in, and puts a pinned one on that body itself, which is how the two are told apart here. A rigid
    flex has every anchor on one body, which is then its parent and all of them count as pinned.
    """
    anchors = flex_dof_body_ids(model, flex_id)
    parents = {int(model.body_parentid[b]) for b in anchors}
    own = [b for b in anchors if b not in parents]
    if not own:
        return anchors[0], len(anchors)
    candidates = [int(model.body_parentid[b]) for b in own]
    parent = max(set(candidates), key=candidates.count)
    return parent, sum(1 for b in anchors if b == parent)


def flex_dofs(model: mujoco.MjModel, flex_id: int):
    """The model DOF indices that are this flex's own: its unpinned anchors' joints."""
    import numpy as np

    parent, _ = flex_parent(model, flex_id)
    dofs: list[int] = []
    for body in dict.fromkeys(flex_dof_body_ids(model, flex_id)):
        if body == parent:
            continue
        start = int(model.body_dofadr[body])
        dofs.extend(range(start, start + int(model.body_dofnum[body])))
    return np.array(dofs, dtype=int)


def is_elastic(model: mujoco.MjModel, flex_id: int) -> bool:
    """Whether the flex resists deformation with a passive force (elasticity, bending, or edge stiffness).

    Read off the compiled fields MuJoCo computes it from: ``young`` survives compile only as
    ``flex_stiffness`` (a solid, a shell with ``elastic2d`` stretch) or ``flex_bending`` (a shell
    with bending); an edge spring is ``flex_edgestiffness``. A rigid flex does not deform.
    """
    if bool(model.flex_rigid[flex_id]):
        return False
    stiffness = _slice(model.flex_stiffness, model.flex_stiffnessadr, flex_id)
    bending = _slice(model.flex_bending, model.flex_bendingadr, flex_id)
    return bool(
        (len(stiffness) and abs(stiffness).max() > 0)
        or (len(bending) and abs(bending).max() > 0)
        or model.flex_edgestiffness[flex_id] > 0
    )


def _owner(model: mujoco.MjModel, body: int, entity_bodies: dict[str, str]) -> str | None:
    """The entity whose body is *body* or its nearest ancestor."""
    by_body: dict[str, str] = {}
    for entity, body_name in sorted(entity_bodies.items()):
        if body_name:
            by_body.setdefault(body_name, entity)
    while True:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if name in by_body:
            return by_body[name]
        if body == 0:
            return None
        body = int(model.body_parentid[body])


def describe_flexes(
    model: mujoco.MjModel, entity_bodies: dict[str, str] | None = None
) -> list[dict]:
    """One row per flex: what it compiled into. Reads fields only; costs nothing.

    *entity_bodies* maps entity name to its body name; a flex's ``entity`` is the one whose body is
    the flex's parent or its nearest ancestor, ``None`` without one.
    """
    rows = []
    for f in range(model.nflex):
        parent, pinned = flex_parent(model, f)
        interp = int(model.flex_interp[f])
        rows.append(
            {
                "name": flex_name(model, f),
                "dim": int(model.flex_dim[f]),
                "vertices": int(model.flex_vertnum[f]),
                "elements": int(model.flex_elemnum[f]),
                "dof": DOF_MODES.get(interp, str(interp)),
                "nodes": int(model.flex_nodenum[f]),
                # Of its anchors -- the vertices, or the nodes under trilinear/quadratic.
                "pinned": pinned,
                "parent": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) or "world",
                "entity": _owner(model, parent, entity_bodies or {}),
                "rigid": bool(model.flex_rigid[f]),
                "elastic": is_elastic(model, f),
                "passive_contact": bool(model.flex_passive[f]),
                "ndof": int(len(flex_dofs(model, f))),
            }
        )
    return rows


def _passive(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """The part of ``mj_forward`` that ``qfrc_passive`` depends on at rest -- no collision, no solve."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_flex(model, data)
    mujoco.mj_tendon(model, data)
    mujoco.mj_passive(model, data)


def first_modes(
    model: mujoco.MjModel, flex_id: int, n: int = 3, *, shapes: bool = False
) -> FlexModes:
    """The *n* lowest elastic modes of flex *flex_id*, at rest, with every other DOF held.

    The stiffness ``K`` is the central difference of ``qfrc_passive`` over the flex's own DOFs at
    ``qpos0``, symmetrised; the mass ``M`` is that block of ``mj_fullM``. ``K v = omega^2 M v`` is
    solved through a Cholesky factor of ``M`` and a symmetric eigen solve (numpy only), and rigid-
    body modes (eigenvalues below ``_RIGID_TOL`` of the largest) are dropped. What it measures is
    the passive stiffness only: a flex held in shape by equality constraints (``edge equality``) has
    none, and returns no modes.

    With *shapes*, the mode shapes come back too (a full eigen solve rather than eigenvalues only).
    Refuses above :data:`MODES_DOF_CAP` DOFs with :class:`FlexTooLarge`, whose ``hint`` says how to
    get under it. Validated against a free-vibration measurement in ``tests/test_flex_modes.py``.
    """
    import numpy as np

    dofs = flex_dofs(model, flex_id)
    if len(dofs) > MODES_DOF_CAP:
        raise FlexTooLarge(flex_name(model, flex_id), len(dofs))
    if not len(dofs):
        return FlexModes((), 0, 0)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    qpos_of = model.jnt_qposadr[model.dof_jntid[dofs]]
    stiffness = np.zeros((len(dofs), len(dofs)))
    for column, address in enumerate(qpos_of):
        data.qpos[address] = model.qpos0[address] + _FD_STEP
        _passive(model, data)
        plus = data.qfrc_passive[dofs].copy()
        data.qpos[address] = model.qpos0[address] - _FD_STEP
        _passive(model, data)
        minus = data.qfrc_passive[dofs].copy()
        data.qpos[address] = model.qpos0[address]
        stiffness[:, column] = -(plus - minus) / (2 * _FD_STEP)
    stiffness = 0.5 * (stiffness + stiffness.T)

    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    inv_chol = np.linalg.inv(np.linalg.cholesky(full[np.ix_(dofs, dofs)]))
    reduced = inv_chol @ stiffness @ inv_chol.T
    reduced = 0.5 * (reduced + reduced.T)
    if shapes:
        eigenvalues, vectors = np.linalg.eigh(reduced)
    else:
        eigenvalues, vectors = np.linalg.eigvalsh(reduced), None

    top = float(eigenvalues.max())
    keep = eigenvalues > _RIGID_TOL * top if top > 0 else np.zeros(len(eigenvalues), dtype=bool)
    chosen = np.flatnonzero(keep)[:n]
    omega = tuple(float(np.sqrt(eigenvalues[i])) for i in chosen)
    rigid = int(len(eigenvalues) - keep.sum())
    if not shapes:
        return FlexModes(omega, len(dofs), rigid)
    # K v = w^2 M v with v = L^-T u, where u are the eigenvectors of L^-1 K L^-T: mass-normalised.
    return FlexModes(omega, len(dofs), rigid, dofs, inv_chol.T @ vectors[:, chosen])


def solref_floor(model: mujoco.MjModel) -> float | None:
    """The time constant below which MuJoCo raises a contact's ``solref``, ``None`` if there is none.

    **The solref floor.** With ``refsafe`` enabled (MuJoCo's default) a contact's time constant is
    raised to a floor before it is used, so a ``solref`` stiffer than the floor is silently not the
    one that runs. The floor is **one** timestep under ``discrete`` and two under every other integrator;
    with ``refsafe`` disabled there is none. Measured as the depth a resting sphere sits at: equal
    for every time constant below the floor, deeper above it. A ``solref`` whose first entry is not
    positive states stiffness and damping directly and has no floor.
    """
    if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE:
        return None
    steps = 1 if model.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE else 2
    return steps * float(model.opt.timestep)


def contact_solref(model: mujoco.MjModel, flex_id: int) -> tuple[list[float], str]:
    """The ``solref`` this flex's contacts start from, and where it comes from.

    ``sim.contact_override`` (MuJoCo's ``o_solref`` under the override flag) replaces every
    contact's; otherwise it is the flex's own, which MuJoCo then mixes with the other side's by
    ``solmix``.
    """
    if model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_OVERRIDE:
        return [float(v) for v in model.opt.o_solref], "sim.contact_override"
    return [float(v) for v in model.flex_solref[flex_id]], "flex"


def explain_flex(model: mujoco.MjModel, flex_id: int, n: int = 3) -> tuple[dict, list[dict]]:
    """``(derived, warnings)`` for one flex: its modes, damping, and contact, as ``roqsim check`` reports them.

    ``derived`` carries ``modes`` (``None`` with ``modes_skipped`` and a ``hint`` above the DOF
    cap, ``[]`` for a flex that is not elastic), each with ``hz``, the effective damping ratio
    ``zeta`` and its ``zeta_numerical`` share, its ``omega_dt`` (``omega * timestep``) and whether
    it is ``resolved`` (the resolution limit, below: ``False`` means its ``hz`` and ``zeta`` are
    not what a run shows); the flex's ``damping``, the ``timestep``, the ``numerical_share`` of the
    damping, the ``max_timestep_for_half`` that keeps it at most half and the
    ``max_timestep_resolved`` that resolves every reported mode; and the contact ``solref``, its
    ``solref_floor`` and whether it is ``below_floor``. The numerical terms and the resolution are
    those of the ``discrete`` integrator (below) and ``None`` under any other. ``warnings`` are
    ``roqsim check`` warnings -- ``{"check", "message", "hint"}``, with the flex's name in the
    message and, as an extra key, in ``flex`` -- and never make a world fail. ``check`` is
    ``flex-damping`` (the integrator's share of the damping is above half), ``flex-timestep`` (a
    reported mode is under-resolved by the timestep) or ``flex-solref`` (the contact ``solref`` is
    below the floor).

    **Numerical damping under ``discrete``.** The integrator damps a flex's elastic mode *i*
    as if the stated Rayleigh damping (``<elasticity damping>``, a time) were one timestep larger:
    ``zeta_i = (damping + timestep) * omega_i / 2``. The integrator's share of the damping is
    therefore ``timestep / (damping + timestep)``, the same for every mode, and it is at most half
    exactly when ``timestep <= damping``. It is a small-step relation, as ``hz`` is: both hold while
    ``omega_i * timestep`` is small, which is the resolution limit.

    **The resolution limit.** As ``omega * timestep`` grows, the ``discrete`` integrator damps a
    mode less and rings it slower than ``zeta`` and ``hz`` say. Measured on a pinned block's first
    mode, excited alone and read from the poles of its sampled ring-down, with the damping one
    timestep (so ``zeta = omega * timestep``): the damping ratio falls short by 0.5 % at
    ``omega * timestep`` 0.1, 2 % at 0.2, 4.4 % at 0.3, 7 % at 0.4, 11 % at 0.5 and 23 % at 0.85,
    and the natural frequency by 0.5 %, 1.8 %, 3.8 %, 6 %, 9 % and 20 %. With no stated damping the
    shortfall is about half that; at a fixed ``omega * timestep`` it grows with ``zeta`` (at 0.3:
    2.4 % at ``zeta`` 0.15, 7 % at 0.5, 11 % at 0.9). Hence :data:`MAX_OMEGA_DT` of 0.3: at or below
    it, a mode damped at ``zeta`` up to 0.3 is within 5 % of both figures; above it the mode is
    under-resolved, marked ``resolved: False``, and warned about. Only the reported modes are
    judged; a flex's higher modes ring faster still and are not reported. A coarser grid does
    not lower the lowest modes (it raises them slightly); a softer material does, ``omega`` scaling
    with the square root of ``<elasticity young>``.
    """
    import math

    name = flex_name(model, flex_id)
    timestep = float(model.opt.timestep)
    damping = float(model.flex_damping[flex_id])
    discrete = model.opt.integrator == mujoco.mjtIntegrator.mjINT_DISCRETE
    derived: dict = {"name": name, "damping": damping, "timestep": timestep}
    warnings: list[dict] = []

    modes = None
    if is_elastic(model, flex_id):
        try:
            modes = first_modes(model, flex_id, n)
        except FlexTooLarge as exc:
            derived["modes_skipped"] = str(exc)
            derived["hint"] = exc.hint
    else:
        modes = FlexModes((), len(flex_dofs(model, flex_id)), 0)
    if modes is None:
        derived["modes"] = None
    else:
        numerical = timestep if discrete else 0.0
        derived["modes"] = [
            {
                "hz": w / (2 * math.pi),
                "zeta": (damping + numerical) * w / 2,
                "zeta_numerical": numerical * w / 2 if discrete else None,
                "omega_dt": w * timestep,
                "resolved": w * timestep <= MAX_OMEGA_DT if discrete else None,
            }
            for w in modes.omega
        ]
    if discrete:
        derived["numerical_share"] = timestep / (damping + timestep)
        derived["max_timestep_for_half"] = damping if damping > 0 else None
        derived["max_timestep_resolved"] = (
            _round_down(MAX_OMEGA_DT / max(modes.omega)) if modes and modes.omega else None
        )
    else:
        derived["numerical_share"] = None
        derived["max_timestep_for_half"] = None
        derived["max_timestep_resolved"] = None

    if discrete and derived["modes"] and derived["numerical_share"] > 0.5:
        first = derived["modes"][0]
        if damping > 0:
            hint = (
                f"set sim.timestep <= {damping:g} s (the flex's damping) to keep the integrator's "
                "share at most half, or raise <elasticity damping>"
            )
        else:
            hint = (
                "state <elasticity damping> (a time, s) and keep sim.timestep at or below it: "
                "zeta_i = (damping + timestep) * omega_i / 2, with omega_1 = "
                f"{2 * math.pi * first['hz']:.4g} rad/s"
            )
        warnings.append(
            {
                "check": "flex-damping",
                "message": (
                    f"flex {name!r}: numerical damping is {derived['numerical_share']:.0%} of its "
                    "damping "
                    f"(zeta_1 = {first['zeta']:.3g}, of which {first['zeta_numerical']:.3g} is the "
                    f"discrete integrator's timestep * omega / 2), so its ringing changes with "
                    "sim.timestep"
                ),
                "hint": hint,
                "flex": name,
            }
        )

    coarse = [i for i, mode in enumerate(derived["modes"] or []) if mode["resolved"] is False]
    if coarse:
        first = derived["modes"][coarse[0]]
        others = [str(i + 1) for i in coarse[1:]]
        also, them, their = "", "it", "its"
        if others:
            them, their = "them", "their"
            also = (
                f", and so {'are modes' if len(others) > 1 else 'is mode'} {' and '.join(others)}"
            )
        warnings.append(
            {
                "check": "flex-timestep",
                "message": (
                    f"flex {name!r}: mode {coarse[0] + 1} ({first['hz']:.3g} Hz) is under-resolved "
                    f"by the timestep (omega * timestep = {first['omega_dt']:.2g}, above "
                    f"{MAX_OMEGA_DT:g}){also}; the damping ratio and frequency reported for "
                    f"{them} are not the ones that run: the discrete integrator distorts {their} "
                    f"dynamics, damping {them} less and ringing {them} slower than stated"
                ),
                "hint": (
                    f"set sim.timestep <= {derived['max_timestep_resolved']:g} s to bring omega * "
                    f"timestep to {MAX_OMEGA_DT:g} or below for every reported mode, or lower "
                    "the frequencies with a softer material (omega scales with the square root of "
                    "<elasticity young>)"
                ),
                "flex": name,
            }
        )

    solref, source = contact_solref(model, flex_id)
    floor = solref_floor(model)
    derived["solref"] = solref
    derived["solref_source"] = source
    derived["solref_floor"] = floor
    derived["below_floor"] = bool(floor is not None and 0 < solref[0] < floor)
    if derived["below_floor"]:
        integrator = mujoco.mjtIntegrator(model.opt.integrator).name.removeprefix("mjINT_").lower()
        steps = "one timestep" if discrete else "two timesteps"
        warnings.append(
            {
                "check": "flex-solref",
                "message": (
                    f"flex {name!r}: contact solref time constant {solref[0]:g} s ({source}) is below MuJoCo's "
                    f"floor of {floor:g} s ({steps} under {integrator}); MuJoCo uses the floor, so "
                    "the stated contact stiffness is not the one that runs"
                ),
                "hint": f"raise solref[0] to at least {floor:g} s, or lower sim.timestep",
                "flex": name,
            }
        )
    return derived, warnings
