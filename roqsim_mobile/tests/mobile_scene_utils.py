"""Shared helpers for the mobile-base scene tests.

Deliberately **not** ``conftest.py``. Importing a helper with ``from conftest import ...`` looks
natural and works when one package's tests run alone, then breaks the moment the whole suite does:
several packages here have a ``conftest.py``, the name is ambiguous on ``sys.path``, and whichever
one is imported first wins. It failed with "cannot import name from
roqsim_sensors/tests/conftest.py". A conftest is for fixtures pytest injects, not a module to import
by name.

The one thing here exists because of a defect it would have caught twice in one session.

``mujoco.mj_name2id`` returns **-1** for a name that is not in the model, and ``-1`` is a perfectly
good Python index. So ``model.geom_priority[mj_name2id(m, GEOM, "typo")]`` silently reads the *last*
geom, and ``"absent_geom" not in touching`` is trivially true. Both failure modes assert something
about a geom that does not exist and pass.

That is not hypothetical. Porting the Maker's Pet Loki, the root link's own body cylinder -- the
largest part of the robot, visual and collision -- was never emitted at all, and the test asserting
"the body is not dragging on the floor" passed *because* the geom was missing. Minutes earlier, a
caster renamed by its contact class made a priority assertion read a different geom entirely.

:func:`named` turns both into an immediate, legible failure.
"""

from __future__ import annotations

import mujoco
import pytest


def named(model, objtype: int, name: str) -> int:
    """The id of *name*, failing the test if the model has no such object.

    Use this instead of ``mj_name2id`` anywhere the result is used as an index or in a membership
    test -- which is everywhere. A missing name is a broken test or a broken model, never a pass.
    """
    ident = mujoco.mj_name2id(model, objtype, name)
    if ident < 0:
        kind = {
            mujoco.mjtObj.mjOBJ_BODY: "body",
            mujoco.mjtObj.mjOBJ_GEOM: "geom",
            mujoco.mjtObj.mjOBJ_SITE: "site",
            mujoco.mjtObj.mjOBJ_JOINT: "joint",
            mujoco.mjtObj.mjOBJ_ACTUATOR: "actuator",
        }.get(objtype, str(objtype))
        available = sorted(
            filter(None, (mujoco.mj_id2name(model, objtype, i)
                          for i in range(_count(model, objtype))))
        )
        pytest.fail(
            f"no {kind} named {name!r} in this model -- mj_name2id would have returned -1 and the "
            f"assertion would have silently used the LAST {kind}. Available: {', '.join(available)}"
        )
    return ident


def _count(model, objtype: int) -> int:
    return {
        mujoco.mjtObj.mjOBJ_BODY: model.nbody,
        mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
        mujoco.mjtObj.mjOBJ_SITE: model.nsite,
        mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
        mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
    }.get(objtype, 0)
