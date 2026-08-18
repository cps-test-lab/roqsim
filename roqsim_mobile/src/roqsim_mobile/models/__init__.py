"""Bundled wheeled-base MJCF models and meshes, one folder per model.

Layout is ``models/<name>/<name>.xml`` with that model's meshes in its own ``meshes/`` subdir --
referenced bare via ``<compiler meshdir="meshes">`` -- and its manifest, licence, port log and
thumbnail beside it. The same shape ``roqsim_manipulation_assets`` and ``roqsim_assets`` use, and a form
:func:`roqsim.models.resolve_model` accepts directly. This replaced a flat ``models/<name>.xml`` over one
shared ``models/meshes/``, where a model's files were spread across four globs and two robots'
same-named link meshes could only be kept apart by a per-model mesh subdirectory anyway. A folder makes
the grouping the filesystem's job, so adding or removing a model is one ``mv``.

Each robot ships a ``<model>.manifest.yaml`` listing the plugins intrinsic to it -- for a wheeled base
that is its drive (``diff_drive`` / ``omni_drive``) carrying the platform's wheel geometry and
kinematic limits, plus its stock sensors. ``spawn_robot`` pulls it in via
:func:`roqsim.manifest.expand_manifest`, so a world spawns the model instead of re-declaring its
controller.

``MESHES_DIR`` is this directory rather than a ``meshes/`` child, because with per-model folders the
root a mesh reference resolves against IS the models root: a model of its own reaches its meshes
through its ``meshdir``, and anything reaching ACROSS models -- another package's model borrowing this
provider via ``assets: [roqsim_mobile]`` -- names them ``<model>/meshes/<file>``. Getting this wrong is
silent: MuJoCo does not error on an unresolvable mesh path, it compiles the model with the reference as
given, so the robot loses its geometry rather than failing. ``tests/test_model_layout.py`` is what catches it.

``floor`` is the one entry here that is not a robot: a bare checker ground plane whose geom is named
``floor``, carried as a model so a scene can include it by name.
"""

from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).parent
MESHES_DIR = MODELS_DIR


def model_path(name: str) -> Path:
    """Resolve a bundled model file (accepts a bare name like ``turtlebot4`` or a filename)."""
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for candidate in (
        MODELS_DIR / name,
        MODELS_DIR / f"{name}.xml",
        MODELS_DIR / name / f"{name}.xml",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model {name!r} not found under {MODELS_DIR}")
