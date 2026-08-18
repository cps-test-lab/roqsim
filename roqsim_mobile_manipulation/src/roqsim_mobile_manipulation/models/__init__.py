"""Bundled mobile-manipulator MJCF models and meshes, one folder per model.

Layout is ``models/<name>/<name>.xml`` with that model's meshes in its own ``meshes/`` subdir --
referenced bare via ``<compiler meshdir="meshes">`` -- and its manifest, licence, port log and
thumbnail beside it. The same shape ``roqsim_mobile``, ``roqsim_manipulation_assets``, ``roqsim_sensors`` and
``roqsim_assets`` use, and a form :func:`roqsim.models.resolve_model` accepts directly.

Each model ships a ``<model>.manifest.yaml`` listing the plugins intrinsic to it -- for these robots
that is a drive (``diff_drive`` / ``omni_drive``), one ``arm_controller`` per arm, and a ``lidar``;
``spawn_robot`` pulls them in via :func:`roqsim.manifest.expand_manifest`.

The manifests name their ``joints:`` explicitly rather than letting ``arm_controller`` scan by prefix.
That is load-bearing on a mobile manipulator: the prefix scan would also claim the wheel motors and
write arm position targets into drives another plugin owns, and the robot then will not move at all.
"""

from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).parent
#: This directory, not a ``meshes/`` child: with per-model folders the root a mesh reference resolves
#: against IS the models root. A model of its own reaches its meshes through its ``meshdir``, and
#: anything reaching ACROSS models -- another package borrowing this provider via ``assets:`` -- names
#: them ``<model>/meshes/<file>``. Getting this wrong is silent: MuJoCo does not error on an
#: unresolvable mesh path, so the robot loses its geometry rather than failing to compile.
MESHES_DIR = MODELS_DIR


def model_path(name: str) -> Path:
    """Resolve a bundled model file (accepts a bare name like ``frankie`` or a filename)."""
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
