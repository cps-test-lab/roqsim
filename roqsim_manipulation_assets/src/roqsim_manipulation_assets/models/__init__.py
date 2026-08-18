"""Bundled arm/gripper MJCF models and meshes, one folder per model.

Layout is ``models/<name>/<name>.xml`` with that model's meshes in its own ``meshes/`` subdir and its
manifest, licence, port log and thumbnail beside it -- the same one-folder-per-model shape
``roqsim_assets`` uses for props, and a form :func:`roqsim.models.resolve_model` accepts directly. The
alternative (every MJCF flat in this directory over one shared ``meshes/``) is what this replaced: a
model's files were scattered across four globs, and Menagerie link meshes with generic names
(``base_0.obj``, ``link1.stl``) could only be kept apart by a per-model mesh subdirectory anyway. A
folder makes the grouping the filesystem's job, so adding or removing a model is one ``mv``.

Each ships a ``<model>.manifest.yaml`` listing the plugins intrinsic to it -- for an arm that is an
``arm_controller`` carrying its joints, gains and (where it has a hand) the gripper joint and the
open/closed values in that joint's own units. ``spawn_arm`` pulls it in via
:func:`roqsim.manifest.expand_manifest`, and ``spawn_arm``'s ``end_effector:`` merges a gripper model's
manifest on top -- which is how a bare arm and a separate hand reach the same state as a
factory-assembled one.

Grippers here are standalone and attachable (``robotiq_2f85``, ``schunk_pg70``): root body at the
origin with the fingers along +z, the tool-flange convention MuJoCo Menagerie arms use for their
``attachment_site``, so welding one onto an arm needs no rotation.

``MESHES_DIR`` is this directory rather than a ``meshes/`` child, because with per-model folders the
root a mesh reference resolves against IS the models root: a model of its own reaches its meshes
through ``<compiler meshdir="meshes">``, and anything reaching ACROSS models -- ``gen3`` borrowing the
2F-85's meshes, or another package's model borrowing this provider via ``assets:
[roqsim_manipulation_assets]`` -- names them ``<model>/meshes/<file>``. Getting this wrong is silent: the
mesh simply fails to resolve and MuJoCo compiles the model with the reference as given.
"""

from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).parent
MESHES_DIR = MODELS_DIR


def model_path(name: str) -> Path:
    """Resolve a bundled model file (accepts a bare name like ``ur10e`` or a filename)."""
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
