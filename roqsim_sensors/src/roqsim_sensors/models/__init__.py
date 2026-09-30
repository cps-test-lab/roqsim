"""Bundled sensor MJCF models and meshes, one folder per model.

Layout is ``models/<name>/<name>.xml`` with that model's meshes in its own ``meshes/`` subdir --
reached bare through ``<compiler meshdir="meshes">`` -- and its manifest, licence and thumbnail
beside it. Same shape as ``roqsim_manipulation_assets`` and ``roqsim_assets``, and a form
:func:`roqsim.models.resolve_model` accepts directly.

Not every MJCF flat in this directory over one shared ``meshes/``, for two reasons specific to
sensors: there a device's mesh files could only be kept apart from another's by prefixing each one
(``mid360_body.obj``, ``zivid_body.obj`` -- the folder does that job), and three of the six models
have meshes that are *generated, not committed* (see ``external/external_assets.yaml``), so a pooled
dir would mix tracked and git-ignored files with nothing in the path to say which was which. A
folder makes the grouping the filesystem's job: adding or removing a sensor is one ``mv``, and its
licence sidecar travels with the mesh it covers.

``MESHES_DIR`` is this directory rather than a ``meshes/`` child, because with per-model folders the
root a mesh reference resolves against IS the models root: a model reaches its own meshes through its
``meshdir``, and anything reaching ACROSS models -- another package's model borrowing this provider
via ``assets: [roqsim_sensors]`` for a camera mesh -- names them ``<model>/meshes/<file>``. Getting this
wrong is silent: the mesh simply fails to resolve and MuJoCo compiles the model with the reference as
given (``tests/test_sensor_model_layout.py`` is what catches it).
"""

from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).parent
MESHES_DIR = MODELS_DIR
