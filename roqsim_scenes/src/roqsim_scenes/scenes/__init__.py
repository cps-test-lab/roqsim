"""Bundled imported scenes (``<name>/scene.json`` + ``scene.yaml`` + ``meshes/*.obj``).

``SCENES_DIR`` lets the pipeline (:mod:`roqsim_scenes.cli.scene_to_mjcf`) locate a bundled scene by name.
"""

import os

SCENES_DIR = os.path.dirname(__file__)
