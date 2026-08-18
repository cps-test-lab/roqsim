"""Asset provider for the ``roqsim.models`` entry-point group: reusable **props**.

Props are small, static MJCFs (furniture, equipment) imported by ``sketchfab_helper import`` and laid
out one-per-folder as ``models/<name>/<name>.xml`` -- a self-contained MJCF (its own ``meshdir``
resolves the sibling mesh + PNG textures). Registering this directory as a model provider makes each
prop resolvable by short name (:func:`roqsim.models.resolve_model`), so a world YAML can place it
with the generic ``spawn_model`` plugin instead of only ``<include>``-ing it into a baked scene.

``MODELS_DIR`` points at this nested layout; the model resolver accepts the ``<name>/<name>.xml``
form (as it does for baked worlds).
"""

from pathlib import Path

MODELS_DIR = Path(__file__).parent
