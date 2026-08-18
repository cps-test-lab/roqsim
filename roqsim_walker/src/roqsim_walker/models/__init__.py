"""Asset provider for the ``roqsim.models`` entry-point group.

Unlike the MJCF-based packages, a walker has no ``<model>.xml``: its bodies and skin are generated
programmatically at build time from the blueprint folder (``people/<Walker>/`` -- a textured OBJ plus
its ``*.walker.json`` sidecar) and the locomotion clips in ``anims/<set>/``. ``MODELS_DIR`` is what
:mod:`roqsim_walker.blueprint` resolves those against.
"""

from pathlib import Path

MODELS_DIR = Path(__file__).parent
MESHES_DIR = MODELS_DIR / "people"
TEXTUREDIR = MODELS_DIR / "people"
