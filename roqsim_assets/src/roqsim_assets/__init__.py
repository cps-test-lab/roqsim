"""roqsim_assets: shared, reusable surface textures for roqsim scenes.

A single home for texture assets so scene/floor plugins across packages (the mobile ``floorplan``,
the ``roqsim_scenes`` ``static_scene``, ...) share one copy instead of each vendoring their own.

Assets live one-per-folder under ``assets/<Name>/`` (a single Color PNG + an optional
``manifest.yaml`` with ``reflectance`` / ``physical_size``). This module exposes ``ASSETS_DIR``; a
plugin references a texture explicitly as ``roqsim_assets:<Name>`` (resolved by
:func:`roqsim.textures.resolve_texture`, which imports this module and reads ``ASSETS_DIR``).
"""

import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
