"""Roqsim loading overlay drawn *inside* the MuJoCo viewer window (not a separate window).

The passive viewer exposes ``set_images``/``set_texts``/``clear_*`` on its handle; we centre the
splash art (logo + toolkit text) on a full-viewport image and lay a "Loading…" line under it, then
clear both once the scene's first frame is up. The runner opens the window on an empty placeholder
first, so this covers the whole world compile, not just the first-frame mesh/texture upload.

Cosmetic and dependency-soft: it needs ``numpy`` (always present with MuJoCo) and ``Pillow`` for the
image. Missing Pillow, or with the asset absent, every function is a silent no-op -- a loading
overlay is not an artifact, so best-effort is the right altitude (same stance as
:mod:`roqsim.window_branding`).
"""

from __future__ import annotations

import logging
import os
from importlib import resources

log = logging.getLogger(__name__)

#: Splash art with the logo + "robotics simulation toolkit" text baked in.
_SPLASH = ("assets", "splash.jpg")
#: Status line drawn under the baked subtitle. Passed through :func:`show_loading_overlay`, so a
#: caller can swap it for progress messages ("Compiling world…", etc.) without touching the art.
_LOADING_TEXT = "Loading…"
#: Vertical placement (fraction of viewport height) of the status line's top edge, and its colour.
_TEXT_Y_FRAC = 0.82
_TEXT_RGB = (159, 178, 194)  # muted slate, matches the baked subtitle


def _splash_path() -> str | None:
    res = resources.files(__package__).joinpath(*_SPLASH)
    try:
        path = os.fspath(res)
    except TypeError:
        return None  # zipped install: no real file to open
    return path if os.path.exists(path) else None


def _draw_status_text(image, text: str) -> None:
    """Draw ``text`` centred horizontally in the lower region (below the baked subtitle)."""
    if not text:
        return
    from PIL import ImageDraw, ImageFont

    width, height = image.size
    try:
        font = ImageFont.load_default(size=max(14, round(height * 0.033)))
    except TypeError:  # Pillow < 10.1: size-less bitmap default
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((width - (x1 - x0)) // 2 - x0, round(height * _TEXT_Y_FRAC) - y0),
        text,
        fill=_TEXT_RGB,
        font=font,
    )


def _overlay_image(width: int, height: int, text: str = _LOADING_TEXT):
    """Full-viewport RGB array (H, W, 3) uint8 -- splash scaled to *cover* the viewport (no bars),
    with ``text`` drawn in the lower region -- or ``None`` if unavailable. The art's own navy fills
    any cropped edge, so it reads full-bleed."""
    if width <= 0 or height <= 0:
        return None
    path = _splash_path()
    if path is None:
        log.debug("loading overlay skipped: splash asset not found")
        return None
    try:
        import numpy as np
        from PIL import Image
    except ImportError as err:
        log.debug("loading overlay skipped: %s", err)
        return None

    splash = Image.open(path).convert("RGB")
    # Cover (CSS background-size: cover): scale so the image fills both axes, centre-crop the excess.
    scale = max(width / splash.width, height / splash.height)
    scaled = splash.resize(
        (max(1, round(splash.width * scale)), max(1, round(splash.height * scale))), Image.LANCZOS
    )
    left = (scaled.width - width) // 2
    top = (scaled.height - height) // 2
    covered = scaled.crop((left, top, left + width, top + height))
    _draw_status_text(covered, text)
    return np.asarray(covered, dtype=np.uint8)


def show_loading_overlay(viewer, text: str = _LOADING_TEXT) -> bool:
    """Draw the full-bleed splash with ``text`` in its lower region. Returns whether it was shown.

    ``text`` is easy to swap (e.g. a progress message); pass ``""`` for the splash alone.
    Best-effort: any failure (no Pillow, no asset, a zero-sized viewport, an overlay API the
    installed MuJoCo lacks) is swallowed and reported as ``False``.
    """
    try:
        import mujoco

        rect = viewer.viewport
        image = _overlay_image(rect.width, rect.height, text)
        if image is None:
            return False
        viewer.set_images((mujoco.MjrRect(0, 0, rect.width, rect.height), image))
        viewer.sync()
        return True
    except Exception as err:  # noqa: BLE001 — a cosmetic overlay never breaks the run
        log.debug("loading overlay skipped: %s", err)
        return False


def clear_loading_overlay(viewer) -> None:
    """Remove the splash overlay. Safe to call even if nothing was shown."""
    try:
        viewer.clear_images()
    except Exception as err:  # noqa: BLE001
        log.debug("clearing loading overlay failed: %s", err)
