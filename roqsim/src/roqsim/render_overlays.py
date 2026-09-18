"""Insets drawn onto rendered frames: a clock in the corner, a map beside the robot, a caption.

``roqsim render`` draws a recording one sample per frame. An **overlay** takes each frame after the
scene is rasterised and before it is encoded, and paints on it -- 2D, in pixels, with the sample's
simulated time in hand. That is the hook; what is painted is up to the overlay.

An overlay is any object with::

    name: str
    prepare(width, height, *, state) -> None     # optional; once, before the first frame
    draw(frame, t) -> frame                      # every frame; HxWx3 uint8 in, the same out

``frame`` may be painted in place. ``state`` is the recording being drawn (a path or ``None``), so an
overlay that reads files beside it can find them without being told where.

Overlays are found by name. ``clock`` ships here; other packages register theirs under the
``roqsim.render_overlays`` entry-point group (a nav package's costmap, say) and they become available by
name without roqsim knowing them::

    [project.entry-points."roqsim.render_overlays"]
    costmap = "some_package.video:CostmapOverlay"

A spec is a bare name or ``{"name": {options}}``. The options every overlay takes are its
placement -- ``anchor`` (``top-right`` and the other corners and edges), ``width`` (a fraction of the
frame width, or pixels when over 1) and ``margin`` (pixels); an overlay defines the rest.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "roqsim.render_overlays"

ANCHORS = (
    "top-left",
    "top",
    "top-right",
    "left",
    "center",
    "right",
    "bottom-left",
    "bottom",
    "bottom-right",
)

#: Options every overlay understands, taken out of a spec before the rest reaches the overlay.
PLACEMENT_KEYS = ("anchor", "width", "margin")


class OverlayError(ValueError):
    """An overlay spec names nothing installed, or an overlay returned a frame ffmpeg cannot take."""


@dataclass(frozen=True)
class Placement:
    """Where an inset goes on a frame, in terms that do not depend on the frame's size."""

    anchor: str = "top-right"
    width: float = 0.3
    margin: int = 12

    @classmethod
    def from_spec(cls, options: dict, name: str, default_anchor: str | None = None) -> Placement:
        """The placement an overlay's options state, over ``default_anchor`` (an overlay may prefer a
        corner: the clock takes the top-left so it does not sit on an inset in the top-right)."""
        anchor = str(options.get("anchor", default_anchor or cls.anchor))
        if anchor not in ANCHORS:
            raise OverlayError(
                f"overlay {name!r}: anchor {anchor!r} is not one of {', '.join(ANCHORS)}"
            )
        width = options.get("width", cls.width)
        margin = options.get("margin", cls.margin)
        if isinstance(width, bool) or not isinstance(width, (int, float)) or width <= 0:
            raise OverlayError(f"overlay {name!r}: width must be a positive number, got {width!r}")
        if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
            raise OverlayError(f"overlay {name!r}: margin must be pixels (an int), got {margin!r}")
        return cls(anchor, float(width), int(margin))

    def pixels(self, frame_width: int) -> int:
        """The inset's width in pixels: a fraction of the frame, or stated outright when over 1."""
        return int(round(self.width if self.width > 1 else self.width * frame_width))

    def box(self, frame_w: int, frame_h: int, inset_w: int, inset_h: int) -> tuple[int, int]:
        """Top-left pixel of an inset of ``inset_w`` x ``inset_h`` on a frame of ``frame_w`` x ``frame_h``."""
        m = self.margin
        xs = {"left": m, "center": (frame_w - inset_w) // 2, "right": frame_w - inset_w - m}
        ys = {"top": m, "center": (frame_h - inset_h) // 2, "bottom": frame_h - inset_h - m}
        vertical, _, horizontal = self.anchor.partition("-")
        if self.anchor in ("left", "right"):
            vertical, horizontal = "center", self.anchor
        elif self.anchor in ("top", "bottom"):
            horizontal = "center"
        elif self.anchor == "center":
            vertical = horizontal = "center"
        return xs[horizontal], ys[vertical]


def paste(frame: np.ndarray, inset, placement: Placement) -> np.ndarray:
    """Composite a PIL image (RGB or RGBA) onto ``frame`` at its placement; returns ``frame``."""
    from PIL import Image

    h, w = frame.shape[:2]
    x, y = placement.box(w, h, inset.width, inset.height)
    canvas = Image.fromarray(frame)
    if inset.mode == "RGBA":
        canvas.paste(inset, (x, y), inset)
    else:
        canvas.paste(inset, (x, y))
    frame[...] = np.asarray(canvas)
    return frame


def font(size: int):
    """Pillow's bundled font at ``size``; no font files are looked for on the host."""
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=max(8, int(size)))
    except TypeError:  # Pillow < 10.1: size-less bitmap default
        return ImageFont.load_default()


class ClockOverlay:
    """The simulated time, as text: ``t = 12.40 s``.

    Options: ``format`` (a Python format string with ``{t}``), ``size`` (text height as a fraction
    of the frame height), and the placement keys.
    """

    name = "clock"
    #: Out of the way of an inset, which takes the top-right unless told otherwise.
    default_anchor = "top-left"

    def __init__(self, placement: Placement, fmt: str = "t = {t:6.2f} s", size: float = 0.035):
        self.placement = placement
        self.fmt = str(fmt)
        self.size = float(size)
        self._font = None
        self._pad = 6

    @classmethod
    def from_spec(cls, options: dict, placement: Placement) -> ClockOverlay:
        allowed = {"format", "size"}
        if unknown := set(options) - allowed:
            raise OverlayError(
                f"overlay 'clock': unknown option(s) {', '.join(sorted(unknown))}; "
                f"it takes {', '.join(sorted(allowed))} and the placement keys"
            )
        return cls(placement, options.get("format", "t = {t:6.2f} s"), options.get("size", 0.035))

    def prepare(self, width: int, height: int, *, state=None) -> None:
        self._font = font(round(height * self.size))

    def draw(self, frame: np.ndarray, t: float) -> np.ndarray:
        from PIL import Image, ImageDraw

        if self._font is None:
            self.prepare(frame.shape[1], frame.shape[0])
        text = self.fmt.format(t=t)
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=self._font)
        pad = self._pad
        inset = Image.new("RGBA", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), (0, 0, 0, 140))
        ImageDraw.Draw(inset).text(
            (pad - x0, pad - y0), text, fill=(255, 255, 255, 255), font=self._font
        )
        return paste(frame, inset, self.placement)


BUILTIN = {"clock": ClockOverlay}


@functools.cache
def _entry_points():
    eps = metadata.entry_points()
    if hasattr(eps, "select"):  # Python 3.10+
        return tuple(eps.select(group=ENTRY_POINT_GROUP))
    return tuple(eps.get(ENTRY_POINT_GROUP, ()))  # pragma: no cover - legacy


def available() -> dict[str, str]:
    """Every overlay name this environment offers, with where it comes from."""
    names = {name: "roqsim" for name in BUILTIN}
    for ep in _entry_points():
        names.setdefault(ep.name, ep.value)
    return names


def _resolve(name: str):
    if name in BUILTIN:
        return BUILTIN[name]
    for ep in _entry_points():
        if ep.name == name:
            try:
                return ep.load()
            except ImportError as err:
                raise OverlayError(
                    f"overlay {name!r} is registered by an installed package but could not be "
                    f"imported: {err}"
                ) from err
    have = ", ".join(sorted(available())) or "none"
    raise OverlayError(f"overlay {name!r}: nothing installed registers it. Available: {have}.")


def parse_spec(spec) -> tuple[str, dict]:
    """``"clock"`` or ``{"clock": {...}}`` (a dict, or that dict as JSON text) -> ``(name, options)``."""
    if isinstance(spec, str):
        text = spec.strip()
        if text[:1] == "{":
            try:
                spec = json.loads(text)
            except json.JSONDecodeError as err:
                raise OverlayError(f"--overlay: not JSON: {err}") from None
        else:
            return text, {}
    if isinstance(spec, dict) and len(spec) == 1:
        ((name, options),) = spec.items()
        if options is None:
            options = {}
        if not isinstance(options, dict):
            raise OverlayError(f"overlay {name!r}: options must be a mapping, got {options!r}")
        return str(name), dict(options)
    raise OverlayError(f"an overlay is a name or {{name: {{options}}}}, got {spec!r}")


def build(spec, width: int, height: int, *, state=None):
    """One overlay from its spec, prepared for frames of ``width`` x ``height``."""
    if hasattr(spec, "draw"):  # already an overlay object, from a Python caller
        if prepare := getattr(spec, "prepare", None):
            prepare(width, height, state=Path(state) if state else None)
        return spec
    name, options = parse_spec(spec)
    cls = _resolve(name)
    placement = Placement.from_spec(options, name, getattr(cls, "default_anchor", None))
    rest = {k: v for k, v in options.items() if k not in PLACEMENT_KEYS}
    factory = getattr(cls, "from_spec", None)
    try:
        overlay = factory(rest, placement) if factory else cls(placement, **rest)
    except TypeError as err:
        raise OverlayError(f"overlay {name!r}: {err}") from None
    if not hasattr(overlay, "draw"):
        raise OverlayError(f"overlay {name!r} ({cls}) has no draw(frame, t)")
    if not getattr(overlay, "name", None):
        overlay.name = name
    if prepare := getattr(overlay, "prepare", None):
        prepare(width, height, state=Path(state) if state else None)
    return overlay


def build_overlays(specs, width: int, height: int, *, state=None) -> list:
    return [build(spec, width, height, state=state) for spec in specs or ()]


def apply_all(overlays, frame: np.ndarray, t: float) -> np.ndarray:
    """Run every overlay on ``frame`` and refuse a result ffmpeg would misread.

    The encoder is handed raw bytes against a fixed ``WxH`` and ``rgb24``: a wrong shape or dtype
    would not fail, it would produce a sheared or striped video. So the check is here, by name.
    """
    expected = frame.shape
    for overlay in overlays:
        out = overlay.draw(frame, t)
        if not isinstance(out, np.ndarray) or out.shape != expected or out.dtype != np.uint8:
            got = (
                f"{type(out).__name__}"
                if not isinstance(out, np.ndarray)
                else f"shape {out.shape} {out.dtype}"
            )
            raise OverlayError(
                f"overlay {overlay.name!r} returned {got}; a frame is shape {expected} uint8"
            )
        frame = np.ascontiguousarray(out)
    return frame


__all__ = [
    "ENTRY_POINT_GROUP",
    "OverlayError",
    "Placement",
    "ClockOverlay",
    "available",
    "build",
    "build_overlays",
    "apply_all",
    "paste",
    "font",
]
