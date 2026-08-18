# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the roqsim doc-generation directives.

The directives enumerate the framework's own registries at build time and emit RST, so the plugin /
model / world / texture catalogs can never drift from what is actually registered. Two rendering
idioms are shared here:

* **rubric group headings** -- ``.. rubric::`` renders as a heading *without* opening a document
  section, which is what lets us build headings inside ``nested_parse`` without the "Unexpected
  section title" errors real underlines would raise (and which would fail the ``-W`` build).
* **catalog rows** -- a two-column ``list-table`` row per item: the preview thumbnail on the left,
  the name + synthesised description on the right. Thumbnails live **beside their model**
  (``<model-dir>/<name>.thumb.png``, generated once by ``make thumbnails``); the directive references
  that file by a path relative to the docs source dir, so Sphinx copies it into the build. A missing
  thumbnail leaves the left cell blank -- the row still aligns and the ``-W`` build stays green with
  no GL at doc-build time.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive  # noqa: F401 - re-exported for the directive modules
from docutils.statemachine import StringList


def rubric(text: str) -> list[str]:
    """RST lines for a heading that does not open a section (safe inside ``nested_parse``)."""
    return [f".. rubric:: {text}", ""]


def _image_lines(env, thumb: str | Path | None, indent: str, *, width: str) -> list[str]:
    """``.. image::`` lines for a co-located thumbnail, referenced relative to the docs source dir.

    Returns an empty list (blank cell) when the thumbnail file is absent, so the build never
    references a missing image.
    """
    if not thumb or not Path(thumb).is_file():
        return []
    rel = os.path.relpath(Path(thumb), env.srcdir)
    return [f"{indent}.. image:: {rel}", f"{indent}   :width: {width}"]


def catalog_table(
    env, rows: list[tuple[str | Path | None, list[str]]], *, width: str = "150px"
) -> list[str]:
    """RST for a two-column catalog: ``(thumbnail_path | None, body_lines)`` per row.

    Column 1 is the preview image (blank when the file is absent); column 2 is ``body_lines``
    (already-formatted RST, first line is the bold item name). Content of both cells sits at the
    same indent so the image directive inside a cell parses cleanly.
    """
    out = [".. list-table::", "   :widths: 22 78", "   :class: ss-catalog", ""]
    for thumb, body in rows:
        img = _image_lines(env, thumb, "       ", width=width)
        if img:
            out.append("   * - " + img[0].strip())
            out.extend(img[1:])
        else:
            out.append("   * -")
        first, *rest = body or [""]
        out.append(f"     - {first}")
        out.extend(f"       {line}" if line else "" for line in rest)
    out.append("")
    return out


def docstring_lines(doc: str | None) -> list[str]:
    """Dedented docstring split into lines for embedding in generated RST (empty when None)."""
    if not doc:
        return []
    return textwrap.dedent(doc).strip("\n").splitlines()


def parse_rst(directive, lines: list[str]) -> list[nodes.Node]:
    """Parse generated RST ``lines`` in the directive's context and return the child nodes."""
    container = nodes.section()
    container.document = directive.state.document
    directive.state.nested_parse(StringList(lines), directive.content_offset, container)
    return list(container.children)
