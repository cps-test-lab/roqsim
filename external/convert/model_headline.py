# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""Give a generated MJCF the opening sentence that says what the model is.

The model catalog in the docs (``docs/_ext/model_docs.py``) describes each model with the first
sentence of the comment that opens its ``<mujoco>`` element, so a generated model that starts
straight into ``<compiler>`` -- or into its generator banner -- has nothing to be described by, and
the catalog falls back to repeating the package summary once per row. Emitting the sentence here
keeps it with the model rather than in prose that a re-run cannot refresh.
"""

from __future__ import annotations

import re


def with_headline(xml: str, sentence: str) -> str:
    """``xml`` with ``sentence`` as the comment opening its ``<mujoco>`` element.

    Raises if the document has no ``<mujoco>`` element or already opens with a comment: a second
    headline silently below the first would describe nothing, and a converter that quietly wrote no
    headline would leave the catalog describing the package instead of the model.
    """
    opening = re.search(r"<mujoco\b[^>]*>", xml)
    if opening is None:
        raise ValueError("no <mujoco> element to put a headline on")
    if re.match(r"\s*<!--", xml[opening.end() :]):
        raise ValueError("<mujoco> already opens with a comment")
    if not sentence.endswith("."):
        raise ValueError(f"a headline is a sentence and ends with a period: {sentence!r}")
    return f"{xml[: opening.end()]}\n  <!-- {sentence} -->{xml[opening.end() :]}"
