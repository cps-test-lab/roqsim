# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A key cannot be added without documenting it.

The window's list is generated, so it cannot drift from the keys. The docs are prose and can, which
is how F9 came to be a key nobody outside the source knew about. So the quickstart shows the block
the window shows, and this is what holds the two together: add a key and this test fails until the
docs say what it does.
"""

import re
import textwrap
from pathlib import Path

import pytest

from roqsim import keys

_QUICKSTART = Path(__file__).resolve().parents[2] / "docs" / "quickstart.rst"

#: The literal block under the ``.. _viewer-keys:`` label -- the list as the window renders it.
_PINNED = re.compile(
    r"^\.\. _viewer-keys:$.*?^\.\. code-block:: text$\n\n(.*?)(?=^\S)",
    re.MULTILINE | re.DOTALL,
)


def _documented_block() -> str:
    text = _QUICKSTART.read_text(encoding="utf-8")
    match = _PINNED.search(text)
    assert match, f"{_QUICKSTART.name} has no `.. _viewer-keys:` section with a code block under it"
    return textwrap.dedent(match.group(1)).strip("\n")


def test_the_quickstart_shows_the_list_the_window_shows():
    expected = keys.help_block(keys.CATALOGUE)
    assert _documented_block() == expected, (
        "the key list in docs/quickstart.rst is not what the window renders any more.\n"
        "Replace the code block under `.. _viewer-keys:` with, indented by three spaces:\n\n"
        + expected
    )


@pytest.mark.parametrize(
    "binding", [b for b in keys.CATALOGUE if b.label.startswith("F")], ids=lambda b: b.action
)
def test_every_function_key_is_explained_in_prose_too(binding):
    """A line in the list says what a key does; a key also deserves a sentence saying why.

    Function keys only: ``Shift`` and ``Up/Down`` appear all over the page as ordinary words, so a
    substring check on them would pass on anything.
    """
    text = _QUICKSTART.read_text(encoding="utf-8")
    outside_the_block = text.replace(textwrap.indent(_documented_block(), "   "), "")
    assert binding.label in outside_the_block, (
        f"{binding.label} ({binding.help}) is in the window's list but explained nowhere in "
        f"{_QUICKSTART.name}"
    )
