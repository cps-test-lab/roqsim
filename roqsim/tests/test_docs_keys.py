# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A key cannot be added without documenting it.

The window's list is generated from the bindings, so the *list* cannot drift and the docs do not
reproduce it. What prose still owes a key is the part a list cannot carry -- when it applies, what it
writes, why it is that key -- and that is what goes stale silently. F9 was a key nobody outside the
source knew about; this is what stops the next one.
"""

from pathlib import Path

import pytest

from roqsim import keys

_QUICKSTART = Path(__file__).resolve().parents[2] / "docs" / "quickstart.rst"


@pytest.mark.parametrize(
    "binding", [b for b in keys.CATALOGUE if b.label.startswith("F")], ids=lambda b: b.action
)
def test_every_function_key_is_named_in_the_quickstart(binding):
    """Function keys only: ``Shift`` and ``Up/Down`` are ordinary words on that page, so a substring
    check on them would pass on anything. An F-key is unambiguous, and is also the kind nobody
    guesses -- exactly the kind that needs a sentence."""
    assert binding.label in _QUICKSTART.read_text(encoding="utf-8"), (
        f"{binding.label} ({binding.help}) is in the window's key list but named nowhere in "
        f"{_QUICKSTART.name}"
    )
