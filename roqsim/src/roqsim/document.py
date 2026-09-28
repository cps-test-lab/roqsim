# SPDX-License-Identifier: Apache-2.0
"""The two checks every document roqsim reads shares: its stated version, and keys nothing reads.

A document that outlives the code that wrote it (a world, a recording, a scene manifest, a sketch)
states its version, and an absent stamp is version 1. A reader refuses a version above the one it
implements, naming both, rather than reading the keys that happen to overlap.

A document read by name refuses a key outside the set its readers consume, naming the nearest one
that exists: nothing reads it, so it is a misspelling whose default silently takes its place.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from difflib import get_close_matches


def check_version(
    doc: Mapping,
    key: str,
    *,
    reads: int,
    document: str,
    where: str,
    error: type[Exception] = ValueError,
) -> int:
    """The version *doc* states under *key* (1 when absent), refused unless ``1 <= version <= reads``."""
    if key not in doc:
        return 1
    version = doc[key]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise error(
            f"{where}: '{key}: {version!r}' is not a {document} version; it is a positive integer "
            f"(this roqsim reads up to {reads}), or absent for version 1."
        )
    if version > reads:
        raise error(
            f"{where} is {document} version {version}; this roqsim reads up to {reads}. Read it "
            f"with the roqsim version that wrote it."
        )
    return version


def nearest(key: str, known: Collection[str]) -> str | None:
    """The known key a typo most likely meant, or None when none is close.

    The cutoff is tight: 'radius' for 'radius_m' helps, 'body' for 'mass' sends someone to the
    wrong line.
    """
    matches = get_close_matches(str(key), sorted(known), n=1, cutoff=0.8)
    return matches[0] if matches else None


def refuse_unknown_keys(
    block: Mapping,
    known: Collection[str],
    where: str,
    *,
    error: type[Exception] = ValueError,
) -> None:
    """Refuse a *block* carrying a key outside *known*, naming the nearest known key for each."""
    if not isinstance(block, Mapping):
        raise error(f"{where}: must be a mapping, not {type(block).__name__}")
    unknown = sorted(set(block) - set(known), key=str)
    if not unknown:
        return
    named = []
    for key in unknown:
        near = nearest(key, known)
        named.append(f"{key!r} (did you mean {near!r}?)" if near else repr(key))
    raise error(
        f"{where}: unknown key(s) {', '.join(named)}; it takes {', '.join(sorted(known))}. "
        f"Nothing reads any other key, so it would be ignored rather than applied."
    )
