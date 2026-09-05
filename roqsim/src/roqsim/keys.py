# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""The keys roqsim adds to the viewer window: what each one is, and what it says of itself.

Every key roqsim binds is declared here once, as a :class:`KeyBinding`, and the handlers take their
keycodes from these records rather than from constants of their own. The F1 overlay is rendered from
the same records (:func:`help_columns`), so what the window says the keys do and what they do cannot
drift apart -- there is nothing to keep in step, only one thing.

**Which keys are takeable.** MuJoCo's Simulate owns the window and roqsim's callback runs *in
addition to* its handling rather than instead of it, so a key can never be taken away from Simulate,
only shared. That leaves:

* **F1-F7** are Simulate's (help / info / profiler / sensor / fullscreen / frame / label). F1 is the
  one roqsim shares deliberately: it means "help" in both, so a press opens Simulate's list and this
  one side by side. Every other one is left alone.
* **The 26 letters** are all bound by Simulate to a visualization or rendering flag (``V`` is
  ``mjVIS_TENDON``; see ``mjVISSTRING`` / ``mjRNDSTRING``), so a letter here would toggle rendering
  every time it was pressed. This is what put camera travel on the arrows rather than on WASD.
* **The arrows and Page Up/Down** are free while the simulation runs: Simulate's own arrow bindings
  only step the physics while it is *paused*.
* **F8, F9, F10** are unbound by Simulate and are roqsim's. **F11 and F12** are still free.

A binding is owned by the handler that implements it, which declares it in a ``key_bindings``
attribute; :func:`merge` collects those from whatever objects it is handed. So the list a window
shows is the keys that run actually has, and a source roqsim knows nothing about can join it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Auto-repeat delivers several presses from one held key (~0.2 s apart). A handler ignores a repeat
#: within this interval -- above the repeat rate, below a deliberate double-press.
DEBOUNCE_S = 0.4

#: Simulate's own function keys, which roqsim does not bind (F1 excepted, deliberately -- see above).
SIMULATE_FUNCTION_KEYS = frozenset(range(290, 297))

KEY_F1, KEY_F8, KEY_F9, KEY_F10 = 290, 297, 298, 299

#: The token both Shifts carry: a modifier, not a direction to travel in.
_SPRINT = "shift"

#: Overlay headings, in the order the list renders them.
GROUP_CAMERA = "camera"
GROUP_RUN = "run"
GROUPS = (GROUP_CAMERA, GROUP_RUN)


class KeyBindingError(RuntimeError):
    """Two sources claim one key, or one action. Raised by :func:`merge`, never swallowed."""


@dataclass(frozen=True)
class Key:
    """One physical key: what MuJoCo calls it, what X calls it, and what it means to its handler."""

    #: GLFW keycode, as the passive viewer's key callback reports it.
    code: int
    #: X keysym name for the :mod:`roqsim.key_state` poll. Empty where the key is not polled.
    keysym: str = ""
    #: What this code carries to its handler. Empty where the handler matches on the code itself.
    token: str = ""


@dataclass(frozen=True)
class KeyBinding:
    """One line of the help overlay, and the keys that line describes.

    Several keys to a line is the normal case, not an exception: one line covers both Shifts (the
    same token from two codes) and the forward/back pair (two tokens from two codes). Splitting them
    would put four near-identical lines in a list whose whole point is being short.
    """

    #: Stable id, e.g. ``camera.height``. What :func:`merge` dedupes on and orders by.
    action: str
    #: Which heading the line renders under.
    group: str
    #: The overlay's left column: how the key is written on a keyboard.
    label: str
    #: The overlay's right column: what it does, in as few words as carry the meaning.
    help: str
    #: The keys themselves.
    keys: tuple[Key, ...]

    @property
    def codes(self) -> tuple[int, ...]:
        return tuple(k.code for k in self.keys)

    def tokens(self) -> dict[int, str]:
        """GLFW code -> token, for the keys that carry one."""
        return {k.code: k.token for k in self.keys if k.token}

    def keysym_tokens(self) -> dict[str, str]:
        """X keysym -> token, for the keys that are polled."""
        return {k.keysym: k.token for k in self.keys if k.keysym and k.token}


# -- the keys ---------------------------------------------------------------------------------------
#
# The help text carries no numbers. "hold to fly faster" rather than "3x faster", because the factor
# is `roqsim.viewer.WALK_SPRINT` and a number written here would be a second copy of it, free to
# drift. The text is ASCII for a harder reason: the overlay is drawn with MuJoCo's built-in bitmap
# font, where a multi-byte glyph comes out as rubbish, so the arrows are spelled out.

CAMERA_MODE = KeyBinding("camera.mode", GROUP_CAMERA, "F10", "mouse / fly", (Key(KEY_F10),))
CAMERA_FORWARD = KeyBinding(
    "camera.forward",
    GROUP_CAMERA,
    "Up/Down",
    "fly forward / back",
    (Key(265, "Up", "w"), Key(264, "Down", "s")),
)
CAMERA_STRAFE = KeyBinding(
    "camera.strafe",
    GROUP_CAMERA,
    "Left/Right",
    "strafe level",
    (Key(263, "Left", "a"), Key(262, "Right", "d")),
)
CAMERA_HEIGHT = KeyBinding(
    "camera.height",
    GROUP_CAMERA,
    "PgUp/PgDn",
    "rise / drop",
    (Key(266, "Prior", "e"), Key(267, "Next", "q")),
)
CAMERA_SPRINT = KeyBinding(
    "camera.sprint",
    GROUP_CAMERA,
    "Shift",
    "hold to fly faster",
    (Key(340, "Shift_L", _SPRINT), Key(344, "Shift_R", _SPRINT)),
)

RECORD_TAKE = KeyBinding("run.record", GROUP_RUN, "F9", "recording take on / off", (Key(KEY_F9),))
SAVE_VIEW = KeyBinding(
    "run.save_view", GROUP_RUN, "F8", "save camera into the world", (Key(KEY_F8),)
)
SHOW_HELP = KeyBinding("run.help", GROUP_RUN, "F1", "this list on / off", (Key(KEY_F1),))

#: Every key roqsim binds, in the order the overlay lists them. A run shows the subset it has.
CATALOGUE = (
    CAMERA_MODE,
    CAMERA_FORWARD,
    CAMERA_STRAFE,
    CAMERA_HEIGHT,
    CAMERA_SPRINT,
    RECORD_TAKE,
    SAVE_VIEW,
    SHOW_HELP,
)

#: The keys that travel: what :class:`roqsim.viewer.WalkKeys` polls, and the source of its maps.
WALK = (CAMERA_FORWARD, CAMERA_STRAFE, CAMERA_HEIGHT, CAMERA_SPRINT)

#: Everything the camera owns, mode switch included: what ``WalkKeys`` declares.
CAMERA = (CAMERA_MODE, *WALK)


# -- what the handlers read -------------------------------------------------------------------------


def keycodes(bindings) -> dict[int, str]:
    """GLFW code -> token, over ``bindings``. What a key callback matches an event against."""
    out: dict[int, str] = {}
    for binding in bindings:
        out.update(binding.tokens())
    return out


def keysyms(bindings) -> dict[str, str]:
    """X keysym -> token, over ``bindings``. What :class:`roqsim.key_state.KeyState` polls."""
    out: dict[str, str] = {}
    for binding in bindings:
        out.update(binding.keysym_tokens())
    return out


def merge(*sources) -> tuple[KeyBinding, ...]:
    """The bindings of every source, in catalogue order, refusing a clash.

    A source is anything carrying a ``key_bindings`` attribute -- read with ``getattr``, so this
    knows nothing about what kinds of thing exist: a handler, its class, and (once plugin-declared
    keys land) a plugin are sources on the same footing. ``None`` sources are skipped, so a caller
    need not branch on a handler it did not install.

    The order is the catalogue's, not the order the sources were passed, so wiring cannot reshuffle
    the list a reader has learned; anything the catalogue does not name sorts after what it does.

    Raises :class:`KeyBindingError` when two bindings claim one keycode or one action. A key that
    silently did the second thing bound to it, or a list that named a key twice, is worse than a
    window that refuses to open.
    """
    seen: dict[int, KeyBinding] = {}
    actions: dict[str, KeyBinding] = {}
    for source in sources:
        for binding in getattr(source, "key_bindings", ()) if source is not None else ():
            first = actions.get(binding.action)
            if first is not None:
                if first is binding:
                    continue  # the same binding declared by two sources is one key, not a clash
                raise KeyBindingError(
                    f"two keys share the action {binding.action!r}: "
                    f"{first.label} and {binding.label}"
                )
            for code in binding.codes:
                claimed = seen.get(code)
                if claimed is not None:
                    raise KeyBindingError(
                        f"{binding.label} ({code}) is claimed by both "
                        f"{claimed.action!r} and {binding.action!r}"
                    )
                seen[code] = binding
            actions[binding.action] = binding
    return tuple(sorted(actions.values(), key=_order))


def _order(binding: KeyBinding) -> tuple[int, int]:
    """Catalogue position, group first. What the catalogue does not name goes last, in both."""
    group = GROUPS.index(binding.group) if binding.group in GROUPS else len(GROUPS)
    known = [b.action for b in CATALOGUE]
    index = known.index(binding.action) if binding.action in known else len(known)
    return group, index


# -- what the window shows --------------------------------------------------------------------------

#: Names the list, because Simulate's own help is on screen at the same time in the other corner.
HELP_TITLE = "roqsim keys"

#: Indent under a heading.
_INDENT = "  "


def _lines(bindings) -> list[tuple[str, str]]:
    """The overlay's lines as ``(label, help)`` pairs, headings and blanks included."""
    out: list[tuple[str, str]] = [(HELP_TITLE, ""), ("", "")]
    for group in (*GROUPS, *sorted({b.group for b in bindings} - set(GROUPS))):
        listed = [b for b in bindings if b.group == group]
        if not listed:
            continue  # a group whose every key this run lacks is not a heading over nothing
        if out[-1] != ("", ""):
            out.append(("", ""))
        out.append((group.upper(), ""))
        out.extend((_INDENT + b.label, b.help) for b in listed)
    return out


def help_columns(bindings) -> tuple[str, str]:
    """The two columns ``set_texts`` aligns line by line: key labels, and what they do."""
    lines = _lines(bindings)
    return "\n".join(label for label, _ in lines), "\n".join(text for _, text in lines)
