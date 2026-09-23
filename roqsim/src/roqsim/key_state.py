"""Physical keyboard state via X11 (ctypes) -- what a *held* key really is.

MuJoCo's passive viewer forwards key **presses** to a Python callback and never releases, and the
auto-repeat stream it does forward is unusable as a hold signal: measured against a keyboard set to
25 Hz repeat it arrived at ~5 Hz with second-long holes, so a camera driven from those events stalls
mid-flight and cannot know when to stop. ``XQueryKeymap`` answers the actual question -- it returns
the keyboard's live bitmap, so a held key flies for exactly as long as it is down.

Best-effort and X11-only, like :mod:`roqsim.window_branding`: :meth:`KeyState.open` returns ``None`` on
Wayland/macOS, without a ``DISPLAY``, or when libX11 is missing, and the caller falls back to the
event stream. Keys are only reported while a window of *this* process has the input focus, so arrows
typed into another application never fly our camera.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys

log = logging.getLogger(__name__)

_ANY_PROPERTY_TYPE = 0
_MAX_ANCESTOR_HOPS = 8  # focus usually lands on the top-level itself or a child a hop or two down


class KeyState:
    """Live state of a fixed set of keys, polled from the X server.

    Construct with :meth:`open` (which returns ``None`` when X11 is unavailable), call :meth:`held`
    once per frame, and :meth:`close` at shutdown. ``keys`` maps an X keysym name (``"Up"``,
    ``"Prior"``, ``"w"``) to whatever token the caller wants back from :meth:`held`.
    """

    def __init__(self, xlib, disp, keycodes: dict[int, str]):
        self._xlib = xlib
        self._disp = disp
        self._keycodes = keycodes
        self._pid = os.getpid()
        self._focus_owner: dict[int, bool] = {}  # window id -> is it ours (cached; ids are stable)
        self._net_wm_pid = xlib.XInternAtom(disp, b"_NET_WM_PID", False)

    @classmethod
    def open(cls, keys: dict[str, str]) -> KeyState | None:
        """Open an X connection and resolve ``keys``, or return ``None`` if that is not possible."""
        if sys.platform != "linux" or not os.environ.get("DISPLAY"):
            return None
        try:
            xlib = _load_xlib()
        except OSError as err:
            log.debug("key state unavailable: libX11 (%s)", err)
            return None
        disp = xlib.XOpenDisplay(None)
        if not disp:
            log.debug("key state unavailable: cannot open display %r", os.environ.get("DISPLAY"))
            return None
        keycodes: dict[int, str] = {}
        for name, token in keys.items():
            keysym = xlib.XStringToKeysym(name.encode())
            code = int(xlib.XKeysymToKeycode(disp, keysym)) if keysym else 0
            if code:
                keycodes[code] = token
            else:
                log.debug("key state: no keycode for %r on this layout", name)
        if not keycodes:
            xlib.XCloseDisplay(disp)
            return None
        return cls(xlib, disp, keycodes)

    def held(self) -> set[str]:
        """The tokens of the keys currently held down -- empty unless this process has the focus."""
        if not self._focused():
            return set()
        bitmap = (ctypes.c_char * 32)()
        self._xlib.XQueryKeymap(self._disp, bitmap)
        raw = bytes(bitmap)
        return {
            token
            for code, token in self._keycodes.items()
            if raw[code // 8] & (1 << (code % 8))  # X packs one bit per keycode, LSB first
        }

    def close(self) -> None:
        if self._disp:
            self._xlib.XCloseDisplay(self._disp)
            self._disp = None

    def _focused(self) -> bool:
        """Does the focused window belong to this process? (Keystrokes elsewhere are not ours.)"""
        win = ctypes.c_ulong()
        revert = ctypes.c_int()
        self._xlib.XGetInputFocus(self._disp, ctypes.byref(win), ctypes.byref(revert))
        wid = int(win.value)
        if wid <= 1:  # None (0) / PointerRoot (1): nobody holds the focus
            return False
        if wid not in self._focus_owner:
            self._focus_owner[wid] = self._owns(wid)
        return self._focus_owner[wid]

    def _owns(self, win: int) -> bool:
        """True if ``win`` or one of its ancestors carries our PID.

        The focus often sits on a child window while ``_NET_WM_PID`` is set only on the top-level the
        window manager reparented, so walk up until the property turns up.
        """
        # Xlib's default error handler calls exit() -- and a window can vanish between the focus
        # query and reading its property. Ignore errors for the walk, then restore.
        prev = self._xlib.XSetErrorHandler(ctypes.cast(_IGNORE_X_ERRORS, ctypes.c_void_p))
        try:
            for _ in range(_MAX_ANCESTOR_HOPS):
                if win <= 0:
                    return False
                pid = self._window_pid(win)
                if pid is not None:
                    return pid == self._pid
                win = self._parent(win)
            return False
        finally:
            self._xlib.XSetErrorHandler(prev)

    def _window_pid(self, win: int) -> int | None:
        actual_type = ctypes.c_ulong()
        actual_format = ctypes.c_int()
        nitems = ctypes.c_ulong()
        bytes_after = ctypes.c_ulong()
        data = ctypes.POINTER(ctypes.c_ubyte)()
        status = self._xlib.XGetWindowProperty(
            self._disp,
            win,
            self._net_wm_pid,
            0,
            1,
            False,
            _ANY_PROPERTY_TYPE,
            ctypes.byref(actual_type),
            ctypes.byref(actual_format),
            ctypes.byref(nitems),
            ctypes.byref(bytes_after),
            ctypes.byref(data),
        )
        if status != 0 or not data or nitems.value < 1:
            return None
        try:
            # A 32-bit X property is handed back as an array of C long (8 bytes each on LP64).
            return int(ctypes.cast(data, ctypes.POINTER(ctypes.c_ulong))[0])
        finally:
            self._xlib.XFree(data)

    def _parent(self, win: int) -> int:
        root = ctypes.c_ulong()
        parent = ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        nchildren = ctypes.c_uint()
        ok = self._xlib.XQueryTree(
            self._disp,
            win,
            ctypes.byref(root),
            ctypes.byref(parent),
            ctypes.byref(children),
            ctypes.byref(nchildren),
        )
        if children:
            self._xlib.XFree(children)
        if not ok or parent.value == root.value:
            return 0  # reached the root without finding a PID
        return int(parent.value)


class _XErrorEvent(ctypes.Structure):
    """``XErrorEvent`` from ``Xlib.h`` -- only its head is read, to log which request failed."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("resourceid", ctypes.c_ulong),
        ("serial", ctypes.c_ulong),
        ("error_code", ctypes.c_ubyte),
        ("request_code", ctypes.c_ubyte),
        ("minor_code", ctypes.c_ubyte),
    ]


_X_ERROR_HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(_XErrorEvent))


def _log_and_ignore_x_error(_disp, event) -> int:
    err = event.contents
    log.debug(
        "key state: ignoring X error %d on request %d.%d (window 0x%x)",
        err.error_code,
        err.request_code,
        err.minor_code,
        err.resourceid,
    )
    return 0


#: Module-level so ctypes keeps the trampoline alive for as long as Xlib may call it.
_IGNORE_X_ERRORS = _X_ERROR_HANDLER(_log_and_ignore_x_error)


def _load_xlib() -> ctypes.CDLL:
    """libX11 with the prototypes this module calls (see also :mod:`roqsim.window_branding`)."""
    xlib = ctypes.CDLL("libX11.so.6")
    ulong_p = ctypes.POINTER(ctypes.c_ulong)
    xlib.XOpenDisplay.restype = ctypes.c_void_p
    xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    xlib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    xlib.XInternAtom.restype = ctypes.c_ulong
    xlib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    xlib.XStringToKeysym.restype = ctypes.c_ulong
    xlib.XStringToKeysym.argtypes = [ctypes.c_char_p]
    xlib.XKeysymToKeycode.restype = ctypes.c_ubyte
    xlib.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    xlib.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char * 32]
    xlib.XGetInputFocus.argtypes = [ctypes.c_void_p, ulong_p, ctypes.POINTER(ctypes.c_int)]
    xlib.XGetWindowProperty.restype = ctypes.c_int
    xlib.XGetWindowProperty.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_ulong,
        ulong_p,
        ctypes.POINTER(ctypes.c_int),
        ulong_p,
        ulong_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
    ]
    xlib.XQueryTree.restype = ctypes.c_int
    xlib.XQueryTree.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ulong_p,
        ulong_p,
        ctypes.POINTER(ulong_p),
        ctypes.POINTER(ctypes.c_uint),
    ]
    xlib.XFree.argtypes = [ctypes.c_void_p]
    xlib.XSetErrorHandler.restype = ctypes.c_void_p
    xlib.XSetErrorHandler.argtypes = [ctypes.c_void_p]
    return xlib
