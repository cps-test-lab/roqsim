"""Rebrand MuJoCo's hardcoded ``MuJoCo : `` window title to ``Roqsim: <model>``.

MuJoCo's C++ ``Simulate`` sets the GLFW window title to ``"MuJoCo : <model name>"`` once
per model load (``glfwSetWindowTitle``) and exposes no Python hook to change it. We rename
the top-level X11 window out-of-band via libX11 (ctypes), swapping the ``MuJoCo : `` prefix
for ``Roqsim: `` so the window reads ``Roqsim: <world name>`` -- or just ``Roqsim`` when the
world carries no meaningful name, so MuJoCo's default ``MuJoCo Model`` is never shown. (The engine
also names the model itself, from ``sim.name`` or ``Roqsim``, so the native title is meaningful even
where this X11 rename can't run.)

Cosmetic and X11-only. The window is created asynchronously on MuJoCo's render thread and
its title is set slightly after ``launch_passive`` returns, so we poll for a short window
from a daemon thread. On Wayland/macOS, without a ``DISPLAY``, or if libX11 is unavailable,
this is a silent no-op -- a window title is not an artifact, so best-effort is the right
altitude here rather than a hard failure.

The poll walks *other clients'* top-level windows, so a window closing mid-pass (including our own
viewer at shutdown) makes ``XGetWindowProperty`` fail -- and Xlib's default error handler kills the
whole process. :func:`_watch_and_rename` therefore runs under an ignoring error handler; see there.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time

log = logging.getLogger(__name__)

#: MuJoCo prepends this to the model name for the passive-viewer window title.
_MUJOCO_PREFIX = "MuJoCo : "
#: What we replace it with, so the window reads ``Roqsim: <world name>``.
_ROQSIM_PREFIX = "Roqsim: "
#: Brand shown alone when the world carries no meaningful name -- never "MuJoCo Model".
_ROQSIM_BRAND = "Roqsim"


def _display_title(name: str) -> str:
    """The window title for a world name: ``Roqsim: <name>``, or just ``Roqsim`` when the name is
    absent or MuJoCo's default -- so ``MuJoCo Model`` is never shown."""
    if not name or name in ("MuJoCo Model", _ROQSIM_BRAND, _ROQSIM_BRAND.lower()):
        return _ROQSIM_BRAND
    return _ROQSIM_PREFIX + name


# X11 constants
_ANY_PROPERTY_TYPE = 0
_XA_STRING = 31
_PROP_MODE_REPLACE = 0


def _model_name(model) -> str:
    """The model name MuJoCo puts in the title: the first NUL-terminated token of ``names``."""
    raw = bytes(model.names)
    nul = raw.find(b"\x00")
    return raw[: nul if nul >= 0 else len(raw)].decode("utf-8", "replace")


def retitle_window_async(
    model, *, name: str | None = None, timeout_s: float = 4.0
) -> threading.Thread | None:
    """Spawn a daemon watcher that renames this process's MuJoCo viewer window.

    ``name`` overrides the title's world name (from the world YAML's ``sim.name``); it defaults
    to MuJoCo's model name, which is what MuJoCo itself would have shown.

    Returns the watcher thread (already started), or ``None`` if we won't attempt it
    (non-X11 platform, no display, empty name).
    """
    if sys.platform != "linux" or not os.environ.get("DISPLAY"):
        return None
    name = name or _model_name(model)
    if not name:
        return None

    thread = threading.Thread(
        target=_watch_and_rename, args=(name, timeout_s), name="roqsim-retitle", daemon=True
    )
    thread.start()
    return thread


class _XErrorEvent(ctypes.Structure):
    """``XErrorEvent`` from ``Xlib.h`` -- only read to log which request failed."""

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
        "window retitle: ignoring X error %d on request %d.%d (window 0x%x)",
        err.error_code,
        err.request_code,
        err.minor_code,
        err.resourceid,
    )
    return 0


#: Module-level so ctypes keeps the trampoline alive for as long as Xlib may call it.
_IGNORE_X_ERRORS = _X_ERROR_HANDLER(_log_and_ignore_x_error)


def _watch_and_rename(name: str, timeout_s: float) -> None:
    try:
        xlib = _load_xlib()
    except OSError as err:
        log.debug("window retitle skipped: libX11 unavailable (%s)", err)
        return

    disp = xlib.XOpenDisplay(None)
    if not disp:
        log.debug("window retitle skipped: cannot open X display %r", os.environ.get("DISPLAY"))
        return

    # Xlib's default error handler prints the request and calls exit(1) -- so a window that closes
    # between listing _NET_CLIENT_LIST and reading its properties (our own viewer window on the way
    # out, or any other client's) would take the process down over a cosmetic title. Errors here mean
    # exactly "that window is gone, skip it", so ignore them. The handler is process-global (Xlib has
    # no per-connection hook), which is why it is installed only for this poll and restored after.
    prev_handler = xlib.XSetErrorHandler(ctypes.cast(_IGNORE_X_ERRORS, ctypes.c_void_p))
    try:
        atoms = {
            key: xlib.XInternAtom(disp, key.encode(), False)
            for key in ("_NET_CLIENT_LIST", "_NET_WM_PID", "_NET_WM_NAME", "UTF8_STRING", "WM_NAME")
        }
        pid = os.getpid()
        deadline = time.monotonic() + timeout_s
        renamed = False
        while time.monotonic() < deadline:
            for win in _top_level_windows(xlib, disp, atoms):
                title = _get_text(xlib, disp, win, atoms["_NET_WM_NAME"])
                if title is None or not title.startswith(_MUJOCO_PREFIX):
                    continue
                if not _belongs_to(xlib, disp, win, atoms["_NET_WM_PID"], pid):
                    continue
                _set_title(xlib, disp, win, atoms, _display_title(name))
                renamed = True
            xlib.XFlush(disp)
            # Keep polling after the first hit: a reset/reload re-sets the prefix, and a
            # freshly mapped window may not carry _NET_WM_PID yet on the first pass.
            time.sleep(0.1)
        if not renamed:
            log.debug(
                "window retitle: no %r window found for pid %d within %.1fs",
                _MUJOCO_PREFIX,
                pid,
                timeout_s,
            )
    finally:
        xlib.XSetErrorHandler(prev_handler)
        xlib.XCloseDisplay(disp)


def _load_xlib() -> ctypes.CDLL:
    xlib = ctypes.CDLL("libX11.so.6")
    ulong_p = ctypes.POINTER(ctypes.c_ulong)
    ubyte_pp = ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))
    xlib.XOpenDisplay.restype = ctypes.c_void_p
    xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    xlib.XCloseDisplay.argtypes = [ctypes.c_void_p]
    xlib.XInternAtom.restype = ctypes.c_ulong
    xlib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    xlib.XDefaultRootWindow.restype = ctypes.c_ulong
    xlib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
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
        ubyte_pp,
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
    xlib.XChangeProperty.restype = ctypes.c_int
    xlib.XChangeProperty.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    xlib.XFree.argtypes = [ctypes.c_void_p]
    xlib.XFlush.argtypes = [ctypes.c_void_p]
    # Handlers are passed/returned as opaque pointers so the previous one can be restored as-is.
    xlib.XSetErrorHandler.restype = ctypes.c_void_p
    xlib.XSetErrorHandler.argtypes = [ctypes.c_void_p]
    return xlib


def _get_property(xlib, disp, win, prop, req_type):
    """Return ``(format, nitems, data_ptr)`` for a window property, or ``None``.

    Caller must ``XFree`` the returned pointer. 32-bit properties come back as an array of
    C ``long`` (8 bytes each on LP64), a well-known Xlib quirk the readers below account for.
    """
    actual_type = ctypes.c_ulong()
    actual_format = ctypes.c_int()
    nitems = ctypes.c_ulong()
    bytes_after = ctypes.c_ulong()
    data = ctypes.POINTER(ctypes.c_ubyte)()
    status = xlib.XGetWindowProperty(
        disp,
        win,
        prop,
        0,
        (1 << 24),
        False,
        req_type,
        ctypes.byref(actual_type),
        ctypes.byref(actual_format),
        ctypes.byref(nitems),
        ctypes.byref(bytes_after),
        ctypes.byref(data),
    )
    if status != 0 or not data:
        return None
    if actual_format.value == 0 or nitems.value == 0:
        xlib.XFree(data)
        return None
    return actual_format.value, nitems.value, data


def _top_level_windows(xlib, disp, atoms) -> list[int]:
    """Managed top-level windows: ``_NET_CLIENT_LIST`` if the WM sets it, else root children."""
    root = xlib.XDefaultRootWindow(disp)
    got = _get_property(xlib, disp, root, atoms["_NET_CLIENT_LIST"], _ANY_PROPERTY_TYPE)
    if got is not None:
        fmt, n, data = got
        try:
            if fmt == 32:
                arr = ctypes.cast(data, ctypes.POINTER(ctypes.c_ulong * n)).contents
                return list(arr)
        finally:
            xlib.XFree(data)

    # Fallback: enumerate the root's direct children.
    root_ret = ctypes.c_ulong()
    parent_ret = ctypes.c_ulong()
    children = ctypes.POINTER(ctypes.c_ulong)()
    nchildren = ctypes.c_uint()
    if not xlib.XQueryTree(
        disp,
        root,
        ctypes.byref(root_ret),
        ctypes.byref(parent_ret),
        ctypes.byref(children),
        ctypes.byref(nchildren),
    ):
        return []
    try:
        return [children[i] for i in range(nchildren.value)]
    finally:
        if children:
            xlib.XFree(children)


def _get_text(xlib, disp, win, prop) -> str | None:
    got = _get_property(xlib, disp, win, prop, _ANY_PROPERTY_TYPE)
    if got is None:
        return None
    _fmt, n, data = got
    try:
        return ctypes.string_at(data, n).decode("utf-8", "replace")
    finally:
        xlib.XFree(data)


def _belongs_to(xlib, disp, win, prop, pid) -> bool:
    got = _get_property(xlib, disp, win, prop, _ANY_PROPERTY_TYPE)
    if got is None:
        # No _NET_WM_PID (some WMs/toolkits omit it): the "MuJoCo : " prefix + our own
        # display is a strong enough signal on its own, so accept it.
        return True
    fmt, _n, data = got
    try:
        if fmt != 32:
            return True
        return ctypes.cast(data, ctypes.POINTER(ctypes.c_ulong)).contents.value == pid
    finally:
        xlib.XFree(data)


def _set_title(xlib, disp, win, atoms, title: str) -> None:
    utf8 = title.encode("utf-8")
    xlib.XChangeProperty(
        disp,
        win,
        atoms["_NET_WM_NAME"],
        atoms["UTF8_STRING"],
        8,
        _PROP_MODE_REPLACE,
        utf8,
        len(utf8),
    )
    # Legacy WM_NAME for anything that ignores _NET_WM_NAME.
    xlib.XChangeProperty(
        disp, win, atoms["WM_NAME"], _XA_STRING, 8, _PROP_MODE_REPLACE, utf8, len(utf8)
    )
