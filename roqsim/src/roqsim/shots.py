"""A **shot**: one moment of one recording, framed, written down so it can be drawn again.

A **public API**. The replay window appends shots to a file as a person picks them, and whatever
draws the figures reads that file back and renders each one::

    from roqsim.shots import read_shots, render_args

    for doc in read_shots("shots.yaml"):
        subprocess.run(["roqsim", "render", *render_args(doc, size="1920x1080")], cwd=doc["project"])

The file is a multi-document YAML -- one ``---`` per shot, appended, never rewritten -- so a picking
session that is interrupted keeps every shot it had already taken.

:func:`render_args` is the only place ``roqsim render`` flags are built. The window's own render
button and every consumer go through it, so what a shot promises and what a render does cannot
disagree. Three of its rules are load-bearing:

* **No world target.** :meth:`roqsim.recording.Recording.build` rebuilds from the recording's own
  resolved component tree only while no target is passed; naming the world instead re-resolves the
  overrides against today's copy of it. ``world`` in a document identifies the recording, and is not
  an argument.
* **Overrides are not re-passed.** For a recording, ``roqsim render`` takes only ``sim.view`` from the
  command line; everything else is already in the provenance.
* **A tracked world is un-tracked explicitly.** ``--view`` merges *over* the recording world's own
  ``sim.view``, and :class:`roqsim.viewer.TrackingCamera` is active for any ``track`` target -- so a
  hand-framed shot in a world that tracks the robot would render tracked, with ``lookat`` ignored.
  Such a shot carries ``track: null`` *and* ``follow_heading: false``: tracking off needs both, since
  ``follow_heading`` without a target is refused.

A shot framed by the camera the run was watched through carries no ``view`` at all, which is what
leaves ``roqsim render`` following the recorded camera.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

#: The document shape. A reader that does not know a version refuses the file rather than reading
#: the keys that happen to overlap.
SHOT_SCHEMA = 1

#: Decimals a camera keeps, matching :mod:`roqsim.view_save` so a shot and a saved world view round
#: a pose the same way.
_LENGTH_DP = 3
_ANGLE_DP = 1

#: Decimals a sim time keeps. A recording's times are multiples of its timestep, so this reproduces
#: the sample exactly while staying readable.
_TIME_DP = 6

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def shot_document(
    rec,
    sample,
    camera=None,
    *,
    state: str | Path,
    project: str | Path = ".",
    label: str = "",
    size: str = "1920x1080",
    no_ceiling: bool = False,
    png: str | None = None,
    event: dict | None = None,
    source: dict | None = None,
    taken: tuple[str, ...] = (),
    world: str | Path | None = None,
) -> dict:
    """Describe ``sample`` of ``rec`` framed through ``camera``, as a document to append.

    ``camera`` is the free camera the moment was framed with; ``None`` means the shot follows the
    recording's own camera and therefore states no view. ``state`` is the recording's path as the
    render will be given it -- relative to ``project``, which is the directory the render runs in.
    ``taken`` is the ids already in the file, so this one does not collide with them. ``world`` is
    the world the replay was told to rebuild from, where the recording's own could not be: the
    render is then told the same, as ``world_target``.
    """
    view = None if camera is None else _view_for(camera, rec.view)
    doc = {
        "schema": SHOT_SCHEMA,
        "id": shot_id(label, sample.index, Path(state).parent.name, taken=taken),
        "label": label,
        "project": str(project),
        "state": str(state),
        "world": rec.meta.get("world") or "",
        "at": round(float(sample.sim_time), _TIME_DP),
        "sample_index": int(sample.index),
        "fps": float(rec.fps),
        "camera_source": "recorded" if camera is None else "free",
        "size": size,
        "no_ceiling": bool(no_ceiling),
    }
    if view is not None:
        doc["view"] = view
    if world:
        doc["world_target"] = str(world)
    doc["png"] = png or f"{doc['id']}.png"
    if event:
        doc["event"] = dict(event)
    if source:
        doc["source"] = dict(source)
    doc["provenance"] = {
        "packages": dict(rec.meta.get("packages") or {}),
        "samples": len(rec),
        "span": [round(v, _TIME_DP) for v in rec.span],
    }
    return doc


def _view_for(camera, world_view: dict | None) -> dict:
    """The ``sim.view`` that reproduces ``camera``, un-tracking a world that tracks.

    Built from the camera alone rather than through :func:`roqsim.view_save.view_from_camera`, which
    carries a world's tracking setup over: that is right for a world saving its own framing and wrong
    for a shot, where the pose the person flew to is the whole point.
    """
    view = {
        "lookat": [_round(v, _LENGTH_DP) for v in camera.lookat],
        "distance": _round(camera.distance, _LENGTH_DP),
        "azimuth": _round(camera.azimuth, _ANGLE_DP),
        "elevation": _round(camera.elevation, _ANGLE_DP),
    }
    if (world_view or {}).get("track") is not None:
        view["track"] = None
        view["follow_heading"] = False
    return view


def render_args(doc: dict, *, size: str | None = None, out: str | Path | None = None) -> list[str]:
    """The argv that draws ``doc``, everything after ``roqsim render``.

    ``size`` and ``out`` override what the document states, which is how one shot list renders at a
    second resolution without being rewritten.
    """
    _check_schema(doc)
    # The world comes first, as `roqsim render`'s target: named only where the recording's own
    # provenance cannot rebuild it, which is the one case a shot states one.
    args = [str(doc["world_target"])] if doc.get("world_target") else []
    args += ["--state", str(doc["state"])]
    args += _moment_args(doc)
    if focus := doc.get("focus"):
        # Before --view, which wins per key: the occlusion search picks a base camera and the stated
        # keys are applied on top, so a document can frame on a body and still fix its distance.
        args += ["--focus", *(str(name) for name in focus)]
    if view := doc.get("view"):
        args += ["--view", *_view_tokens(view)]
    if doc.get("no_ceiling"):
        args.append("--no-ceiling")
    args += ["--size", str(size or doc["size"])]
    target = out or doc.get("video") or doc.get("png")
    if not target:
        raise ValueError(
            f"shot {doc.get('id', '?')!r} says nothing about where to write: give it a 'png' (a "
            "moment) or a 'video' (a clip), or pass out=."
        )
    args += ["--out", str(target)]
    return args


def _moment_args(doc: dict) -> list[str]:
    """``--at`` for a shot, ``--from``/``--to`` for a clip.

    A **clip** is a shot with a range where a shot has a moment, which is the whole of the difference:
    it carries the same ``state``, ``view``, ``size`` and ``source``, so a framing picked in the replay
    window is usable as either without being rewritten.

    ``from: onset`` is passed through to ``roqsim render``, which resolves it against the recording --
    the caller does not need the recording open to build these arguments.

    Note what is **not** emitted: ``--fps``. A shot document's ``fps`` is provenance, the rate the
    recording was captured at, and turning it into a flag would silently restate every existing shot's
    capture rate as a playback rate. A clip states playback as ``speed`` instead.
    """
    at, start, stop = doc.get("at"), doc.get("from"), doc.get("to")
    if at is not None and (start is not None or stop is not None):
        raise ValueError(
            f"shot {doc.get('id', '?')!r} has both a moment ('at') and a range ('from'/'to'); it is "
            "one or the other."
        )
    if at is not None:
        return ["--at", _time(at)]
    if start is None and stop is None:
        raise ValueError(
            f"shot {doc.get('id', '?')!r} selects no moment: give it 'at' for a still, or "
            "'from'/'to' for a clip."
        )
    args = []
    if start is not None:
        args += ["--from", _time(start)]
    if stop is not None:
        args += ["--to", _time(stop)]
    if (speed := doc.get("speed")) is not None:
        args += ["--speed", str(float(speed))]
    return args


def _time(value) -> str:
    """A sim time as ``roqsim render`` takes it, or a keyword such as ``onset`` passed through."""
    if isinstance(value, str) and not value.replace(".", "", 1).replace("-", "", 1).isdigit():
        return value
    return f"{float(value):.{_TIME_DP}f}"


def _view_tokens(view: dict) -> list[str]:
    """``{"azimuth": -37.5}`` -> ``["azimuth=-37.5"]``, in ``--view``'s own grammar."""
    tokens = []
    for key, value in view.items():
        if value is None:
            tokens.append(f"{key}=null")
        elif isinstance(value, bool):
            tokens.append(f"{key}={'true' if value else 'false'}")
        elif isinstance(value, (list, tuple)):
            tokens.append(f"{key}=" + ",".join(_text(v) for v in value))
        else:
            tokens.append(f"{key}={_text(value)}")
    return tokens


def shot_id(label: str, index: int, fallback: str = "shot", *, taken=()) -> str:
    """A filename-safe id for one shot, distinct from the ids already ``taken``."""
    base = f"{_slug(label) or _slug(fallback) or 'shot'}-{int(index):04d}"
    if base not in set(taken):
        return base
    for suffix in range(2, 1000):
        if (candidate := f"{base}-{suffix}") not in set(taken):
            return candidate
    raise ValueError(f"cannot make an id unlike the {len(taken)} already in this file")


def format_shot(doc: dict) -> str:
    """One document, as it is written to the file: a ``---`` marker and the keys in their order."""
    return "---\n" + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


def append_shot(path: str | Path, doc: dict) -> int:
    """Append ``doc`` to the shots file and return how many documents it then holds.

    The count comes from reading the file back rather than from counting appends: a document that
    did not land whole is one every later reader trips over, and the caller shows this number.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(format_shot(doc))
    return len(read_shots(path))


def read_shots(path: str | Path) -> list[dict]:
    """Every shot in the file, in order. A file that does not exist yet holds none."""
    path = Path(path)
    if not path.exists():
        return []
    docs = [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]
    for doc in docs:
        _check_schema(doc, path)
    return docs


def _check_schema(doc: dict, path: str | Path | None = None) -> None:
    schema = (doc or {}).get("schema")
    if schema != SHOT_SCHEMA:
        where = f" in {path}" if path else ""
        raise ValueError(
            f"shot {(doc or {}).get('id', '?')!r}{where} states schema {schema!r}; this roqsim "
            f"reads schema {SHOT_SCHEMA}. Read it with the version that wrote it, or re-pick it."
        )


def _round(value, dp: int) -> float:
    # ``or 0.0`` folds -0.0, which a camera reaches routinely, onto 0.0.
    return round(float(value), dp) or 0.0


def _text(value) -> str:
    """A view value as a command line carries it.

    A number loses its exponent and its trailing zeros and is never a bare ``4.``; a name is passed
    through unchanged. ``track`` is the key that takes a name -- the body a camera follows -- so a
    value is not always a number even though most of them are.
    """
    if isinstance(value, str):
        return value
    text = f"{float(value):.{_LENGTH_DP}f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", str(text).strip().lower()).strip("_")
