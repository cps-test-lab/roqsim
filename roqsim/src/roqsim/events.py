# Copyright (C) 2026 Frederik Pasch
#
# SPDX-License-Identifier: Apache-2.0

"""A run's events: what happened when, stamped on simulated time, recorded with the run.

A state recording says where everything was; it cannot say what a plugin decided. A machine that
started a stroke, a controller that began a move, a gripper that confirmed a hold -- each is an
*event* a reader wants with its time, to build a timeline of the run without re-running it.

Any plugin emits one on the physics thread::

    ctx.events.emit("press_1", "door_close", part="strip_3")

and gets back the event as a dict: ``{"t": <sim seconds>, "episode": <n>, "source": "press_1",
"event": "door_close", "part": "strip_3"}``. Extra fields must be JSON values (a ``TypeError`` at
the emit, not a broken file at the end). The log keeps every event of the process, across resets;
``episode`` tells the trials apart.

A recording (:class:`roqsim.capture.StateRecorder`) writes each event as one message on its
``events`` channel, with the event's own time as the message's log time, so an event between two
samples keeps its exact time. :attr:`roqsim.recording.Recording.events` reads them back in order.
"""

from __future__ import annotations

import json
from collections.abc import Callable

#: The keys an event carries whatever its source adds; an emit may not use them for fields.
RESERVED = ("t", "episode", "source", "event")


class EventLog:
    """The events of a run, in the order they were emitted."""

    def __init__(self, clock: Callable[[], float], episode: Callable[[], int]) -> None:
        self._clock = clock
        self._episode = episode
        self._events: list[dict] = []

    def emit(self, source: str, event: str, **fields) -> dict:
        """Record that ``source`` did ``event`` now; ``fields`` are JSON values describing it."""
        clash = sorted(set(fields) & set(RESERVED))
        if clash:
            raise ValueError(f"event fields may not be named {clash}: they are the event's own")
        record = {
            "t": float(self._clock()),
            "episode": int(self._episode()),
            "source": str(source),
            "event": str(event),
            **fields,
        }
        json.dumps(record)  # a field that is no JSON value fails here, at its source
        self._events.append(record)
        return record

    def __len__(self) -> int:
        return len(self._events)

    def since(self, index: int) -> list[dict]:
        """The events emitted after the first ``index``."""
        return self._events[index:]

    def all(self) -> list[dict]:
        return list(self._events)
