# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0

"""The socket backend: the simulator is another process, reached over the control socket it serves.

``roqsim sim`` serves every endpoint of its world at a control address (:mod:`roqsim.ipc`); this
backend is a client of it (:class:`roqsim.control_client.Client`), under the ROS runner or any
other that hands an action no in-process simulation. The simulator is found as ``roqsim ls`` finds
it -- ``ROQSIM_CONTROL``, the run directory's socket, or the only one running -- and until one
answers, :meth:`IpcAccess.ready` is false, so a simulator that is still starting is waited for.

**Nothing blocks.** Every request is sent and then polled with a zero wait, one in flight per
call object; the tree's tick never waits on the socket.

The world's endpoints are listed once, when the simulator first answers, and every name a
scenario uses is resolved against that list with the same functions the in-process backend uses
(:func:`~scenario_execution_roqsim.access.find_endpoint`), so a refusal reads the same on both.
"""

from __future__ import annotations

import numpy as np

from . import (
    AccessError,
    CommandCall,
    CommandOutcome,
    NavCall,
    NavOutcome,
    Pose,
    ReportCall,
    ReportReading,
    SpawnCall,
    SpawnOutcome,
    TeleportCall,
    TeleportOutcome,
    WorldAccess,
    command_payload,
    find_endpoint,
    no_navigator,
    no_report,
    published_field,
    report_value,
)

#: Where the core serves an entity's pose, and entity placement and presence.
_POSE = "sim/entities/{name}/pose"
_SET_STATE = "sim/entities/set_state"
_SET_PRESENCE = "sim/entities/set_presence"
#: The exception a producer raises for a name the world does not carry (roqsim.entity_control).
_UNKNOWN = "UnknownEntity"


class _Request:
    """One request in flight: sent once, polled without waiting."""

    def __init__(self, access: IpcAccess, op: str, value=None, **fields):
        self._access = access
        self._rid = access._client.send(op, value, **fields)
        self.done = False
        self.value = None
        self.error = None

    def poll(self) -> bool:
        """``True`` once answered; then ``value`` or ``error`` (a ControlError) is set."""
        if self.done:
            return True
        from roqsim.ipc import ControlError

        try:
            self.done, self.value = self._access._client.poll(self._rid, 0.0)
        except ControlError as err:
            self.done, self.error = True, err
        return self.done


def _refusal(err) -> tuple[bool, str]:
    """``(authoring, message)``: whether a refusal names something the world does not carry."""
    return (err.detail or {}).get("exception") == _UNKNOWN, str(err)


class _IpcCommand(CommandCall):
    """A command sent with ``call``. A bare value is sent once the command's parameters are known
    (a ``describe`` first), so it reaches a typed command as :func:`command_payload` shapes it."""

    def __init__(self, access: IpcAccess, path: str, value):
        self._access, self._path, self._value = access, path, value
        self._describe = None
        self._request = None
        if value is None or isinstance(value, dict):
            self._request = _Request(access, "call", value, path=path)
        else:
            self._describe = _Request(access, "describe", path=path)

    def poll(self) -> CommandOutcome | None:
        if self._request is None:
            if not self._describe.poll():
                return None
            if self._describe.error is not None:
                return CommandOutcome(ok=False, detail=str(self._describe.error))
            params = self._describe.value.get("params")
            names = None if params is None else [p["name"] for p in params]
            payload = command_payload(names, self._value)
            self._request = _Request(self._access, "call", payload, path=self._path)
        request = self._request
        if not request.poll():
            return None
        if request.error is not None:
            return CommandOutcome(ok=False, detail=str(request.error))
        reply = request.value or {}
        if reply.get("queued"):
            return CommandOutcome(ok=True, detail="queued")
        confirmation = reply.get("confirmation")
        verdict = ""
        if isinstance(confirmation, dict):
            verdict = str(confirmation.get("verified") or "")
        return CommandOutcome(
            ok=True,
            result=reply.get("result"),
            confirmation=confirmation,
            verified=verdict,
            confirmed=reply.get("verified", True) is not False,
            detail=str(reply.get("note", "")),
        )


class _IpcPlacement(TeleportCall, SpawnCall):
    """A placement or presence flip: an ``entity_control`` command, its refusal a result."""

    def __init__(self, access: IpcAccess, path: str, value, outcome):
        self._request = _Request(access, "call", value, path=path)
        self._outcome = outcome

    def poll(self):
        request = self._request
        if not request.poll():
            return None
        if request.error is not None:
            authoring, message = _refusal(request.error)
            if authoring:
                raise AccessError(message)
            return self._outcome(ok=False, detail=message)
        return self._outcome(ok=True, detail=str((request.value or {}).get("result", "")))


class _IpcRoute(NavCall):
    """A route sent to a navigator's endpoint, followed by its sequence number.

    The same shape as the in-process route: the write returns the route's sequence number, and
    ``route_status`` says which sequence the navigator has applied and whether it finished.
    """

    def __init__(self, access: IpcAccess, owner_path: str, endpoint: str, value, *, wait: bool):
        self._access = access
        self._base = owner_path
        self._wait = wait
        self._send = _Request(access, "call", value, path=f"{owner_path}/{endpoint}")
        self._seq = None
        self._status = None

    def poll(self):
        if self._seq is None:
            if not self._send.poll():
                return None
            if self._send.error is not None:
                raise AccessError(str(self._send.error))
            self._seq = int((self._send.value or {}).get("result"))
            if not self._wait:
                return NavOutcome(True, "route accepted")
        if self._status is None:
            self._status = _Request(self._access, "read", path=f"{self._base}/route_status")
        if not self._status.poll():
            return None
        status, self._status = self._status, None
        if status.error is not None:
            raise AccessError(str(status.error))
        applied, finished = int(status.value["seq"]), bool(status.value["finished"])
        if applied > self._seq:
            return NavOutcome(False, "a newer route preempted this one")
        if applied == self._seq and finished:
            return NavOutcome(True, "arrived")
        return None

    def cancel(self) -> None:
        self._access._client.send("call", None, path=f"{self._base}/cancel_route")


class _IpcReport(ReportCall):
    def __init__(self, access: IpcAccess, row: dict, entity: str, report: str, field: str):
        self._access = access
        self._path = row["path"]
        self._name = f"{entity}.{report}"
        self._field = field
        self._published = None
        self._describe = _Request(access, "describe", path=self._path)
        self._read = None

    def poll(self) -> ReportReading | None:
        if self._published is None:
            if not self._describe.poll():
                return None
            if self._describe.error is not None:
                raise AccessError(str(self._describe.error))
            self._published = published_field(self._describe.value.get("hints") or {})
        if self._read is None:
            self._read = _Request(self._access, "read", path=self._path)
        if not self._read.poll():
            return None
        read, self._read = self._read, None
        if read.error is not None:
            raise AccessError(str(read.error))
        if read.value is None:  # the producer has nothing to report yet
            return None
        return report_value(self._name, read.value, self._field, self._published, "control socket")


class IpcAccess(WorldAccess):
    transport = "control socket"

    def __init__(self, control: str | None = None):
        self._control = control
        self._client = None
        self._rows: list[dict] | None = None
        self._listing: _Request | None = None
        self._waiting_for = "no simulator answered yet"
        self._poses: dict[str, _Request] = {}

    # -- the world ------------------------------------------------------------------------------
    def ready(self) -> bool:
        if self._rows is not None:
            return True
        from roqsim.ipc import ControlError

        if self._client is None:
            try:
                from roqsim.control_client import Client

                self._client = Client(self._control)
            except ControlError as err:
                self._waiting_for = str(err)
                return False
        if self._listing is None:
            self._listing = _Request(self, "describe", path="")
        if not self._listing.poll():
            return False
        listing, self._listing = self._listing, None
        if listing.error is not None:
            self._waiting_for = str(listing.error)
            return False
        self._rows = list(listing.value)
        return True

    def pending_reason(self) -> str:
        """Why :meth:`ready` is still false, for an action's waiting message."""
        return self._waiting_for

    def _require_rows(self) -> list[dict]:
        if self._rows is None:
            raise AccessError("the simulator has not answered yet; call ready() first")
        return self._rows

    def _paths(self) -> set[str]:
        return {row["path"] for row in self._require_rows()}

    def entity_pose(self, name: str) -> Pose | None:
        path = _POSE.format(name=name)
        if path not in self._paths():
            entities = sorted(
                row["path"].split("/")[2]
                for row in self._rows
                if row["path"].startswith("sim/entities/") and row["path"].endswith("/pose")
            )
            raise AccessError(
                f"the simulator has no entity called {name!r} with a body, so it has no pose to "
                f"read. Known entities: {', '.join(entities) or 'none'}."
            )
        request = self._poses.get(name)
        if request is None:
            request = self._poses[name] = _Request(self, "read", path=path)
        if not request.poll():
            return None
        del self._poses[name]
        if request.error is not None:
            raise AccessError(str(request.error))
        if request.value is None:
            raise AccessError(
                f"entity: entity {name!r} is ABSENT (roqsim.presence): nothing can see or touch "
                "it, so its pose is not a fact about the trial. Make it present first."
            )
        pose = request.value
        return Pose(pos=np.asarray(pose["position"]), quat=np.asarray(pose["orientation"]))

    # -- commands -------------------------------------------------------------------------------
    def call_endpoint(self, entity: str, endpoint: str, value=None) -> CommandCall:
        row = find_endpoint(self._require_rows(), entity, endpoint, kind="in")
        return _IpcCommand(self, row["path"], value)

    # -- reports --------------------------------------------------------------------------------
    def entity_report(self, entity: str, report: str, field: str = "") -> ReportCall:
        rows = self._require_rows()
        try:
            row = find_endpoint(rows, entity, report, kind="out")
        except AccessError:
            is_entity = _POSE.format(name=entity) in self._paths()
            raise no_report(rows, entity, report, is_entity=is_entity) from None
        return _IpcReport(self, row, entity, report, field)

    # -- navigation -----------------------------------------------------------------------------
    def _navigator(self, name: str) -> str:
        """The path its route endpoints live under, or the refusal naming what can navigate."""
        rows = self._require_rows()
        found = [
            row["path"].rpartition("/")[0]
            for row in rows
            if row["owner"] == name and row["name"] == "navigate_through_poses"
        ]
        if not found:
            offered = [row["owner"] for row in rows if row["name"] == "navigate_through_poses"]
            raise no_navigator(name, offered)
        return found[0]

    def navigate(self, name: str, goal_poses, *, wait: bool) -> NavCall:
        base = self._navigator(name)
        poses = [[float(p[0]), float(p[1])] for p in goal_poses]
        return _IpcRoute(self, base, "navigate_through_poses", {"poses": poses}, wait=wait)

    def start_route(self, name: str, *, wait: bool) -> NavCall:
        return _IpcRoute(self, self._navigator(name), "start_route", None, wait=wait)

    # -- placement and presence ----------------------------------------------------------------
    def _entity_command(self, path: str) -> str:
        if path not in self._paths():
            raise AccessError(
                f"the simulator serves no {path}: entity placement and presence are served by "
                "`roqsim sim` (its `sim.entities` component), which this simulator does not run"
            )
        return path

    def set_entity_state(
        self, name: str, pos: np.ndarray, quat: np.ndarray, lin=None, ang=None
    ) -> TeleportCall:
        value = {"entity": name, "position": pos, "orientation": quat}
        if lin is not None:
            value["linear_velocity"] = lin
        if ang is not None:
            value["angular_velocity"] = ang
        return _IpcPlacement(self, self._entity_command(_SET_STATE), value, TeleportOutcome)

    def set_entity_presence(self, name: str, present: bool, pos=None, quat=None) -> SpawnCall:
        value = {"entity": name, "present": bool(present)}
        if pos is not None:
            value["position"] = pos
        if quat is not None:
            value["orientation"] = quat
        return _IpcPlacement(self, self._entity_command(_SET_PRESENCE), value, SpawnOutcome)

    def teardown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
