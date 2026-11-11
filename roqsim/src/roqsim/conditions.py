# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""World conditions: what a world loaded as, said where every result of it is kept.

A plugin that loads a world in a weaker form than asked for -- baked lighting whose bake is missing,
so the world is drawn unbaked -- does not stop the world, and a log line alone is lost the moment the
run ends. A result made that way has to be told from one that was not, afterwards and without the
log. So the plugin *raises a condition*, and the condition travels with the world:

* it is kept on the context (:attr:`roqsim.context.SimContext.conditions`) and logged once;
* it is written into the compiled model as the text element :data:`MODEL_TEXT`, so whatever holds
  only the model -- a renderer, a recording's reader, another process that compiled the same
  world -- can read it with :func:`of_model`;
* a recording's provenance (``conditions``), ``roqsim render``'s JSON record, ``roqsim check``'s
  warnings and the run's output directory (:data:`RUN_FILE`, beside the run's other artifacts, which
  is what a campaign collects) carry it.

A condition is plain data: ``source`` (the plugin's address), ``code`` (a stable identifier a reader
filters by, ``bake_missing``), ``level`` (``warning`` for a weaker world, ``info`` for a statement
of what was used), ``message`` (what happened, in words), ``fix`` (the next step, a command where
there is one) and ``details`` (anything further, JSON-serialisable).

    from roqsim.conditions import raise_condition
    raise_condition(ctx, source=self.address, code="bake_missing", message="...", fix="roqsim bake ...")

A condition is raised during the build (``build`` / ``build_late``): it describes how the world was
built, and the model it is written into compiles after the last build hook.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import mujoco

#: The model text element the conditions are written into, as one JSON list.
MODEL_TEXT = "roqsim:conditions"

#: The file :func:`write_run_file` writes into the run's output directory.
RUN_FILE = "roqsim_conditions.json"

#: The environment variables naming the run's own output directory, most specific first: the same
#: anchors a relative ``ROQSIM_RECORD`` resolves against (:mod:`roqsim.runner`).
OUTPUT_DIR_VARS = ("RUN_OUTPUT_DIR", "OUTPUT_DIR")

LEVELS = ("warning", "info")


def make(
    *, source: str, code: str, message: str, fix: str = "", level: str = "warning", details=None
) -> dict:
    """A condition as plain data (see the module docstring)."""
    if level not in LEVELS:
        raise ValueError(f"condition level {level!r}: one of {LEVELS}")
    out = {"source": source, "code": code, "level": level, "message": message, "fix": fix}
    if details:
        out["details"] = details
    # Plain data or nothing: a condition that cannot be written is caught where it is raised.
    json.dumps(out)
    return out


def raise_condition(ctx, *, log: logging.Logger | None = None, **fields) -> dict:
    """Add a condition to ``ctx.conditions`` and log it once; returns the condition."""
    condition = make(**fields)
    ctx.conditions.append(condition)
    logger = log or ctx.logger
    text = condition["message"] + (f" -- {condition['fix']}" if condition["fix"] else "")
    if condition["level"] == "warning":
        logger.warning("%s: %s", condition["source"], text)
    else:
        logger.info("%s: %s", condition["source"], text)
    return condition


def write_to_spec(spec: mujoco.MjSpec, conditions: list[dict]) -> None:
    """Write *conditions* into *spec* as :data:`MODEL_TEXT`, replacing an earlier one."""
    for text in list(spec.texts):
        if text.name == MODEL_TEXT:
            spec.delete(text)
    if conditions:
        spec.add_text(name=MODEL_TEXT, data=json.dumps(conditions, sort_keys=True))


def of_model(model: mujoco.MjModel) -> list[dict]:
    """The conditions a compiled model carries (``[]`` for a model built without any)."""
    try:
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, MODEL_TEXT)
    except Exception:  # noqa: BLE001 - a model without names at all carries no conditions either
        return []
    if tid < 0:
        return []
    adr, size = int(model.text_adr[tid]), int(model.text_size[tid])
    raw = bytes(model.text_data[adr : adr + size]).split(b"\0", 1)[0]
    return json.loads(raw.decode()) if raw else []


def as_check_warnings(conditions: list[dict]) -> list[dict]:
    """The warning-level conditions as ``roqsim check`` warnings (``check: condition``)."""
    out = []
    for c in conditions:
        if c.get("level") != "warning":
            continue
        warning = {"check": "condition", "message": f"{c['source']}: {c['message']}", "code": c["code"]}
        if c.get("fix"):
            warning["hint"] = c["fix"]
        out.append(warning)
    return out


def run_dir(environ=None) -> Path | None:
    """The run's own output directory, from :data:`OUTPUT_DIR_VARS`, or ``None``."""
    environ = os.environ if environ is None else environ
    for var in OUTPUT_DIR_VARS:
        if environ.get(var):
            return Path(environ[var])
    return None


def write_run_file(conditions: list[dict], directory: Path | None = None) -> Path | None:
    """Write *conditions* to :data:`RUN_FILE` in the run's output directory; ``None`` if it has none.

    Written for every run that has an output directory, an empty list included: a file saying "no
    conditions" is what tells a reader the run was checked, where a missing one says nothing.
    """
    directory = directory if directory is not None else run_dir()
    if directory is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUN_FILE
    path.write_text(json.dumps({"conditions": conditions}, indent=1, sort_keys=True) + "\n")
    return path
