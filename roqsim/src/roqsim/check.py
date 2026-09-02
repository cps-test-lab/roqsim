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

"""Would this world load, and what would it be? One command, before any compute is spent.

Everything this reports was already knowable, and that was the problem: it took ``roqsim scenes
inputs`` to learn whether the files resolve, ``roqsim scenes describe`` to learn what the world
contains, ``roqsim render --check`` to learn whether it compiles, and a run to learn whether the
plugins agree with the model. Four commands, three of them in a package a world need not depend on,
each stopping at the first thing it happens to look at.

``roqsim check`` does the whole load once and reports **every** problem it found, in the order the
loader would hit them::

    roqsim check worlds/depot_nav.yaml
    roqsim check roqsim_mobile:husky_demo --json

Five stages, each of which can fail without the next being meaningless:

``resolve``
    the target names a world (a path, or a ``<package>:<world>`` ref that a provider answers).
``inputs``
    the files the world is defined by -- its MJCF, its meshes, the world it ``extends`` -- listed,
    so the answer to "what has to travel with this" comes out of the same call.
``config``
    the document parses and every plugin accepts its own config. This is where the aggregated
    per-plugin validation lands, so a world with three bad keys reports three, not the first.
``build``
    the model compiles: every plugin's ``build`` runs and MuJoCo accepts the result.
``configure``
    every plugin resolves what it needs *in* the compiled model -- the body a sensor is mounted on,
    the actuator a controller drives. Most "it loaded and then died" failures are here, and they are
    exactly the ones a syntax check cannot see.

What it does **not** do is step the simulation. A world that passes here can still behave wrongly;
what it cannot do is fail to start, which is the failure worth catching before a campaign queues a
thousand of them.

The inventory it prints when there are no problems is the other half: the entities that registered,
the endpoints they publish (with topics), and the model's own totals. That is what an agent or a
person needs to write the next thing -- a scenario that drives ``robot``, a bridge that expects
``scan`` -- without opening the world file and its manifests to work out what is in there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Stage names, in the order they run. A problem in one does not stop the report; it stops that
#: world from reaching the next stage, which is stated rather than implied by an empty section.
STAGES = ("resolve", "inputs", "config", "build", "configure")


def _problem(stage: str, message: str, hint: str | None = None) -> dict:
    problem = {"stage": stage, "message": message}
    if hint:
        problem["hint"] = hint
    return problem


def check_world(target: str) -> dict:
    """Load *target* as far as it goes and report what happened, as plain data.

    Returns ``{"target", "ok", "reached", "problems": [...], "world": {...}, "inputs": [...]}``.
    ``reached`` is the last stage that completed, so a caller can tell "the config is wrong" from
    "the config is fine and the model does not compile" without parsing messages.
    """
    from roqsim.config import PluginError, load_config

    report: dict = {
        "target": target,
        "ok": False,
        "reached": None,
        "problems": [],
        "inputs": [],
        "world": {},
    }

    # -- resolve ---------------------------------------------------------------------------
    path = _resolve(target, report)
    if path is None:
        return report
    report["reached"] = "resolve"

    # -- config (which also expands `extends` and resolves every plugin ref) -----------------
    try:
        cfg = load_config(path)
    except PluginError as exc:
        # The aggregated one: every plugin's validation errors, in one message.
        report["problems"].append(_problem("config", str(exc)))
        return report
    except (OSError, ValueError) as exc:
        report["problems"].append(_problem("config", f"{type(exc).__name__}: {exc}"))
        return report

    # -- inputs ------------------------------------------------------------------------------
    # After parsing rather than before it: the list of files a world is defined by is not known
    # until its `extends` chain and its models are resolved, which is what loading does.
    report["inputs"] = _inputs(path, report)

    # -- config, part two: the plugins accept what they were given -----------------------------
    # Constructing the engine is what runs `validate_config` on every plugin and aggregates the
    # errors. It belongs to this stage, not to `build`: nothing has been compiled yet, and the
    # report would otherwise blame the model for a typo in a key.
    from roqsim.engine import Engine

    try:
        engine = Engine(cfg)
    except PluginError as exc:
        report["problems"].append(_problem("config", str(exc)))
        return report
    except Exception as exc:  # noqa: BLE001 - a plugin's constructor is the world's problem too
        report["problems"].append(_problem("config", f"{type(exc).__name__}: {exc}"))
        return report
    report["reached"] = "config"

    try:
        engine.setup()
    except Exception as exc:  # noqa: BLE001 - any plugin's failure is this command's finding
        # Which half of setup() failed, asked of the context rather than guessed: `build` hooks and
        # the compile happen before there is a model, `configure` after. Reporting the wrong one
        # sends a reader to the wrong file -- an unresolvable site is a name that does not exist in
        # a model that compiled fine.
        stage = "configure" if getattr(engine.ctx, "model", None) is not None else "build"
        report["problems"].append(
            _problem(
                stage,
                f"{type(exc).__name__}: {exc}",
                hint=(
                    "a plugin refused what the compiled model offers -- check the names it resolves "
                    "(bodies, sites, actuators) against `roqsim catalog model <model>`"
                    if stage == "configure"
                    else None
                ),
            )
        )
        _shutdown(engine)
        return report
    report["reached"] = "configure"

    try:
        report["world"] = _inventory(engine)
        report["ok"] = True
    finally:
        _shutdown(engine)
    return report


def _resolve(target: str, report: dict) -> Path | None:
    """The world file *target* names, or ``None`` with the reason recorded."""
    from roqsim.world import resolve_world_yaml_ref

    path = Path(target)
    if path.is_file():
        return path
    if ":" in target and not path.exists():
        try:
            resolved = resolve_world_yaml_ref(target)
        except FileNotFoundError as exc:
            report["problems"].append(_problem("resolve", str(exc)))
            return None
        if resolved:
            return Path(resolved)
        report["problems"].append(
            _problem(
                "resolve",
                f"{target!r} names no known 'roqsim.worlds' provider",
                hint="`roqsim catalog worlds` lists the refs that resolve here",
            )
        )
        return None
    report["problems"].append(
        _problem(
            "resolve",
            f"{target!r} is neither a file nor a '<package>:<world>' ref",
            hint="`roqsim catalog worlds` lists what this installation has",
        )
    )
    return None


def _inputs(path: Path, report: dict) -> list[str]:
    """Every file this world is defined by -- the same walk ``roqsim scenes inputs`` prints.

    Reported even though the load succeeded, because it is the other question asked at the same
    moment: what has to travel with this world into a container, and what would make something
    compiled from it stale. :func:`roqsim.config.world_sources` is best-effort by contract (it
    yields what resolved rather than raising), which is why a short list is not itself a problem
    here -- an input that actually matters and is missing fails the stages above.
    """
    from roqsim.config import world_sources

    try:
        return [str(p) for p in world_sources(path)]
    except Exception as exc:  # noqa: BLE001 - a listing failure is a finding, not a crash
        report["problems"].append(_problem("inputs", f"{type(exc).__name__}: {exc}"))
        return []


def _inventory(engine) -> dict:
    """What the loaded world turned out to be -- the half of this that is not about failure."""
    import mujoco

    ctx = engine.ctx
    model = ctx.model
    endpoints = []
    for endpoint in ctx.interface.all():
        hints = endpoint.backend.get("ros2", {})
        endpoints.append(
            {
                "name": endpoint.name,
                "direction": endpoint.direction,
                "owner": endpoint.owner,
                "namespace": endpoint.namespace,
                "type": hints.get("type") or hints.get("service"),
                "topic": hints.get("topic") or hints.get("name"),
                "rate_hz": endpoint.rate_hz,
            }
        )
    entities = [
        {
            "name": entity.name,
            "kind": entity.kind,
            "body": entity.body,
            "namespace": entity.meta.get("namespace", ""),
            "prefix": entity.meta.get("prefix", ""),
        }
        for entity in (ctx.entities.get(n) for n in ctx.entities.names())
    ]
    return {
        "components": [
            {"address": spec.address, "ref": spec.ref, "enabled": spec.enabled}
            for spec in engine.config.plugins
        ],
        "entities": entities,
        "endpoints": sorted(endpoints, key=lambda e: (e["owner"] or "", e["name"])),
        "model": {
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "njnt": int(model.njnt),
            "nu": int(model.nu),
            "nsensor": int(model.nsensor),
            "ncam": int(model.ncam),
            "timestep": float(model.opt.timestep),
            "gravity": [float(v) for v in model.opt.gravity],
        },
        # "implicitfast", the spelling a world writes in `sim.integrator` -- not the enum's
        # mjINT_IMPLICITFAST, which is a name for the C header and not for a config key.
        "integrator": mujoco.mjtIntegrator(model.opt.integrator)
        .name.removeprefix("mjINT_")
        .lower(),
    }


def _shutdown(engine) -> None:
    """Release whatever the partial setup took (a renderer's GL context, a file, a node)."""
    try:
        engine.shutdown()
    except Exception:  # noqa: BLE001 - teardown of a half-built world is best effort
        pass


def _render_text(report: dict) -> str:
    lines = [f"world: {report['target']}"]
    if report["problems"]:
        lines.append("")
        for problem in report["problems"]:
            lines.append(f"FAIL  [{problem['stage']}] {problem['message']}")
            if problem.get("hint"):
                lines.append(f"      hint: {problem['hint']}")
        reached = report["reached"] or "nothing"
        lines.append("")
        lines.append(f"reached: {reached} (of {' -> '.join(STAGES)})")
        return "\n".join(lines)

    world = report["world"]
    model = world["model"]
    lines.append("ok    loads, compiles, and every component resolved")
    lines.append("")
    lines.append(
        f"model: {model['nbody']} bodies, {model['ngeom']} geoms, {model['njnt']} joints, "
        f"{model['nu']} actuators, {model['nsensor']} sensors, {model['ncam']} cameras"
    )
    lines.append(f"       timestep {model['timestep']}s, integrator {world['integrator']}")
    if world["entities"]:
        lines.append("")
        lines.append("entities:")
        for entity in world["entities"]:
            scope = f" (namespace {entity['namespace']})" if entity["namespace"] else ""
            lines.append(f"  {entity['name']}  [{entity['kind']}] on {entity['body']}{scope}")
    if world["endpoints"]:
        lines.append("")
        lines.append("endpoints:")
        for endpoint in world["endpoints"]:
            topic = endpoint["topic"] or endpoint["name"]
            lines.append(f"  {endpoint['direction']:3s} {topic:34s} {endpoint['type'] or ''}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim check",
        description="Load a world as far as it goes and report every problem at once.",
    )
    parser.add_argument("world", help="a world YAML path, or a '<package>:<world>' ref")
    parser.add_argument("--json", action="store_true", help="report as JSON rather than as text")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    from . import logging_setup

    logging_setup.configure(verbose=args.verbose)

    report = check_world(args.world)
    print(json.dumps(report, indent=2) if args.json else _render_text(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
