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

Everything this reports is knowable elsewhere, just not in one place: it takes ``roqsim scenes
inputs`` to learn whether the files resolve, ``roqsim scenes describe`` to learn what the world
contains, ``roqsim render --check`` to learn whether it compiles, and a run to learn whether the
plugins agree with the model. Four commands, three of them in a package a world need not depend on,
each stopping at the first thing it happens to look at.

``roqsim check`` does the whole load once and reports **every** problem it found, in the order the
loader would hit them::

    roqsim check worlds/depot_nav.yaml
    roqsim check roqsim_mobile:husky_demo --json

Six stages, each of which can fail without the next being meaningless:

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
``reset``
    every plugin's ``on_reset`` runs, as it does before each trial, and leaves the state the trial
    starts from: keyframes and home poses applied, props re-seated.

What it does **not** do is step the simulation. A world that passes here can still behave wrongly;
what it cannot do is fail to start, which is the failure worth catching before a campaign queues a
thousand of them.

**Warnings** are what the load found that does not stop a world from starting but is likely to make
it misbehave -- each one ``{"check", "message", "hint"}``, and none of them clears ``ok``. One
check that produces them is ``interpenetration``: the reset state puts two bodies inside one
another deeper than the contact's tolerance (an arm's ``home`` that buries its tool in the table, a
prop spawned into another), which the contact solver resolves on the first steps with forces large
enough to fling them (:mod:`roqsim.interpenetration`). A warning rather than a problem because the
world does start, the tolerance is a judgement rather than a rule, and an overlap can be deliberate;
the run itself logs the same finding at reset, so it is in the run's record too. The others are the
flex checks under ``derived`` below.

The inventory it prints when there are no problems is the other half: the entities that registered,
the endpoints they publish (with topics), the model's own totals, and each flex -- what it compiled
into (dim, vertices, elements, ``dof`` mode, pins, parent body, entity, elastic, passive contact).
That is what an agent or a person needs to write the next thing -- a scenario that drives ``robot``,
a bridge that expects ``scan`` -- without opening the world file and its manifests to work out what
is in there.

``derived`` is what the model will *do*, worked out without stepping it: for each flex its first
elastic modes, the damping ratio each rings down with at this timestep and the integrator's share of
it, whether the timestep resolves each mode, and whether its contact ``solref`` is above the floor
MuJoCo raises it to (:mod:`roqsim.flex_modes`: numerical damping under ``discrete``, the resolution
limit, the ``solref`` floor). The modes are
the one costly computation here -- two passive-force evaluations per flex DOF and an eigen solve,
seconds at most -- and a flex above :data:`roqsim.flex_modes.MODES_DOF_CAP` DOFs is reported
without them rather than analysed at any cost. Three warnings come from it, for what such a world
will do that its author probably did not intend: ``flex-damping`` (its damping is mostly the
integrator's), ``flex-timestep`` (the timestep under-resolves a reported mode, so that mode's damping
ratio and frequency are not the ones that run, and are marked so) and ``flex-solref`` (its contact
stiffness is not the one that runs). A flex's warning also names the flex in an extra ``flex`` key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: Stage names, in the order they run. A problem in one does not stop the report; it stops that
#: world from reaching the next stage, which is stated rather than implied by an empty section.
STAGES = ("resolve", "inputs", "config", "build", "configure", "reset")


def _problem(stage: str, message: str, hint: str | None = None) -> dict:
    problem = {"stage": stage, "message": message}
    if hint:
        problem["hint"] = hint
    return problem


def _warning(check: str, message: str, hint: str | None = None, **extra) -> dict:
    """A finding that does not clear ``ok``: ``{"check", "message", "hint"}``, plus any *extra* keys
    a check adds for a caller that filters by them (a flex's warning names the flex in ``flex``)."""
    warning = {"check": check, "message": message}
    if hint:
        warning["hint"] = hint
    warning.update(extra)
    return warning


def check_world(target: str) -> dict:
    """Load *target* as far as it goes and report what happened, as plain data.

    Returns ``{"target", "ok", "reached", "problems": [...], "warnings": [...], "world": {...},
    "derived": {...}, "inputs": [...]}``. ``reached`` is the last stage that completed, so a caller
    can tell "the config is wrong" from "the config is fine and the model does not compile" without
    parsing messages. ``warnings`` never affect ``ok``: they are things a world that loads will do
    that its author probably did not mean.
    """
    from roqsim.config import PluginError, load_config

    report: dict = {
        "target": target,
        "ok": False,
        "reached": None,
        "problems": [],
        "warnings": [],
        "inputs": [],
        "world": {},
        "derived": {},
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
        engine = Engine(cfg, preview=True)
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

    # -- reset: the state a trial starts from ------------------------------------------------
    try:
        engine.reset()
    except Exception as exc:  # noqa: BLE001 - a plugin's on_reset failing is a trial that cannot start
        report["problems"].append(_problem("reset", f"{type(exc).__name__}: {exc}"))
        _shutdown(engine)
        return report
    report["reached"] = "reset"
    from roqsim.interpenetration import as_warnings

    report["warnings"].extend(as_warnings(engine.interpenetrations))

    try:
        report["world"] = _inventory(engine)
        report["derived"], flex_warnings = _derive(engine)
        report["warnings"].extend(_warning(**warning) for warning in flex_warnings)
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
    from roqsim.flex_modes import describe_flexes

    flexes = describe_flexes(model, {e["name"]: e["body"] for e in entities})
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
            "nflex": int(model.nflex),
            "timestep": float(model.opt.timestep),
            "gravity": [float(v) for v in model.opt.gravity],
        },
        # "implicitfast", the spelling a world writes in `sim.integrator` -- not the enum's
        # mjINT_IMPLICITFAST, which is a name for the C header and not for a config key.
        "integrator": mujoco.mjtIntegrator(model.opt.integrator)
        .name.removeprefix("mjINT_")
        .lower(),
        # Whether it was stated or chosen by `sim.integrator: auto`, and for auto, the flex that
        # decided it -- the reason a world that never named an integrator runs under `discrete`.
        "integrator_reason": engine.integrator.reason,
        # What each flex compiled into: dim, vertices, elements, dof mode, pins, parent, entity,
        # whether it is elastic and has passive contact. Fields only -- the costly half is _derive.
        "flexes": flexes,
    }


def _derive(engine) -> tuple[dict, list[dict]]:
    """What the loaded world will *do*, worked out from the model without stepping it.

    For each flex: its first elastic modes, the damping ratio each will ring down with at this
    timestep and how much of it is the integrator's, whether the timestep resolves each mode well
    enough for those figures to hold, and whether its contact ``solref`` is one MuJoCo
    will actually use (:func:`roqsim.flex_modes.explain_flex`). The modes are the one costly thing
    ``check`` does -- a finite-difference stiffness and an eigen solve per flex, seconds at most,
    refused above :data:`roqsim.flex_modes.MODES_DOF_CAP` degrees of freedom -- and nothing but
    ``check`` computes them.

    Returns ``(derived, warnings)``. A warning names something a world that loads fine will do and
    its author probably did not intend; it never makes the check fail.
    """
    from roqsim.flex_modes import explain_flex

    model = engine.ctx.model
    derived: dict = {"flexes": []}
    warnings: list[dict] = []
    for flex_id in range(model.nflex):
        row, flex_warnings = explain_flex(model, flex_id)
        derived["flexes"].append(row)
        warnings.extend(flex_warnings)
    return derived, warnings


def _shutdown(engine) -> None:
    """Release whatever the partial setup took (a renderer's GL context, a file, a node)."""
    try:
        engine.shutdown()
    except Exception:  # noqa: BLE001 - teardown of a half-built world is best effort
        pass


def _render_warnings(report: dict) -> list[str]:
    lines = []
    for warning in report.get("warnings", []):
        lines.append(f"WARN  [{warning['check']}] {warning['message']}")
        if warning.get("hint"):
            lines.append(f"      hint: {warning['hint']}")
    return lines


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
    lines.append("ok    loads, compiles, every component resolved, and it resets")
    if report.get("warnings"):
        lines.append("")
        lines.extend(_render_warnings(report))
    lines.append("")
    lines.append(
        f"model: {model['nbody']} bodies, {model['ngeom']} geoms, {model['njnt']} joints, "
        f"{model['nu']} actuators, {model['nsensor']} sensors, {model['ncam']} cameras"
    )
    lines.append(
        f"       timestep {model['timestep']}s, integrator {world['integrator']} "
        f"({world['integrator_reason']})"
    )
    lines.extend(_render_flexes(world.get("flexes", []), report.get("derived", {})))
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


def _render_flexes(flexes: list[dict], derived: dict) -> list[str]:
    """The flex block of the text report: one line of what each is, two of what it will do."""
    if not flexes:
        return []
    by_name = {row["name"]: row for row in derived.get("flexes", [])}
    lines = ["", f"flexes ({len(flexes)}):"]
    for flex in flexes:
        count = f"{flex['vertices']} vertices, {flex['elements']} elements"
        dof = flex["dof"] if flex["dof"] == "full" else f"{flex['dof']} ({flex['nodes']} nodes)"
        owner = f" (entity {flex['entity']})" if flex["entity"] else ""
        traits = [
            "rigid" if flex["rigid"] else ("elastic" if flex["elastic"] else "not elastic"),
            "passive contact" if flex["passive_contact"] else "no passive contact",
        ]
        lines.append(
            f"  {flex['name']}  dim {flex['dim']}, {count}, dof {dof}, {flex['pinned']} pinned, "
            f"on {flex['parent']}{owner}; {', '.join(traits)}"
        )
        row = by_name.get(flex["name"])
        if row is None:
            continue
        modes = row.get("modes")
        if modes is None:
            lines.append(f"       modes not computed: {row.get('modes_skipped', '')}")
        elif modes:
            # A mode the timestep under-resolves is starred: its figures are not the ones that run.
            star = ["*" if m.get("resolved") is False else "" for m in modes]
            hz = ", ".join(f"{m['hz']:.3g}{s}" for m, s in zip(modes, star, strict=True))
            zeta = ", ".join(f"{m['zeta']:.3g}{s}" for m, s in zip(modes, star, strict=True))
            share = row.get("numerical_share")
            numerical = (
                f" (numerical share {share:.0%} at timestep {row['timestep']:g} s)"
                if share is not None
                else ""
            )
            lines.append(
                f"       modes {hz} Hz; damping ratio {zeta} at damping {row['damping']:g} s"
                f"{numerical}"
            )
            if any(star):
                from roqsim.flex_modes import MAX_OMEGA_DT

                lines.append(
                    f"       * under-resolved (omega * timestep above {MAX_OMEGA_DT:g}): a run "
                    "damps and rings a starred mode differently -- see the flex-timestep warning"
                )
        floor = row.get("solref_floor")
        solref = " ".join(f"{v:g}" for v in row["solref"])
        lines.append(
            f"       contact solref {solref} ({row['solref_source']}), "
            + (f"floor {floor:g} s" if floor is not None else "no floor (refsafe disabled)")
        )
    return lines


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
