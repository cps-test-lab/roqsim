"""Describe what a world provides, as one line of JSON.

``roqsim scenes inputs`` answers what a world *needs*. This answers what it *offers*, which is
what a caller holding an override needs to know::

    roqsim scenes describe worlds/depot.yaml
    {"world": "...", "packaged": false,
     "inputs": [...],
     "components": [{"address": "robot.lidar", "ref": "lidar",
                     "paths": ["components.robot.lidar.rays", ...]}],
     "addresses": ["robot", "robot.lidar", ...],
     "entities": null,
     "overridable": {"fields": [{"field": "geom_friction", "does": ..., "caveats": ...}, ...],
                     "targets": null},
     "dropped_transport": [], "errors": null}

Why a command and not an import, for the same reason ``inputs`` is one: the caller is usually
*not* a roqsim process. A campaign runner validating ``components.floorplan.frixion`` before it
spends an image pull has no reason to have roqsim installed, and cannot resolve a world's
``extends`` chain without it -- so it asks the image that does.

**Override paths.** ``components`` reports every component that will RUN -- the document's own entries
and everything its models' manifests contribute -- under the ``address`` an override names it by,
with the dotted paths into its config that already exist. ``addresses`` is that set on its own, and
it is exactly what resolution accepts: a caller can check a sweep key against it before spending an
image pull.

That equality is the point. The payload used to be built from the document's own entries, so the set
published and the set an override could reach were the same -- and both excluded everything a model
manifest supplied, which is precisely what a campaign wants to vary. A path not listed is still not
necessarily invalid: a plugin may accept a key its world leaves at the default, so a caller reports
an unlisted *path* as unverifiable rather than as wrong. What the list settles is the expensive
mistake -- an *address* matching nothing, refused at load time inside the container, after the pull
and the schedule.

**Entities** are ``null`` unless ``--entities`` is passed, because naming them means compiling
the model: which entities exist is settled at compile time (roqsim never recompiles mid-run, and
``simulation_interfaces`` serves no ``SpawnEntity``), so there is no cheaper way to ask. A
caller checking that a scenario only drives entities the world actually has pays for it; one
resolving paths does not.

**Overrides.** ``--override FILE`` applies a nested override tree before anything is described,
the same spelling and the same file ``roqsim sim --override`` takes. It matters for the build-fed
halves: which entities a world compiles depends on its plugins' config, so a campaign whose
obstacles come from its own overrides compiles them only with those overrides applied. Without
the flag this command answers about the world the FILE declares, which is a different world than
the run's -- and a caller comparing a scenario's entity names against that answer reads a working
campaign as a broken one. An address the world does not have is still refused
here, exactly as it is refused at load time, which is the cheap mistake this command exists to catch
before an image pull.

**Overridable model values.** ``overridable.fields`` is the allowlist of model values the
``model_override`` plugin can change while a run is in progress, each with what it does and the way it
can silently do nothing. It costs nothing and is always present: ``mjModel``'s field set is a property
of MuJoCo rather than of this world, so it needs no model built. ``overridable.targets`` is the
world-specific half -- the *names* an override can select and their current values -- and it is
``null`` unless ``--overridable GLOB`` is passed, because it needs the compiled model for the same
reason ``--entities`` does. The glob is not a convenience: a world of this size has hundreds of geoms,
and a caller that wants the gripper's pads should not be handed the scene. Unnamed geoms are omitted
because nothing can address them.

**Body tree.** ``--body-tree GLOB`` answers a different question than ``--overridable``: not "what
can I change on named objects" but "what is actually *nested under* this body" -- its descendant
bodies and each one's attached geoms/joints/sites, as a tree rather than a flat list. Same discipline
as ``--overridable``: a glob is required (there is no "the whole world" mode), and each matched body's
subtree is capped in node count, reporting ``truncated`` rather than silently handing back a partial
tree with no indication it was cut.

**The build has no transport in it**, and ``dropped_transport`` names what went. A describe publishes
nothing, so a world's bridge is dead weight here exactly as it is for ``roqsim render`` and the exporters
-- and since the ROS bridge ships in a colcon package, in a pip-only environment it does not even
resolve, which used to fail the build over plugins that contribute no geometry. Only *identified*
transport goes (:func:`roqsim.config.drop_transport`, not the lenient ``drop_transport_plugins``): a
misspelt geometry plugin must stay the loud failure it is, because dropping it would leave an entity
missing and let a caller conclude the world does not have it. The ``components`` list above is computed
before the drop and still reports the bridge, so an override addressing it is still checkable.

**Half an answer is still an answer, and says so.** When only the build fails -- the branches above,
not loading the world -- the reply is printed with ``errors.build`` set and the build-fed keys left
``null``, so a caller keeps the half that cost nothing (which plugin keys exist) instead of losing the
lot. The exit code stays non-zero: ``0`` goes on meaning "fully answered", and a caller that reads only
the status must not be told a partial reply was a complete one.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path

import yaml

from roqsim.config import drop_transport, load_config, world_sources
from roqsim.world import resolve_world_yaml_ref

#: Depth at which a dotted path stops being a *destination* and starts being data. A campaign
#: overrides ``components.floorplan.floor.reflectance``; it does not address individual members of
#: a list of obstacle instances, and listing those would bury the keys that matter.
_MAX_DEPTH = 4


def _paths(value, prefix: str, depth: int = 0) -> list[str]:
    """Dotted paths into a plugin's config, deepest-first within each key."""
    if depth >= _MAX_DEPTH or not isinstance(value, dict) or not value:
        return [prefix]
    out: list[str] = []
    for key, item in value.items():
        out.extend(_paths(item, f"{prefix}.{key}", depth + 1))
    return out


def _describe_components(config) -> list:
    """Every component that will RUN, under the address an override names it by.

    Read off the effective list, so a sensor a model's manifest supplies is here even though no entry
    in the document mentions it. It used to be zipped against the document's own entries, which meant
    the set this command published and the set an override could reach were the same -- and both
    excluded everything a manifest contributed, which is exactly what a campaign wants to sweep.
    """
    return [
        {
            "address": spec.address,
            "ref": spec.ref,
            "name": spec.name,
            "entity": spec.entity,
            "enabled": spec.enabled,
            "origin": "document" if spec in config.declared else "manifest",
            "paths": sorted(_paths(spec.config, f"components.{spec.address}")),
        }
        for spec in config.plugins
    ]


def _overridable_fields() -> list:
    """The allowlist, as the plugin states it. One copy: the runtime guard reads the same rows.

    Imported here rather than at module scope for the same reason the engine is: listing a world's
    paths must not pay for importing MuJoCo.
    """
    from roqsim.plugins.model_override import field_catalog

    return field_catalog()


@contextmanager
def _built(config):
    """The world's compiled context, for the questions that need one. One build serves them all.

    Imported here rather than at module scope: describing a world's *paths* must not pay for
    importing the engine, and these are the only branches that need it.
    """
    from roqsim.engine import Engine

    engine = Engine(config)
    engine.setup()
    try:
        yield engine.ctx
    finally:
        engine.shutdown()


def _overridable_targets(ctx, pattern: str) -> dict:
    """Named objects matching *pattern*, with the current value of every field they can be overridden by.

    Grouped by the namespace an override addresses them in, because that is what a caller has to put
    in ``select:``. A geom also reports its owning body and its ``priority`` -- read-only context, and
    the thing ``geom_friction``'s caveat is about: at equal priority MuJoCo takes the element-wise
    maximum of the two geoms' friction, so which side governs a contact decides whether an override
    does anything at all.
    """
    import mujoco

    from roqsim.plugins.model_override import field_catalog

    by_namespace: dict[str, list[str]] = {}
    for row in field_catalog():
        by_namespace.setdefault(row["namespace"], []).append(row["field"])

    objects = {
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
        "joint": mujoco.mjtObj.mjOBJ_JOINT,
    }
    counts = {
        "geom": ctx.model.ngeom,
        "body": ctx.model.nbody,
        "actuator": ctx.model.nu,
        "joint": ctx.model.njnt,
    }

    targets: dict[str, list[dict]] = {}
    for namespace, fields in sorted(by_namespace.items()):
        rows = []
        for oid in range(counts[namespace]):
            name = mujoco.mj_id2name(ctx.model, objects[namespace], oid)
            if not name or not fnmatch(name, pattern):
                continue  # unnamed objects cannot be addressed, so they are not targets
            row = {"name": name}
            if namespace == "geom":
                body = int(ctx.model.geom_bodyid[oid])
                row["body"] = mujoco.mj_id2name(ctx.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                row["geom_priority"] = int(ctx.model.geom_priority[oid])
            for field in fields:
                row[field] = _plain(getattr(ctx.model, field)[oid])
            rows.append(row)
        if rows:
            targets[namespace] = rows
    return targets


#: Node cap per matched body's subtree -- the same role _MAX_DEPTH plays for override paths: a
#: broad glob still gets a bounded answer, never "the scene" (which can be hundreds of geoms).
_MAX_TREE_NODES = 200


def _body_tree(ctx, pattern: str) -> list[dict]:
    """Each body matching *pattern*, with its kinematic subtree -- descendant bodies plus each
    one's attached geoms/joints/sites -- as ``{name, type, children}`` nodes.

    Unlike ``_overridable_targets``, which lists named objects flat, this nests them the way the
    model actually does: a gripper's pads and finger joint show up *under* the gripper body, not
    beside it. Same discipline as ``_overridable_targets`` -- a glob is required, and each match's
    subtree is capped (``_MAX_TREE_NODES``) rather than risking "the scene" on a broad glob.
    """
    import mujoco

    children_of: dict[int, list[int]] = {}
    for body_id in range(1, ctx.model.nbody):  # body 0 is the worldbody; it has no parent
        children_of.setdefault(int(ctx.model.body_parentid[body_id]), []).append(body_id)

    geoms_of: dict[int, list[int]] = {}
    for geom_id in range(ctx.model.ngeom):
        geoms_of.setdefault(int(ctx.model.geom_bodyid[geom_id]), []).append(geom_id)
    joints_of: dict[int, list[int]] = {}
    for joint_id in range(ctx.model.njnt):
        joints_of.setdefault(int(ctx.model.jnt_bodyid[joint_id]), []).append(joint_id)
    sites_of: dict[int, list[int]] = {}
    for site_id in range(ctx.model.nsite):
        sites_of.setdefault(int(ctx.model.site_bodyid[site_id]), []).append(site_id)

    def _name(kind, oid) -> str:
        return mujoco.mj_id2name(ctx.model, kind, oid) or ""

    def _build(body_id: int, budget: list[int]) -> dict | None:
        """None once *budget* (a one-element mutable counter) is exhausted."""
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        node = {"name": _name(mujoco.mjtObj.mjOBJ_BODY, body_id), "type": "body"}
        children = []
        for geom_id in sorted(
            geoms_of.get(body_id, []), key=lambda g: _name(mujoco.mjtObj.mjOBJ_GEOM, g)
        ):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            children.append({"name": _name(mujoco.mjtObj.mjOBJ_GEOM, geom_id), "type": "geom"})
        for joint_id in sorted(
            joints_of.get(body_id, []), key=lambda j: _name(mujoco.mjtObj.mjOBJ_JOINT, j)
        ):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            children.append({"name": _name(mujoco.mjtObj.mjOBJ_JOINT, joint_id), "type": "joint"})
        for site_id in sorted(
            sites_of.get(body_id, []), key=lambda s: _name(mujoco.mjtObj.mjOBJ_SITE, s)
        ):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            children.append({"name": _name(mujoco.mjtObj.mjOBJ_SITE, site_id), "type": "site"})
        for child_id in sorted(
            children_of.get(body_id, []), key=lambda b: _name(mujoco.mjtObj.mjOBJ_BODY, b)
        ):
            child = _build(child_id, budget)
            if child is None:
                break
            children.append(child)
        if children:
            node["children"] = children
        return node

    results = []
    for body_id in range(1, ctx.model.nbody):
        name = _name(mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or not fnmatch(name, pattern):
            continue  # unnamed bodies cannot be addressed, so they are not roots either
        budget = [_MAX_TREE_NODES]
        tree = _build(body_id, budget)
        results.append({"root": name, "tree": tree, "truncated": budget[0] <= 0})
    results.sort(key=lambda r: r["root"])
    return results


def _plain(value):
    """A numpy row as JSON: a list for a vector, a number for a scalar."""
    if getattr(value, "ndim", 0):
        return [_plain(v) for v in value]
    return int(value) if str(getattr(value, "dtype", "")).startswith("int") else float(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="roqsim scenes describe",
        description="Describe what a world provides, as JSON on stdout.",
    )
    parser.add_argument(
        "world", help="world YAML path, or a package ref such as 'roqsim_scenes:depot'"
    )
    parser.add_argument(
        "--entities",
        action="store_true",
        help="also list the entities the world compiles (builds the model)",
    )
    parser.add_argument(
        "--overridable",
        metavar="GLOB",
        default="",
        help="also list the geoms/bodies/actuators matching GLOB that model_override can change, "
        "with their current values (builds the model)",
    )
    parser.add_argument(
        "--override",
        metavar="FILE",
        default="",
        help="YAML file of overrides to apply before describing, exactly as `roqsim sim "
        "--override` takes them (the answer is then about the world a run with those "
        "overrides would load)",
    )
    parser.add_argument(
        "--body-tree",
        metavar="GLOB",
        default="",
        help="also nest the descendant bodies (and their geoms/joints/sites) under each body "
        "matching GLOB, as a tree rather than a flat list (builds the model)",
    )
    args = parser.parse_args(argv)

    target, packaged = args.world, False
    if ":" in target and not Path(target).exists():
        try:
            resolved = resolve_world_yaml_ref(target)
        except FileNotFoundError as err:
            print(f"cannot resolve world {target!r}: {err}", file=sys.stderr)
            return 1
        if resolved is None:
            print(
                f"{target!r} is not a world ref (no such roqsim.worlds provider)", file=sys.stderr
            )
            return 1
        target, packaged = str(resolved), True
    if not Path(target).exists():
        print(f"world {target!r} does not exist", file=sys.stderr)
        return 1

    world = Path(target).resolve()
    overrides = None
    if args.override:
        # A caller holding overrides is asking about the world its RUN will load, not about the
        # file: the entities a campaign's own obstacle placement compiles in exist only once its
        # overrides are applied, so describing the base world answered a different question than
        # the one asked -- and a caller comparing entity names against that answer concluded a
        # working campaign was broken.
        if not Path(args.override).exists():
            print(f"overrides file {args.override!r} does not exist", file=sys.stderr)
            return 1
        try:
            with open(args.override, encoding="utf-8") as handle:
                overrides = yaml.safe_load(handle) or {}
        except Exception as err:  # noqa: BLE001 - the caller gets the reason, not a traceback
            print(f"cannot read overrides {args.override}: {err}", file=sys.stderr)
            return 1
    try:
        config = load_config(world, overrides)
    except Exception as err:  # noqa: BLE001 - the caller gets the reason, not a traceback
        print(f"cannot load world {world}: {err}", file=sys.stderr)
        return 1

    result = {
        "world": str(world),
        "packaged": packaged,
        "inputs": [str(p) for p in world_sources(world)],
        "components": _describe_components(config),
        # The set an override may name, published so a caller can check a sweep key before spending
        # an image pull. It is exactly what resolution accepts -- asserted by round-tripping each
        # one through the resolver in the tests, because a list that drifted from it would send a
        # caller looking for a mistake that is not there.
        "addresses": sorted(spec.address for spec in config.plugins),
        "entities": None,
        # The allowlist is world-independent, so it costs nothing and is always here. Its
        # world-specific half needs the model, hence the flag -- see the module docstring.
        "overridable": {"fields": _overridable_fields(), "targets": None},
        "body_tree": None,
        # Both are properties of the answer below, so they are here even when there is nothing to
        # say -- a caller reading `entities` has to be able to see how it was arrived at.
        "dropped_transport": [],
        "errors": None,
    }
    if args.entities or args.overridable or args.body_tree:
        # The scene is what these questions are about, and a describe publishes nothing, so the
        # transport goes before the build -- see the module docstring for why the strict variant.
        # `components` is already in `result`, so the bridge stays in the reported list.
        dropped = drop_transport(config)
        if dropped:
            result["dropped_transport"] = dropped
            print(
                f"describing the scene without transport: dropped {', '.join(dropped)}",
                file=sys.stderr,
            )
        # ONE build, however many of these were asked for: compiling this world is the expensive part.
        try:
            with _built(config) as ctx:
                if args.entities:
                    result["entities"] = sorted(ctx.entities.names())
                if args.overridable:
                    result["overridable"]["targets"] = _overridable_targets(ctx, args.overridable)
                if args.body_tree:
                    result["body_tree"] = _body_tree(ctx, args.body_tree)
        except Exception as err:  # noqa: BLE001
            # The half that needed no build is still worth having, so the reply is printed with the
            # reason attached -- and the exit code still says it is not a whole answer.
            print(f"cannot build world {world}: {err}", file=sys.stderr)
            result["errors"] = {"build": str(err)}
            print(json.dumps(result))
            return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
