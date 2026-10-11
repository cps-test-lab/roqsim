"""Where a scene export gets its model: a world YAML through the real build pipeline, or a bare MJCF.

Every exporter of a whole scene (``roqsim export web``, ``roqsim export gltf``) takes the same source
options, through :func:`add_source_options`, and compiles through :func:`compile_source`, so an export
is the scene the simulator builds and two exports of one world are exports of the same model.

* ``--world``: the world YAML, compiled with its plugins, reset, and optionally stepped
  (:func:`compile_from_world`);
* ``--mjcf``: a bare MJCF, compiled directly with no plugins and no reset
  (:func:`compile_from_mjcf`). The options that act on a world are refused with it
  (:func:`refuse_options_for_mjcf`) rather than ignored.
"""

from __future__ import annotations

import argparse
import errno
import json
import logging
from pathlib import Path

import mujoco

from .config import drop_transport_plugins, load_config, world_sources
from .engine import Engine
from .override_options import add_override_options, overrides_from_options, refuse_world_options


def add_source_options(parser: argparse.ArgumentParser) -> None:
    """Add the required source: ``--world`` or ``--mjcf``."""
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--world", help="path to the world YAML (compiled via the plugin pipeline)")
    source.add_argument(
        "--mjcf",
        help="path to a bare MJCF file (compiled directly, with no plugins and no reset, so "
        "--set, --override, --skip-plugins and --settle-steps are refused with it)",
    )


def add_world_options(parser: argparse.ArgumentParser) -> None:
    """Add the options that act on a world YAML: ``--skip-plugins``, overrides, ``--settle-steps``."""
    parser.add_argument(
        "--skip-plugins",
        default="",
        help="comma-separated plugin names/refs to drop before compiling, on top of the "
        "transport/bridge plugins (which contribute no geometry and are always dropped)",
    )
    # The options, and the merge, `roqsim sim` uses: a campaign whose overrides are a nested tree
    # (a list of obstacle instances, say) hands this exporter exactly what it handed the run.
    add_override_options(parser)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help="after reset(), step physics this many times before capturing state (e.g. to let a "
        "dropped free body settle). Default 0 -- capture the reset pose.",
    )


def add_manifest_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        help='also write a JSON source manifest to this path: {"inputs": [<file>, ...]} '
        "listing every file the world is defined by (the YAML, its 'extends' ancestors, the "
        "MJCF and its mesh/texture assets). A build system caching this export re-runs it when "
        "one of them changes -- the leaf world alone is not enough, since an inherited scene or "
        "a replaced mesh changes the result without touching it.",
    )


def refuse_options_for_mjcf(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse the world options with ``--mjcf``: the export would be geometry the caller believes is
    overridden, pruned or settled and is not."""
    if args.mjcf:
        refuse_world_options(
            parser,
            args,
            "--mjcf compiles a bare MJCF with no plugins and no reset -- export the world with "
            "--world instead",
        )


def compile_from_mjcf(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Compile a bare MJCF file directly (no plugins / world YAML). Initial state is the model default."""
    if not path.is_file():
        # MuJoCo reports a missing file as a ValueError from its XML parser; this one the command
        # tree reports as a missing input.
        raise FileNotFoundError(errno.ENOENT, "no such MJCF", str(path))
    model = mujoco.MjSpec.from_file(str(path)).compile()
    return model, mujoco.MjData(model), {}


def compile_from_world(
    world: str | Path,
    skip: set[str],
    overrides: dict,
    logger: logging.Logger,
    settle_steps: int = 0,
) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Compile a world YAML through the real build pipeline, so the export == the simulated scene.

    Transport plugins are dropped first -- a geometry export needs no bridge, and one may not even be
    loadable here (``roqsim.config.drop_transport_plugins``). ``skip`` then drops *further* plugins by
    ``name`` or plugin ref before the engine is built. Everything that contributes geometry
    (floorplan, spawn_robot, walker, conveyor, ...) stays. ``overrides`` (from ``--set``) is
    deep-merged into the world first -- e.g. to supply a floorplan ``mesh`` that a scenario would
    normally inject at run time.

    ``setup()`` alone never runs ``on_reset``, so a mocap-driven walker's bones stay at their park
    pose (z=-50) and free bodies keep only their configure-time seating. We ``reset()`` (which runs
    every plugin's ``on_reset`` -- re-posing the walker's mocap and re-seating robot bases) then a
    second ``mj_forward`` to propagate the re-posed mocap into ``data.xpos``/``xquat`` (reset()'s own
    forward runs *before* the walker re-poses). ``settle_steps`` optionally steps physics a little
    further (e.g. to let a dropped free body settle) before the state is captured.

    Returns ``(model, data, view)``, ``view`` being the world's authored free-camera framing.
    """
    cfg = load_config(world, overrides or None)
    transport, unavailable = drop_transport_plugins(cfg)
    if transport:
        logger.info("skipping transport plugins: %s", ", ".join(transport))
    if unavailable:
        logger.warning(
            "skipping plugin(s) this environment cannot load: %s. They build no geometry, so the "
            "export is unaffected -- but check the spelling if you expected one.",
            ", ".join(unavailable),
        )
    if skip:
        kept = [p for p in cfg.plugins if p.ref not in skip and (p.name or "") not in skip]
        dropped = [p.name or p.ref for p in cfg.plugins if p not in kept]
        if dropped:
            logger.info("skipping plugins: %s", ", ".join(dropped))
        cfg.plugins = kept
    # `preview`: settling a scene to look at it is not a measurement, so the seed is the fixed
    # one rather than the driver's to resolve.
    # The model and data outlive the plugins: an export reads only them.
    with Engine(cfg, preview=True) as engine:  # build + compile + configure
        engine.reset()  # on_reset: re-pose mocap walkers, re-seat robot bases
        mujoco.mj_forward(engine.ctx.model, engine.ctx.data)  # re-posed mocap into data.xpos
        for _ in range(max(0, settle_steps)):
            engine.step()
        return engine.ctx.model, engine.ctx.data, cfg.view


def skip_set(args: argparse.Namespace) -> set[str]:
    """The ``--skip-plugins`` names as a set."""
    return {s.strip() for s in args.skip_plugins.split(",") if s.strip()}


def compile_source(
    args: argparse.Namespace, logger: logging.Logger
) -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    """Compile the source the parsed :func:`add_source_options` name: ``(model, data, view)``."""
    if args.mjcf:
        return compile_from_mjcf(Path(args.mjcf))
    # Resolved by the same function as `roqsim sim`'s, because an export that resolved overrides
    # differently from the run would compile geometry the run never had.
    return compile_from_world(
        args.world,
        skip_set(args),
        overrides_from_options(args),
        logger,
        settle_steps=args.settle_steps,
    )


def write_manifest(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Write the ``--manifest`` source list, when one was asked for."""
    if not args.manifest:
        return
    sources = (
        [str(p) for p in world_sources(args.world)]
        if args.world
        else [str(Path(args.mjcf).resolve())]
    )
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as fh:
        json.dump({"inputs": sources}, fh, indent=2)
    logger.info("wrote source manifest (%d files) to %s", len(sources), args.manifest)
