"""scenario-execution driver: expose a roqsim world as a ``SimulationInterface``.

Usage::

    scenario_execution --simulation roqsim.scenario_adapter:MujocoSim <scenario.osc>

scenario-execution loads this class by ``module:Class``, instantiates it, then drives the lifecycle
(``setup`` → per-scenario ``reset``/``step`` → ``shutdown``). The framework owns the loop and pacing,
so ``step()`` here is a single, non-blocking ``engine.step()``.

The world YAML is chosen via the ``ROQSIM_WORLD`` environment variable (scenario-execution
instantiates the class with no constructor args). ``SimulationInterface`` is ROS-free, so importing
it does not pull ROS into the core; if scenario-execution is not installed, a local ABC fallback with
the identical interface is used (keeps the module importable for tests).

An interactive viewer window can be shown by passing the ``headless: "False"`` scenario parameter
(the adapter is headless by default); it opens roqsim's shared passive viewer and syncs it each
step, honouring the world's ``sim.view`` block -- including a ``track`` target that keeps the camera
on a mobile robot (see :class:`roqsim.viewer.TrackingCamera`). A window needs a display: if
``headless`` is False but no ``DISPLAY`` is set, ``reset()`` raises
:class:`~roqsim.viewer.DisplayError` with guidance rather than crashing in the native viewer init.

Parts of the world can be overridden per scenario via the ``world_overrides`` parameter -- a nested
dict mirroring the world YAML, with plugins addressed by name (see :func:`roqsim.apply_overrides`).
In OSC this is naturally a struct parameter, which scenario-execution passes as a nested dict::

    world_overrides: {plugins: {floorplan: {floor: {reflectance: 0.3}}}}

A scenario whose *experiment is the world* -- a study sweeping one problem instance per
configuration -- can also declare a ``world`` parameter and select the world itself, which is what
makes that one scenario instead of N. Ordinary scenarios declare neither and stay
simulator-agnostic: the world then comes from ``ROQSIM_WORLD``, set by whoever deployed the run.

Because the world is *compiled* when the engine is built (not on ``reset``), the engine is built
lazily -- on the first of ``reset()``/``dt``/``step()`` -- so it is constructed with the world and
overrides in place. A later ``reset()`` naming a different world, or different overrides, rebuilds.

The deployment may also supply overrides without the scenario mentioning them, through
``ROQSIM_WORLD_OVERRIDES`` (a path to the same YAML document ``roqsim sim --override``
reads) -- the stepped shape's equivalent of that flag, since scenario-execution constructs
this class with no arguments. A scenario passing ``world_overrides`` itself wins over it.

ROS transport is **not** a scenario parameter either. A checked-in world stays ROS-free so
``roqsim sim`` can run it in a pip-only environment, so the bridge is appended at load time when
``ROQSIM_ROS`` is set (with ``ROQSIM_TF_NAMESPACE`` / ``ROQSIM_SIM_CONTROL``) -- see
:func:`roqsim.config.with_transport`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import load_config, overrides_from_files
from .context import SimContext
from .engine import Engine
from .viewer import DisplayError, PassiveViewer, has_display

_UNBUILT = object()  # sentinel: no engine built yet (distinct from "built with no overrides")


def _as_headless(headless) -> bool:
    """Interpret a ``headless`` scenario parameter (string/bool/None) as a bool.

    ``None`` (unset) means headless -- the adapter's default. Strings like ``"False"``/``"0"``/``"no"``
    (case-insensitive) mean "show the window".
    """
    if headless is None:
        return True
    if isinstance(headless, str):
        return headless.strip().lower() not in ("false", "0", "no")
    return bool(headless)


def _as_bool(value) -> bool:
    """Interpret an environment flag: unset/empty/``false``/``0``/``no`` are all off."""
    if not value:
        return False
    return str(value).strip().lower() not in ("false", "0", "no")


class _EagerLogger:
    """Format ``%``-style log arguments here, for a logger that cannot.

    roqsim logs the stdlib way -- ``logger.info("seed: %s", seed)`` -- deferring the
    formatting to the handler. scenario-execution's ``Logger`` (and its ROS variant) take
    a single preformatted message, so handing one straight to :class:`~roqsim.engine.Engine`
    fails the moment any plugin logs with arguments: ``Logger.info() takes 2 positional
    arguments but 5 were given``, raised from inside ``reset()`` and reported only as
    "Simulation reset failed".

    Wrapping rather than substituting a stdlib logger keeps the engine's output where the
    run's log is, which is the whole reason the driver passes a logger in.
    """

    __slots__ = ("_target",)

    def __init__(self, target):
        self._target = target

    def _emit(self, level: str, msg, *args, **kwargs):
        method = getattr(self._target, level, None)
        if method is None:
            return
        text = str(msg) % args if args else msg
        try:
            method(text, **kwargs)
        except TypeError:
            # A logger that does not accept the keywords either (exc_info, stacklevel).
            method(text)

    def debug(self, msg, *args, **kwargs):
        self._emit("debug", msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._emit("info", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._emit("warning", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._emit("error", msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        # Not every foreign logger has `exception`; `error` is the honest fallback.
        level = "exception" if hasattr(self._target, "exception") else "error"
        self._emit(level, msg, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._target, name)


def _wrap_logger(logger):
    """Wrap a foreign logger; leave a stdlib one (which formats lazily) alone."""
    if logger is None or isinstance(logger, logging.Logger):
        return logger
    return _EagerLogger(logger)


def _resolve_world(world: str) -> str:
    """Accept the same world references ``roqsim sim`` does, not only a filesystem path.

    A packaged world is named ``<pkg>:<world>`` and resolved through the ``roqsim.worlds``
    entry-point group; a path is passed through untouched. Without this the adapter took
    a strictly narrower input than the CLI for the same thing, so a deployment that set
    ``ROQSIM_WORLD`` to a package ref -- the form that needs no files to travel with the
    run, and therefore the form an embedding driver naturally uses -- failed with
    ``world config <pkg>:<world> does not exist``.

    Resolution failures are left to :func:`load_config`, which reports the reference it
    was given rather than a path invented here.
    """
    if ":" not in world or world.startswith((".", "/")) or Path(world).exists():
        return world
    from .world import resolve_world_yaml_ref

    try:
        resolved = resolve_world_yaml_ref(world)
    except FileNotFoundError:
        return world
    return str(resolved) if resolved is not None else world


try:  # real base when scenario-execution is available
    from scenario_execution.simulation import SimulationInterface as _Base
except Exception:  # pragma: no cover - fallback keeps core importable/testable
    from abc import ABC, abstractmethod

    class _Base(ABC):  # minimal stand-in with the same contract
        @property
        @abstractmethod
        def dt(self) -> float: ...

        def setup(self, **kwargs) -> None: ...

        def reset(self, **kwargs) -> None: ...

        @abstractmethod
        def step(self) -> None: ...

        def shutdown(self) -> None: ...


class MujocoSim(_Base):
    """Adapts an :class:`Engine` to scenario-execution's step-based ``SimulationInterface``."""

    def __init__(self, world: str | None = None, world_overrides: dict | None = None):
        self._default_world = world or os.environ.get("ROQSIM_WORLD")
        if not self._default_world:
            raise ValueError(
                "no world specified: pass world=... or set ROQSIM_WORLD to a world YAML path"
            )
        self._world = self._default_world
        # Transport is a *deployment* fact, not a scenario one: a checked-in world stays
        # ROS-free so it runs standalone, and the scenario must not have to know this
        # simulator has a bridge. So it arrives the same way the world does -- from the
        # environment, set by whoever deployed the run -- rather than as an OSC parameter.
        self._transport = (
            {
                "ros": True,
                "tf_namespace": os.environ.get("ROQSIM_TF_NAMESPACE") or None,
                "control": _as_bool(os.environ.get("ROQSIM_SIM_CONTROL")),
            }
            if _as_bool(os.environ.get("ROQSIM_ROS"))
            else None
        )
        # A deployment's overrides, for the shape that has no command line. `roqsim sim` takes
        # them with --override; here the driver is scenario-execution, which instantiates
        # this class with no arguments, so the path arrives the same way the world does --
        # from the environment, set by whoever deployed the run. A scenario that passes
        # `world_overrides` itself still wins: that is the experiment speaking, and it is
        # more specific than the deployment.
        if world_overrides is None:
            overrides_file = os.environ.get("ROQSIM_WORLD_OVERRIDES")
            world_overrides = overrides_from_files([overrides_file]) if overrides_file else None
        self._default_overrides = world_overrides
        self._built_overrides = _UNBUILT
        self._logger = None
        self._engine: Engine | None = None
        self._viewer: PassiveViewer | None = None
        self._scene_export_pending = False
        self._output_dir: str | None = None
        self._recorder = None

    # -- build ------------------------------------------------------------------------------------

    def _build(self, world: str, overrides: dict | None) -> None:
        """(Re)build the engine for ``world`` with ``overrides`` merged in; no-op if unchanged.

        The world is part of the cache key, not just the overrides: a scenario whose
        *experiment is the world* sweeps it per configuration, and reusing the previous
        model for a new world would silently run the wrong one.
        """
        if self._engine is not None and world == self._world and overrides == self._built_overrides:
            return
        self._teardown_engine()
        cfg = load_config(_resolve_world(world), overrides, self._transport)
        self._world = world
        self._engine = Engine(cfg, logger=_wrap_logger(self._logger))
        self._engine.setup()
        self._built_overrides = overrides
        # A (re)built world means the browser scene descriptor (if requested) is stale; the export
        # itself runs after reset(), when mocap bodies are at their true initial pose.
        self._scene_export_pending = True

    def _teardown_engine(self) -> None:
        """Drop the viewer (its model/data would go stale) and the engine, if any.

        The single point where the live model dies -- on shutdown *and* on a mid-session rebuild -- so
        it is where a recording has to be flushed. Doing it in ``shutdown`` alone would lose a
        recording whenever a scenario reset with different ``world_overrides``.
        """
        self._finish_recording()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None

    def _ensure_built(self) -> Engine:
        if self._engine is None:
            self._build(self._default_world, self._default_overrides)
        return self._engine

    def _export_scene(self) -> None:
        """Write the browser scene descriptor of the just-built world, when requested.

        Opt-in via the ``ROQSIM_SCENE_EXPORT_DIR`` environment variable: when set, the compiled
        world is exported as ``scene.json``/``scene.bin`` (+ textures) into that directory --
        a relative path resolves against the scenario's ``output_dir`` (passed to ``setup()`` by
        the runner; under RoboVAST that is the run's result directory, so the exact simulated scene,
        world_overrides included, ships as a run artifact for browser viewers), falling back to the
        process cwd when the runner provides none. Called after ``engine.reset()`` so mocap-driven
        bodies (walkers) and re-seated robot bases are captured at their true initial pose; the
        forward pass propagates the re-posed mocap into ``data.xpos`` first (mirrors
        ``export_web._compile_from_world``).
        """
        self._scene_export_pending = False
        out = os.environ.get("ROQSIM_SCENE_EXPORT_DIR")
        if not out:
            return
        import mujoco

        from .export_web import export_scene

        out_dir = Path(out)
        if not out_dir.is_absolute() and self._output_dir:
            out_dir = Path(self._output_dir) / out_dir
        mujoco.mj_forward(self._engine.ctx.model, self._engine.ctx.data)
        # NOT self._logger: the runner may hand us a non-stdlib logger (e.g. scenario-execution's
        # RosLogger), whose info() lacks %-style lazy formatting that export_scene uses.
        export_scene(
            self._engine.ctx.model,
            self._engine.ctx.data,
            out_dir,
            logging.getLogger(__name__),
            view=self._engine.config.view,
        )

    def _resolve_out(self, value: str) -> Path:
        """Resolve a relative output path against THIS RUN's directory.

        ``RUN_OUTPUT_DIR`` first, because the scenario's own ``output_dir`` is the campaign root
        (scenario-execution is started with ``-o /out``) while a run's results are collected from
        ``/out/<config>/<run>``. Anchored at the root, every run of a sweep writes the same
        ``run.npz`` and the same ``capture/``, each overwriting the last -- which looks like a working
        capture until you notice every configuration replays identically, and leaves the run
        directories with no capture at all.

        A campaign runner that packs several runs into one job sets no ``RUN_OUTPUT_DIR`` (one
        variable cannot serve them all), and there the scenario's ``output_dir`` is still the best
        answer available -- so it stays the fallback rather than an error.
        """
        path = Path(value)
        if path.is_absolute():
            return path
        base = os.environ.get("RUN_OUTPUT_DIR") or self._output_dir
        return Path(base) / path if base else path

    def _start_recording(self) -> None:
        """Begin sampling MuJoCo state, when the run asked for it.

        Opt-in via ``ROQSIM_RECORD`` (the ``.npz`` path, relative to the scenario's ``output_dir``
        like the scene export), with ``ROQSIM_CAPTURE_FPS`` for the rate. Recording is a *session*
        concern rather than an experiment one -- the same footing as ``sim.headless``, which the world
        YAML rejects on purpose -- so it is driven by the environment here and never by the world.

        The recorder is rebuilt with the world: it holds the model whose state it packs, so a world
        rebuilt with different ``world_overrides`` needs a new one.
        """
        target = os.environ.get("ROQSIM_RECORD")
        if not target:
            return
        from .capture import DEFAULT_FPS, CaptureError, StateRecorder, env_flag, parse_fps, snap_fps

        lg = logging.getLogger(__name__)
        try:
            rate = snap_fps(
                parse_fps(os.environ.get("ROQSIM_CAPTURE_FPS") or DEFAULT_FPS),
                self._engine.dt,
            )
        except CaptureError as err:
            # A rate that cannot exist in this world is a configuration error, and a run that silently
            # recorded at some other rate would be worse than one that refuses.
            raise ValueError(f"ROQSIM_CAPTURE_FPS: {err}") from err
        rate.report(lg)
        self._recorder = StateRecorder(
            self._engine.ctx,
            self._resolve_out(target),
            rate,
            world=self._world,
            overrides=self._built_overrides if isinstance(self._built_overrides, dict) else None,
            sim_poses=env_flag("ROQSIM_SIM_POSES"),
            logger=lg,
        )

    def _finish_recording(self) -> None:
        """Write the recording and, when asked, the browser run capture derived from it.

        Called before the engine is torn down, because the capture is derived against the **live**
        model: no world rebuild, and no GL backend, since nothing is rendered. Both artifacts land only
        on a clean stop -- an ``.npz`` writes its zip index at close, so a hard kill (SIGKILL, a
        campaign's per-run timeout) leaves neither.

        The capture is reported-never-fatal: a run whose results are otherwise good must not fail
        because a viewer artifact could not be written.
        """
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return
        lg = logging.getLogger(__name__)
        out = os.environ.get("ROQSIM_CAPTURE_EXPORT_DIR")
        if out and self._engine is not None and recorder.frames:
            from .export_capture import write_capture

            try:
                write_capture(
                    self._engine.ctx.model,
                    recorder.replay(self._engine.ctx),
                    self._resolve_out(out),
                    world=self._world,
                    overrides=self._built_overrides
                    if isinstance(self._built_overrides, dict)
                    else {},
                    seed=getattr(self._engine.ctx, "seed", None),
                    logger=lg,
                )
            except Exception as err:  # pylint: disable=broad-except
                lg.warning(
                    "run capture export failed (%s); the recording itself is unaffected", err
                )
        recorder.close()

    # -- SimulationInterface ----------------------------------------------------------------------

    @property
    def dt(self) -> float:
        return self._ensure_built().dt

    @property
    def context(self) -> SimContext | None:
        """The running world's :class:`~roqsim.context.SimContext`, or ``None`` before it is built.

        The seam an IN-PROCESS driver reads -- an ``.osc`` action measuring an entity or firing a
        fault, a test asserting on a plugin. It exists because those callers live in other packages
        and were reaching ``_engine``, a private name in another repo that no deprecation could
        protect.

        ``SimContext`` and not ``Engine``, deliberately: the context is already the substrate's
        contract for in-process code -- it is exactly what a plugin is handed -- so publishing it
        grants a caller no rights a plugin does not have, and the single-writer rule
        (:meth:`~roqsim.context.SimContext.post` for anything that mutates ``model``/``data``) applies
        to it verbatim. Publishing the engine instead would hand out ``plugins`` and ``config`` as
        well, and a consumer reaching into the plugin list is the thing this replaces: a plugin an
        external driver must reach publishes a blackboard handle (``metrics:<name>``,
        ``model_override:<name>``, ``task:<name>``), so nothing has to match on class names.

        Deliberately NOT ``_ensure_built()``: a behaviour tree asking whether a world exists must
        not compile one as a side effect, and the tree is set up before the first ``reset()``. A
        caller handles ``None`` by waiting a tick.
        """
        return self._engine.ctx if self._engine is not None else None

    def setup(self, **kwargs) -> None:
        # Only record context: the world is built lazily so that reset()'s ``world_overrides`` are
        # applied when it is compiled (see module docstring). ``output_dir`` (the scenario's result
        # directory, when the runner provides one) anchors a relative scene-export path.
        self._logger = kwargs.get("logger")
        self._output_dir = kwargs.get("output_dir")

    def reset(self, world=None, world_overrides=None, headless=None, **kwargs) -> None:
        self._build(
            world or self._default_world,
            world_overrides if world_overrides is not None else self._default_overrides,
        )
        # scenario-execution injects matching OSC params by name; forward the rest to the engine.
        self._engine.reset(**kwargs)
        if self._scene_export_pending:
            self._export_scene()
        # After reset, so the first sample is the world as the scenario starts rather than as it
        # compiled; a re-reset of the same world restarts the schedule instead of continuing it.
        if self._recorder is None:
            self._start_recording()
        else:
            self._recorder.on_reset()
        # Open the interactive viewer once, when the scenario asked for it (default: headless).
        # A window needs a display; if none is set, fail with a clear message rather than crashing in
        # the native GLFW init. (The offscreen render backend falls back to CPU on its own.)
        if self._viewer is None and not _as_headless(headless):
            if not has_display():
                raise DisplayError(
                    "interactive viewer requested (headless=False) but no DISPLAY is set: run with a "
                    "display available (locally), or run headless (headless=True)."
                )
            self._viewer = PassiveViewer(
                self._engine.ctx, self._engine.config.view, name=self._engine.config.name
            )

    def step(self) -> None:
        self._ensure_built().step()
        if self._recorder is not None:
            self._recorder.sample(self._engine.ctx)
        if self._viewer is not None:
            if self._viewer.is_running():
                self._viewer.sync()
            else:  # window closed by the user -> stop syncing
                self._viewer.close()
                self._viewer = None

    def shutdown(self) -> None:
        self._teardown_engine()  # flushes the recording first; see there
        self._built_overrides = _UNBUILT
