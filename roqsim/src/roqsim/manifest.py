"""Robot-description manifests: the plugins a spawned model brings with it.

A robot/arm ships a ``<model>.manifest.yaml`` next to its MJCF listing the controller/sensor plugins
that are intrinsic to it (a diff-drive + lidar for a mobile base, an arm_controller for a
manipulator). A spawn plugin implements :meth:`roqsim.plugin.Plugin.expand` by delegating to
:func:`expand_manifest`, so a world that just spawns the model gets those plugins without
re-declaring them -- and any spawn plugin (mobile, arm, ...) reuses the same mechanism, differing
only in the config key its injected plugins use to name their entity (``robot`` vs ``arm``).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import PluginError, PluginSpec, document_entries, parse_plugin_entry
from .frames import substitute
from .models import resolve_model
from .registry import resolve_plugin


def manifest_path(model_file: Path) -> Path:
    """``<model-stem>.manifest.yaml`` beside a resolved model file."""
    return model_file.parent / f"{model_file.stem}.manifest.yaml"


def manifest_fov(model_file: Path) -> dict:
    """The ``fov:`` block from a model's manifest (``{}`` when it has no manifest or no block).

    Where a sensor model states its own valid detection range, so neither a world nor an analysis has
    to repeat it: ``near``/``far`` for a camera frustum, plus the angular bounds a camera-less device
    needs (``mid360`` adds ``h_min``/``h_max``/``v_min``/``v_max``). Two callers read it and must
    agree -- ``spawn_sensor`` draws the ``show_fov`` cone from it, and the coverage catalog derives a
    hypothetical placement's range from it -- which is why it is here rather than private to either.

    Deliberately unvalidated: the keys are the reader's business (a frustum and an angular sector want
    different ones), and a model with no ``fov:`` is normal rather than broken.
    """
    path = manifest_path(model_file)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("fov", {}) or {}


def manifest_frames(model_file: Path) -> list:
    """The ``frames:`` block from a model's manifest, as written (``[]`` when there is none).

    The fixed links a vendor description chains and the MJCF flattened (see :mod:`roqsim.frames`
    for the shape and how they are built and published). Returned raw because a device's frame
    names are templates its mount fills in, and validated by :func:`roqsim.frames.parse_frames`
    once they are. Not inherited through ``extends:``: frames describe a model's geometry, and a
    derived model keeps its own MJCF.
    """
    path = manifest_path(model_file)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    frames = data.get("frames")
    if frames is not None and not isinstance(frames, list):
        raise PluginError(f"manifest {path}: 'frames' must be a list of {{name, parent, pos, rpy}}")
    return list(frames or [])


def manifest_frame_id(model_file: Path) -> str | None:
    """The ``frame_id:`` a device model's manifest declares: the vendor's default scan-frame name.

    A mount that sets no ``frame_id`` of its own takes this one, and it is what a device manifest's
    ``{frame_id}`` placeholders are filled with then. ``None`` when there is no manifest or the
    vendor names no default -- a device whose frame name is always its integrator's choice -- so
    such a mount must name the frame itself.

    The one placeholder it may carry is ``{device_name}`` (:func:`manifest_device_name`): a vendor
    macro that prefixes every link with its ``name`` parameter names its optical frame
    ``<name>_..._optical_frame``, and the mount fills it in. Any other placeholder raises, since it
    would reach a TF tree as a literal brace.
    """
    path = manifest_path(model_file)
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    value = data.get("frame_id")
    if value is None:
        return None
    bare = value.replace("{device_name}", "") if isinstance(value, str) else value
    if not isinstance(value, str) or not value or "{" in bare or "}" in bare:
        raise PluginError(
            f"manifest {path}: 'frame_id' is the vendor's default scan-frame name, a non-empty "
            f"string whose only placeholder may be '{{device_name}}' -- got {value!r}"
        )
    return value


def manifest_device_name(model_file: Path) -> str | None:
    """The ``device_name:`` a device model's manifest declares: the vendor macro's default ``name``.

    A vendor description that instantiates a device through a macro prefixes each link it creates
    with the macro's ``name`` parameter (``camera_link``, ``camera_color_optical_frame``). The
    manifest spells those frames ``{device_name}_link``, and a mount that sets no ``device_name`` of
    its own takes this default. ``None`` when there is no manifest or it declares none. A value that
    is not a plain name raises.
    """
    path = manifest_path(model_file)
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text()) or {}
    value = data.get("device_name")
    if value is None:
        return None
    if not isinstance(value, str) or not value or "{" in value or "}" in value:
        raise PluginError(
            f"manifest {path}: 'device_name' is the vendor macro's default 'name', a non-empty "
            f"string with no placeholder -- got {value!r}"
        )
    return value


def manifest_license(model_file: Path) -> list[Path]:
    """The licence sidecars a model's manifest declares (``license:``), as existing paths.

    Which licence covers a model is only self-evident in the folder-per-model layout, where the
    sidecar sits alone beside the MJCF. A provider that ships its models flat has one directory
    holding every model and every vendored licence, and "the file next to it" then attributes one
    vendor's terms to another vendor's robot -- a wrong answer of the worst kind. So a model whose
    licence is not obvious from the layout names it::

        license: LICENSE.unitree_ros        # one, or a list when several apply

    A name is relative to the model's own directory and must exist -- a declaration pointing at
    nothing raises here rather than quietly dropping the model's licence from the catalog. A model
    that declares none is normal (a hand-authored one carries no vendor terms), and its readers fall
    back to what ships beside it.
    """
    path = manifest_path(model_file)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    declared = data.get("license")
    names = [declared] if isinstance(declared, str) else list(declared or [])
    paths = []
    for name in names:
        sidecar = model_file.parent / name
        if not sidecar.is_file():
            raise PluginError(f"{path}: license: {name!r} is not a file beside the model")
        paths.append(sidecar)
    return paths


def load_manifest(
    model_file: Path, base_dir: Path | None = None, seen: frozenset[Path] = frozenset()
) -> list[dict]:
    """The manifest's ``components:`` list for a resolved model file, or ``[]`` when it has none.

    A manifest is a document like any other, so it may ``extends:`` another model's manifest and
    inherit its components: ``unitree_g1_dex1`` is ``unitree_g1`` plus hands, and says exactly that
    instead of repeating the base's ``g1_locomotion`` and ``lidar`` blocks for someone to keep in
    step by hand.
    It inherits **components, not geometry** -- a derived model keeps its own MJCF; ``extends:`` never
    carried geometry, ``sim.world`` did, and a manifest may not carry ``sim:`` at all (see below).

    The base is named the way anything else names a model (a ``roqsim.models`` ref, or a path
    relative to this manifest), and cycles raise rather than recursing forever.
    """
    path = manifest_path(model_file)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise PluginError(f"manifest {path} must be a mapping at the top level")
    if "sim" in data:
        # A model is a component of a world, and `sim:` belongs to the run and the scene. Letting a
        # manifest set `seed`, `pacing` or `contact_override` would let a robot reach up and change
        # the experiment it is part of -- action at a distance, from a file the world never opened.
        raise PluginError(
            f"manifest {path} has a 'sim:' block. `sim:` belongs to the world being run, not to a "
            f"model included in it: a manifest cannot set the run's seed, pacing or contact "
            f"overrides. Move those keys to the world that spawns this model."
        )
    inherited: list[dict] = []
    ext = data.get("extends")
    if ext is not None:
        if path in seen:
            chain = " -> ".join(str(p) for p in (*seen, path))
            raise PluginError(f"manifest 'extends' cycle detected: {chain}")
        base_model = resolve_model(str(ext), base_dir=base_dir or path.parent).path
        inherited = load_manifest(base_model, base_dir=base_dir, seen=seen | {path})
    return inherited + document_entries(data, str(path))


def expand_manifest(
    spec: PluginSpec,
    world: list[PluginSpec],
    *,
    base_dir: Path | None = None,
    substitutions: dict[str, str] | None = None,
) -> list[PluginSpec]:
    """Plugin specs a spawn plugin should inject for its model, wired to its entity.

    ``substitutions`` fills ``{placeholder}`` fields in every string of the manifest's component
    configs (nested ones included) before anything is merged, and refuses a placeholder it does not
    name (:func:`roqsim.frames.substitute`). A mount uses it to hand its device the names only the
    placement knows -- ``frame_id: "{frame_id}"``. ``None`` leaves strings as written.

    The model is resolved via :func:`roqsim.models.resolve_model` (so it may live in any installed
    package), and its ``<model>.manifest.yaml`` manifest is read from beside the resolved file. Each
    (a mobile spawn wires ``robot: <name>``, an arm spawn ``arm: <name>``) and inherits the spawn's
    ``prefix`` -- so a build-time plugin (e.g. ``fiducial_marker`` welding a geom onto a spawned body)
    can form prefixed names like ``<prefix>wrist_3_link`` before the entity registry exists; runtime
    plugins keep reading the prefix from the entity. Distinct entities (two arms) never collide.

    When the owner already declares a component with the same **label**, the manifest default is not
    injected -- the world's entry is the one that runs -- but the manifest's config is **merged
    underneath it** (per key: the world's value always wins, missing keys are filled from the
    manifest). That is what lets a world override *part* of a default: ``diff_drive:
    {test_cmd: [0.5, 0.4]}`` adds a scripted command while keeping the model's wheel geometry, slip
    calibration and actuator names.

    The merge is shallow, deliberately: a nested value the world sets (``topics: {scan: /s}``)
    replaces the manifest's whole mapping rather than being deep-merged into it, so what a world says
    is what a reader gets, without a per-key excavation of two files. The world's ``PluginSpec`` is
    mutated in place; that is safe because every plugin is constructed only after expansion finishes
    (see :func:`roqsim.config.instantiate_plugins`), so it does not matter whether the spawn is
    declared before or after the plugin it fills in.

    **A manifest may mount what has a manifest of its own.** An entry whose plugin itself provides an
    entity (a ``spawn_sensor`` on a robot) gets the spawn's prefix as ``attach_prefix`` rather than
    ``prefix`` -- it derives its own MJCF prefix from its carrier -- and its nested ``components:``
    are kept: emitted right after it, owned by its address (``robot.scan_front.lidar``), and merged
    by label exactly like the entries above. A robot manifest therefore overrides part of a mounted
    device's defaults by nesting, and precedence is nearer-wins all the way down: the world's value,
    then the robot manifest's, then the device manifest's (:func:`roqsim.config.expand_document`
    expands the device after this, against what is already declared). Nested children get no
    ``prefix``: their owner is the device, which fills its own in when it expands.

    Off with ``default_plugins: false`` on the spawn config; a no-op when there is no ``model``.
    """
    cfg = spec.config
    if not cfg.get("default_plugins", True) or not cfg.get("model"):
        return []
    model_file = resolve_model(cfg["model"], base_dir=base_dir).path
    # Keyed on the ADDRESS, whose last segment is the LABEL, not the plugin ref: a model may ship two
    # of a kind (tiago_pro's front and rear lidars), and keying on the ref would collapse them onto
    # one entry -- silently losing a sensor. A label is unique among an owner's components by
    # construction, and an address is unique in a document, so this cannot.
    declared: dict[str, PluginSpec] = {}
    for s in world:
        declared.setdefault(s.address, s)
    out: list[PluginSpec] = []
    prefix = cfg.get("prefix", "")
    where = str(manifest_path(model_file))
    for entry in load_manifest(model_file, base_dir=base_dir):
        base = parse_plugin_entry(entry, "manifest plugin")
        if substitutions is not None:
            _substitute_tree(base, substitutions, where)
        if _provides_entity(base.ref, base_dir):
            base.config.setdefault("attach_prefix", prefix)
        else:
            base.config.setdefault("prefix", prefix)
        _merge_or_inject(base, spec.address, declared, out, where, base_dir)
    return out


def _substitute_tree(spec: PluginSpec, values: dict[str, str], where: str) -> None:
    spec.config = substitute(spec.config, values, f"{where}: {spec.label}")
    for child in spec.children:
        _substitute_tree(child, values, where)


def _provides_entity(ref: str, base_dir: Path | None) -> bool:
    """Whether *ref*'s plugin registers an entity; ``False`` for a ref that does not resolve.

    An unresolvable ref is recorded and refused later, by :func:`roqsim.config.instantiate_plugins`,
    the same way one declared in a world is -- so a scene-only consumer still loads the model.
    """
    try:
        return resolve_plugin(ref, base_dir=base_dir).provides_entity
    except PluginError:
        return False


def _merge_or_inject(
    base: PluginSpec,
    owner: str,
    declared: dict[str, PluginSpec],
    out: list[PluginSpec],
    where: str,
    base_dir: Path | None,
) -> None:
    """Inject *base* under *owner*, or merge it under the entry already declared there; then its children."""
    address = f"{owner}.{base.label}"
    if base.children and not _provides_entity(base.ref, base_dir):
        try:
            resolve_plugin(base.ref, base_dir=base_dir)
        except PluginError:
            pass  # unresolvable: refused at instantiation, with the rest of the document's
        else:
            raise PluginError(
                f"{where}: '{address}' ({base.ref}) has a 'components:' block but registers no "
                f"entity, so there is nothing for those {len(base.children)} entries to attach to."
            )
    target = declared.get(address)
    if target is not None:
        # The world runs its own entry; give it the manifest's defaults for everything it did
        # not say. Without this, a partial override silently drops the rest of the model's
        # description -- e.g. a husky's diff_drive falling back to the plugin's TurtleBot
        # wheel_radius/actuator names, which then fails to resolve against the husky's MJCF.
        for key, value in base.config.items():
            target.config.setdefault(key, value)
    else:
        target = PluginSpec(
            ref=base.ref, name=base.name, config=base.config, entity=owner, enabled=base.enabled
        )
        out.append(target)
        declared[address] = target
    for child in base.children:
        _merge_or_inject(child, address, declared, out, where, base_dir)
