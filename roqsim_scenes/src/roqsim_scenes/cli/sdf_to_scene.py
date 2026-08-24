"""Convert a Gazebo/Ignition SDF world into a roqsim static scene (scene.json + OBJs).

Stage 1 of the import pipeline, sibling to ``usd_to_scene.py`` and feeding the same stage 2::

    world.sdf --[ sdf_to_scene.py ]--> scene.json + meshes/*.obj --[ scene_to_mjcf.py ]--> <name>.xml

Run::

    roqsim scenes sdf-to-scene --world worlds/warehouse.sdf \
        --out-dir <experiment>/simulation/scenes/scene1 --scene-name scene1 \
        --lock <experiment>/simulation/scenes/scene1/assets.lock.json
    roqsim scenes scene-to-mjcf --scene <experiment>/simulation/scenes/scene1

What makes SDF worlds different from USD: **they usually contain no geometry**. A typical ROS-era
world is a bill of materials of ``<include><uri>`` entries pointing into a model registry, with only
``<pose>`` pinned locally. So this tool spends most of its effort resolving and pinning assets
(``fuel_fetch``), then composing the SDF pose tree (world -> include -> model -> link -> visual) into
the flat world-space that ``scene.json`` expects.

Deliberately mechanical. Everything here is determinate: parse, resolve, compose, tessellate, write.
The judgement calls -- which world maps to which paper scene, whether a missing asset is link rot or
never-published, what an absent ``<actor>`` means for a paper's claimed dynamic obstacle -- belong to
the `scene-porting` skill and the spec, not to this script. Accordingly it **fails loudly** rather
than guessing: an unresolvable asset is an error with a message, never a silent omission or a
substitution.

Collision policy: honour the SDF's own visual/collision split (see ``_link``). Each emitted object
carries ``collide`` (does it take part in physics) and ``render`` (is it drawn) independently, because
a model's collision geometry is usually a crude stand-in for its visual -- collidable but not fit to
look at. ``--collision-only`` drops the visuals entirely: fastest, ugliest.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from lxml import etree

from roqsim_scenes import scene_mesh_io as mio

from . import fuel_fetch

_DEFAULT_RGBA = [0.7, 0.7, 0.72, 1.0]


def _welded(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Face indices over coincident-vertex-welded ids, so split triangles register as adjacent."""
    keys = np.round(verts / 1e-5).astype(np.int64)
    _, wid = np.unique(keys, axis=0, return_inverse=True)
    return wid[faces]


def _is_closed(verts: np.ndarray, faces: np.ndarray) -> bool:
    """True when every edge is shared by exactly two faces (a watertight surface)."""
    wf = _welded(verts, faces)
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in wf:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            counts[key] = counts.get(key, 0) + 1
    return bool(counts) and all(n == 2 for n in counts.values())


def _enclosed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume enclosed by a closed surface (divergence theorem over its triangles)."""
    tri = verts[faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def _ln(el) -> str:
    # Comments/PIs have a callable .tag and are not QNames; real worlds are full of commented-out models.
    return etree.QName(el).localname if isinstance(el.tag, str) else ""


def _kids(el, name: str) -> list:
    return [c for c in el if _ln(c) == name]


def _kid(el, name: str):
    k = _kids(el, name)
    return k[0] if k else None


def _text(el, name: str, default: str | None = None) -> str | None:
    k = _kid(el, name)
    return k.text.strip() if k is not None and k.text else default


def _pose_of(el) -> np.ndarray:
    return mio.pose_to_matrix(_text(el, "pose"))


def _slug(name: str, used: set[str]) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "obj"
    out, i = s, 1
    while out in used:
        i += 1
        out = f"{s}_{i}"
    used.add(out)
    return out


def _rgba_of(visual) -> list[float]:
    mat = _kid(visual, "material")
    if mat is None:
        return list(_DEFAULT_RGBA)
    for key in ("diffuse", "ambient"):
        t = _text(mat, key)
        if t:
            v = [float(x) for x in t.split()]
            return (v + [1.0] * 4)[:4]
    return list(_DEFAULT_RGBA)


class Importer:
    def __init__(self, args):
        self.args = args
        self.cache = Path(args.cache)
        self.model_paths = fuel_fetch.default_model_paths(args.model_path)
        self.assets: dict[str, fuel_fetch.Asset] = {}
        self.objects: list[dict] = []
        #: ``(verts, faces)`` of every part that will COLLIDE, for the scene-level passage check.
        self.collidable: list[tuple] = []
        self.used: set[str] = set()
        self.lo = np.full(3, np.inf)
        self.hi = np.full(3, -np.inf)
        self.skipped: list[str] = []
        self.textures: set[str] = set()
        self.ground_z: float | None = None

    # ---------------- asset resolution

    def _model_dir(self, uri: str) -> Path:
        if uri.startswith("model://"):
            return fuel_fetch.resolve_model_uri(uri, self.model_paths)
        asset = self.assets.get(uri)
        if asset is None:
            asset = fuel_fetch.fetch(uri, cache=self.cache)
            self.assets[uri] = asset
            print(f"  fetched {asset.owner}/{asset.name} v{asset.version} [{asset.licence}]")
        return self.cache / asset.local

    def _mesh_path(self, uri: str, model_dir: Path) -> Path:
        """A mesh <uri> inside a model: model://<name>/path, a bare relative path, or file://."""
        if uri.startswith("model://"):
            rest = uri[len("model://") :]
            _, _, rel = rest.partition("/")
            # Prefer the owning model's own dir (Fuel zips are self-contained) before the search path
            local = model_dir / rel
            if local.exists():
                return local
            return fuel_fetch.resolve_model_uri(uri, self.model_paths)
        if uri.startswith("file://"):
            return Path(uri[len("file://") :])
        p = model_dir / uri
        if p.exists():
            return p
        raise fuel_fetch.FuelError(f"mesh {uri} not found under {model_dir}")

    # ---------------- geometry

    def _geometry(self, geom, world: np.ndarray) -> list[mio.Submesh]:
        """One <geometry> -> world-space submeshes (one per material), or [] for shapes we drop."""
        if (m := _kid(geom, "mesh")) is not None:
            uri = _text(m, "uri")
            if not uri:
                raise fuel_fetch.FuelError("<mesh> without <uri>")
            path = self._mesh_path(uri, self._current_model_dir)
            # <submesh> narrows a shared mesh file to one named piece (Warehouse re-uses warehouse.dae
            # for its drop zone). Ignoring it silently duplicates the whole file at the visual's pose.
            sm = _kid(m, "submesh")
            subs = mio.read_mesh(
                path,
                submesh=_text(sm, "name") if sm is not None else None,
                center=(_text(sm, "center", "false").strip().lower() in ("1", "true"))
                if sm is not None
                else False,
            )
            sc = _text(m, "scale")
            scale = np.array([float(x) for x in sc.split()]) if sc else None
            for s in subs:
                s.verts = mio.transform(s.verts, world, scale)
            return subs
        if (b := _kid(geom, "box")) is not None:
            v, f = mio.box(_text(b, "size", "1 1 1"))
            return [mio.Submesh(verts=mio.transform(v, world), faces=f)]
        if (c := _kid(geom, "cylinder")) is not None:
            v, f = mio.cylinder(float(_text(c, "radius", "0.5")), float(_text(c, "length", "1")))
            return [mio.Submesh(verts=mio.transform(v, world), faces=f)]
        if (s := _kid(geom, "sphere")) is not None:
            v, f = mio.sphere(float(_text(s, "radius", "0.5")))
            return [mio.Submesh(verts=mio.transform(v, world), faces=f)]
        if _kid(geom, "plane") is not None:
            # An infinite ground plane is not scene geometry: scene_to_mjcf.py adds its own
            # (config `ground_plane`), and baking a finite stand-in would fake the extent. Its HEIGHT
            # is data, though, and the only statement of where the floor is: warehouse.sdf puts the
            # plane at z=0 and drops the building to z=-0.1, so its outdoor apron -- not its walkable
            # floor -- is the scene's lowest geometry. Guessing the ground from the scene bounds picks
            # that apron and sinks every robot 10 cm into the floor it drives on.
            z = float(world[2, 3])
            if self.ground_z is not None and abs(self.ground_z - z) > 1e-6:
                raise fuel_fetch.FuelError(
                    f"world declares ground planes at conflicting heights ({self.ground_z} and {z}); "
                    "pick one explicitly with scene.yaml `ground_z` rather than letting the import guess"
                )
            self.ground_z = z
            self.skipped.append(f"plane at z={z:g} (ground handled by scene_to_mjcf ground_plane)")
            return []
        if _kid(geom, "heightmap") is not None:
            raise fuel_fetch.FuelError(
                "<heightmap> is not supported yet: it needs its image + size/pos mapped to a MuJoCo "
                "hfield. Report it as an escape hatch rather than approximating the terrain."
            )
        self.skipped.append(f"unsupported geometry: {[_ln(c) for c in geom]}")
        return []

    def _link(self, link, parent: np.ndarray, model_name: str) -> None:
        """Emit a link's geometry, honouring the SDF's own visual/collision split.

        SDF states visual and collision separately, and model authors mean it: shelf_big's visual is a
        210-part mesh, its collision is a single 2.1x18x6 box. Colliding the visual instead would both
        hull a concave shell into a solid block AND cost 210 geoms for one shelf. So:

        * <visual>    -> rendered, `collide: false`  (MuJoCo contype/conaffinity 0)
        * <collision> -> collided, `collide: true`, and NOT rendered when a distinct <visual> exists:
          it is a stand-in shape, so drawing it would hide the real mesh inside a box.
        * a link with visual but NO collision -> the visual must collide; split it into components so
          its hull is sane. It stays rendered -- it is the only geometry the link has.
        """
        world_link = parent @ _pose_of(link)
        visuals = _kids(link, "visual")
        collisions = _kids(link, "collision")

        if self.args.collision_only:
            self._emit(
                collisions or visuals, world_link, model_name, collide=True, split=True, render=True
            )
            return

        # --no-collide: import this model as scenery only. For an axis-aligned building shell the
        # hull of a closed ring is unusable (see _check_hulls_are_faithful) and a handful of `box`
        # plugins in the world YAML is both correct and cheaper than any decomposition. The geometry
        # is still DRAWN and still generates the occupancy grid -- only physics is handed over.
        if any(pat in model_name for pat in self.args.no_collide):
            self._emit(
                visuals or collisions,
                world_link,
                model_name,
                collide=False,
                split=False,
                render=True,
                suffix="_vis",
            )
            print(
                f"  --no-collide: {model_name} imported as visual only (declare its collision "
                f"as primitives in the world YAML)"
            )
            return

        # Collidable geometry: the SDF's own <collision> where the author provided it, else the visual.
        self._emit(
            collisions or visuals,
            world_link,
            model_name,
            collide=True,
            split=True,
            render=not (collisions and visuals),
        )
        # Rendered geometry: only when it is distinct from what we just collided.
        if collisions and visuals:
            self._emit(
                visuals,
                world_link,
                model_name,
                collide=False,
                split=False,
                suffix="_vis",
                render=True,
            )

    def _emit(
        self,
        elems,
        world_link: np.ndarray,
        model_name: str,
        *,
        collide: bool,
        split: bool,
        render: bool,
        suffix: str = "",
    ) -> None:
        for el in elems:
            geom = _kid(el, "geometry")
            if geom is None:
                continue
            subs = self._geometry(geom, world_link @ _pose_of(el))
            sdf_rgba = _rgba_of(el)
            multi = len(subs) > 1
            for si, sub in enumerate(subs):
                # A mesh's own material beats the SDF <visual><material>: the SDF states one colour for
                # a link whose mesh may bind eight, and it is the mesh the author textured.
                rgba = sub.rgba if (sub.rgba and sub.texture is None) else sdf_rgba
                stem = f"{model_name}_{el.get('name') or _ln(el)}{suffix}"
                base = f"{stem}_m{si:02d}" if multi else stem

                # One MuJoCo geom per CONVEX PIECE for anything COLLIDABLE. MuJoCo collides a mesh by
                # its convex hull, so a whole building as one geom hulls into a solid block that fills
                # the interior and swallows the robot. usd_to_scene.py gets the split for free (USD
                # prims); SDF gives one <visual> for an entire warehouse, so recover it here.
                # Non-colliding visuals need no split — splitting them only multiplies geoms.
                #
                # Two cuts, and BOTH are needed: connected components separate objects that merely
                # share a mesh file, then a reflex-edge split separates the walls *within* one welded
                # shell. Components alone leave the warehouse a single 18,750 m^3 brick.
                do_split = split and collide and self.args.split_components
                if do_split:
                    parts = [
                        p
                        for pv, pf, puv in mio.split_components(sub.verts, sub.faces, uv=sub.uv)
                        for p in mio.split_convex_parts(pv, pf, uv=puv)
                    ]
                    # A cut can shed slivers: a part of <4 vertices is a lone triangle or an edge, and
                    # MuJoCo rejects it outright ("at least 4 vertices required") because a mesh geom
                    # is collided by its convex HULL, which needs a tetrahedron to exist. Such a part
                    # encloses no volume, so dropping it removes nothing collidable — but say so, since
                    # a silently thinner collision shell is exactly the failure that looks like success.
                    # Losing them ALL means the cut destroyed the geometry rather than trimmed it.
                    solid = [p for p in parts if len(p[0]) >= 4]
                    if len(solid) != len(parts):
                        print(
                            f"  dropped {len(parts) - len(solid)} degenerate part(s) (<4 verts) of {base}"
                        )
                    if not solid:
                        raise SystemExit(
                            f"{base}: every one of the {len(parts)} split parts is degenerate (<4 "
                            f"vertices) — the convex split destroyed this collision mesh instead of "
                            f"cutting it. Re-run with --no-split-components to keep it whole."
                        )
                    parts = solid
                    self._check_hulls_are_faithful(parts, base, model_name)
                else:
                    parts = [(sub.verts, sub.faces, sub.uv)]
                if collide:
                    # Kept for the scene-level passage check in run(): a way through can be closed
                    # JOINTLY by parts that individually close nothing, so no per-part test can see it.
                    self.collidable += [(pv, pf) for pv, pf, _ in parts]
                for i, (pv, pf, puv) in enumerate(parts):
                    name = _slug(base if len(parts) == 1 else f"{base}_{i:03d}", self.used)
                    rel = Path("meshes") / f"{name}.obj"
                    # A hidden collision hull is never sampled, so its UVs are dead weight in the OBJ.
                    mio.write_obj(Path(self.args.out_dir) / rel, pv, pf, uv=puv if render else None)
                    self.lo = np.minimum(self.lo, pv.min(axis=0))
                    self.hi = np.maximum(self.hi, pv.max(axis=0))
                    obj = {
                        "name": name,
                        "mesh": str(rel),
                        "rgba": rgba,
                        "collide": collide,
                        "render": render,
                    }
                    if render and sub.texture is not None and puv is not None:
                        obj["texture"] = str(self._pin_texture(sub.texture))
                    self.objects.append(obj)
                if len(parts) > 1:
                    print(f"  split {base} -> {len(parts)} components")

    def _check_nothing_is_walled_off(self) -> None:
        """Refuse a scene whose collision hulls seal off floor its own geometry leaves reachable.

        The scene-level counterpart to :meth:`_check_hulls_are_faithful`, and the one that catches the
        failure that actually ships. That one asks whether a *part* is faithful, judges only watertight
        parts, and scores on volume -- so a wall skin with a door cut out of it slips through twice
        over: it is an open surface, and a 0.8 x 2.0 m doorway is a rounding error in a 13 m wall's
        volume while being 100 % of its passability.

        This one asks the only question that matters to a robot: **is there still a way through?** See
        :mod:`roqsim_scenes.passability` for why it is connectivity rather than any per-part measure.
        """
        from roqsim_scenes import passability

        sealed = passability.closed_passages(self.collidable)
        if not sealed:
            return
        where = "\n".join(f"  - {p}" for p in sealed[:8])
        more = f"\n  ... and {len(sealed) - 8} more" if len(sealed) > 8 else ""
        raise SystemExit(
            f"{self.args.scene_name}: the collision hulls wall off floor this world's own geometry "
            f"leaves open, in {len(sealed)} place(s):\n{where}{more}\n"
            f"MuJoCo collides a mesh by its convex HULL, so a wall with a door cut out of it is a "
            f"solid wall. Renderers and mj_ray both test the real triangles, so the doorway stays open "
            f"in every picture and on /scan -- only physics disagrees, and only the robot finds out.\n"
            f"Ways forward:\n"
            f"  a Floorplan-DSL room (a <env>/json-ld/ beside the mesh) -> `roqsim scenes jsonld-to-scene`,\n"
            f"                              which collides the json-ld's exact convex walls and uses\n"
            f"                              the mesh for looks only. This is the common case.\n"
            f"  --no-collide <model>        import this model's geometry as VISUAL only, and declare\n"
            f"                              its collision as primitives in the world YAML (`box`).\n"
            f"  --collision-only            if the model's <visual> is the sane shape here.\n"
            f"Do not silence this by lowering the probe radius: the passage would still be shut."
        )

    # Ratio of convex-hull volume to the part's own enclosed volume above which the hull is not a
    # stand-in for the mesh but a lie about it. A box scores 1.0; a slab assembly the reflex split
    # handled correctly scores near 1.0. 4.0 leaves room for a chamfered or slightly concave prop
    # while catching the failure this exists for.
    _MAX_HULL_RATIO = 4.0

    def _check_hulls_are_faithful(self, parts, base: str, model_name: str) -> None:
        """Refuse a collidable part whose convex hull swallows its own interior.

        MuJoCo collides a mesh by its CONVEX HULL, so a part that encloses a void -- a room, a closed
        wall ring, a pipe -- becomes solid. The reflex-edge split in ``scene_mesh_io`` is what normally
        prevents that, but it cuts a graph and therefore cannot decompose a *ring*: cutting the four
        inner corners of a rectangular wall loop still leaves the outer faces connected all the way
        round, so every face unions back into one piece. The AWS small warehouse is exactly that shape
        (64 triangles, genus 1, hull/material ratio 31.7).

        This has to fail loudly. The silent outcome is a world whose robot spawns *inside* solid
        geometry: it never moves, every trial reports a collision on the first step, and the campaign
        still produces a complete, plausible, entirely wrong results table.

        Only CLOSED parts are judged. An open patch has no meaningful enclosed volume, so there is
        nothing to compare a hull against and the check abstains rather than guessing.

        **That abstention is why this is not the whole story**, and why it is no longer the only check.
        A building's walls arrive from an STL as open skins, so this one never looks at them -- and it
        would clear them anyway, because a doorway is negligible next to a wall's volume. Whether a way
        through survived is asked separately and scene-wide, by
        :meth:`_check_nothing_is_walled_off`. Keep both: this catches a single part swallowing a room
        (the AWS warehouse, ratio 31.7) cheaply and by name; that one catches what only the finished
        scene can show.
        """
        from scipy.spatial import ConvexHull  # local: only this check needs it

        for pv, pf, _ in parts:
            if len(pv) < 4 or not _is_closed(pv, pf):
                continue
            mesh_vol = abs(_enclosed_volume(pv, pf))
            if mesh_vol <= 1e-9:
                continue
            try:
                hull_vol = float(ConvexHull(pv).volume)
            except Exception:  # noqa: S112 - degenerate/coplanar: no volume to swallow
                continue
            ratio = hull_vol / mesh_vol
            if ratio <= self._MAX_HULL_RATIO:
                continue
            raise SystemExit(
                f"{model_name}: collision part of {base!r} encloses a void that its convex hull would "
                f"fill (hull {hull_vol:.1f} m^3 vs mesh {mesh_vol:.1f} m^3, ratio {ratio:.1f}).\n"
                f"MuJoCo collides meshes by their hull, so importing this as-is puts solid geometry "
                f"through the interior -- a robot spawned inside cannot move, and every trial reports "
                f"a collision on step 1 while still looking like a valid run.\n"
                f"This is the ring case the reflex-edge split cannot cut: it separates faces by "
                f"dihedral, and a closed loop stays connected the long way round.\n"
                f"Ways forward:\n"
                f"  --no-collide {model_name}   import this model's geometry as VISUAL only, and\n"
                f"                              declare its collision as primitives in the world YAML\n"
                f"                              (the `box` plugin) -- right for an axis-aligned shell.\n"
                f"  --collision-only            if the model's <visual> is the sane shape here.\n"
                f"Do not silence this by raising _MAX_HULL_RATIO: the hull would still be wrong."
            )

    def _pin_texture(self, src: Path) -> Path:
        """Copy a texture into the scene dir and return its scene-relative path. The scene must stay
        portable: `assets.lock.json` pins the model it came from, but the Fuel cache is not the port.

        Re-encodes to PNG, because MuJoCo's texture loader takes PNG only while Fuel ships plenty of
        JPEG (Terrazzo005, Asphalt010, the Jersey Barrier). This is a container change, not a
        resample: the pixels are handed over as-is.
        """
        rel = Path("textures") / (src.stem + ".png")
        dst = Path(self.args.out_dir) / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() == ".png":
                shutil.copyfile(src, dst)
            else:
                try:
                    from PIL import Image
                except ImportError as e:  # a silent skip here would look like an untextured scene
                    raise fuel_fetch.FuelError(
                        f"{src.name} needs re-encoding to PNG for MuJoCo, which requires Pillow: {e}"
                    ) from e
                with Image.open(src) as im:
                    im.convert("RGBA" if "A" in im.getbands() else "RGB").save(dst)
            self.textures.add(dst.name)
        return rel

    def _model(self, model, parent: np.ndarray, name_prefix: str, model_dir: Path) -> None:
        prev = getattr(self, "_current_model_dir", None)
        self._current_model_dir = model_dir
        world_model = parent @ _pose_of(model)
        # Names come from `name_prefix`, not from the SDF's own <model name>: nesting and repeated
        # includes make the latter ambiguous, and the prefix is what --no-collide matches against.
        for link in _kids(model, "link"):
            self._link(link, world_model, f"{name_prefix}_{link.get('name') or 'link'}")
        for nested in _kids(model, "model"):  # SDF allows model nesting
            self._model(
                nested, world_model, f"{name_prefix}_{nested.get('name') or 'm'}", model_dir
            )
        for inc in _kids(model, "include"):  # and includes inside models
            self._include(inc, world_model, name_prefix)
        self._current_model_dir = prev

    def _include(self, inc, parent: np.ndarray, name_prefix: str = "") -> None:
        uri = _text(inc, "uri")
        if not uri:
            raise fuel_fetch.FuelError("<include> without <uri>")
        name = inc.get("name") or _text(inc, "name") or uri.rstrip("/").split("/")[-1]
        model_dir = self._model_dir(uri)
        sdf_file = model_dir / "model.sdf"
        if not sdf_file.exists():
            cands = sorted(model_dir.glob("*.sdf"))
            if not cands:
                raise fuel_fetch.FuelError(f"no .sdf inside resolved model {model_dir} (uri {uri})")
            sdf_file = cands[0]
        root = etree.parse(str(sdf_file)).getroot()
        world_inc = parent @ _pose_of(inc)
        for model in [c for c in root.iter() if _ln(c) == "model"]:
            if _ln(model.getparent()) == "sdf":  # top-level models only; nesting handled in _model
                self._model(model, world_inc, _slug_prefix(name_prefix, name), model_dir)

    # ---------------- entry

    def run(self) -> dict:
        root = etree.parse(str(self.args.world)).getroot()
        world = next((c for c in root.iter() if _ln(c) == "world"), None)
        if world is None:
            raise SystemExit("no <world> in the SDF")

        n_inc = len(_kids(world, "include"))
        n_model = len(_kids(world, "model"))
        print(f"world '{world.get('name')}': {n_inc} include(s), {n_model} inline model(s)")

        for inc in _kids(world, "include"):
            self._include(inc, np.eye(4))
        for model in _kids(world, "model"):
            self._model(
                model, np.eye(4), model.get("name") or "model", Path(self.args.world).parent
            )

        if not self.objects:
            raise SystemExit(
                "no geometry emitted. If the world is pure <include> and every asset failed to "
                "resolve, that is a provenance finding for the spec -- not an empty scene."
            )

        self._check_nothing_is_walled_off()

        manifest = {
            "name": self.args.scene_name,
            "source": Path(self.args.world).name,
            "unit_scale": 1.0,  # SDF is metric + Z-up already; no conversion, unlike USD
            "bounds_min": [round(float(v), 4) for v in self.lo],
            "bounds_max": [round(float(v), 4) for v in self.hi],
            "objects": self.objects,
        }
        if self.ground_z is not None:
            manifest["ground_z"] = round(self.ground_z, 6)
        out = Path(self.args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "scene.json").write_text(json.dumps(manifest, indent=2) + "\n")

        if self.args.lock and self.assets:
            fuel_fetch.write_lock(
                Path(self.args.lock), list(self.assets.values()), world=str(self.args.world)
            )
            print(f"pinned {len(self.assets)} external asset(s) -> {self.args.lock}")

        for s in dict.fromkeys(self.skipped):
            print(f"  skipped: {s}")
        if self.textures:
            print(f"pinned {len(self.textures)} texture(s) -> {out / 'textures'}")
        print(
            f"SCENE_OK objects={len(self.objects)} textured={sum(1 for o in self.objects if o.get('texture'))} "
            f"bounds_min={manifest['bounds_min']} bounds_max={manifest['bounds_max']}"
        )
        return manifest


def _slug_prefix(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--world", type=Path, required=True, help="the SDF world file")
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="scene dir to write (scene.json + meshes/)"
    )
    ap.add_argument("--scene-name", help="scene.json `name` (default: out-dir basename)")
    ap.add_argument(
        "--lock", type=Path, help="write assets.lock.json here (pins every fetched model)"
    )
    ap.add_argument("--cache", type=Path, default=fuel_fetch._DEFAULT_CACHE)
    ap.add_argument("--model-path", action="append", default=[], help="extra dir for model:// URIs")
    ap.add_argument(
        "--no-collide",
        action="append",
        default=[],
        metavar="MODEL_SUBSTRING",
        help="import matching models as VISUAL only (repeatable); declare their collision "
        "as primitives in the world YAML. The documented way past the closed-shell "
        "hull check for axis-aligned buildings.",
    )
    ap.add_argument(
        "--collision-only",
        action="store_true",
        help="emit ONLY collidable geometry, skipping the rendered visuals: fastest, ugliest",
    )
    ap.add_argument(
        "--no-split-components",
        dest="split_components",
        action="store_false",
        help="emit one geom per <visual> instead of per connected component. Almost always "
        "wrong for buildings: MuJoCo hulls each mesh, so a merged shell becomes a solid "
        "block that swallows the robot.",
    )
    ap.add_argument("--collision", default="visual", choices=["visual"], help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.scene_name = args.scene_name or Path(args.out_dir).name

    try:
        Importer(args).run()
    except fuel_fetch.FuelError as e:
        print(
            f"\nFAILED: {e}\n\nThis is a finding, not a bug: record it as a resolution_attempt in "
            f"the spec's gap record. Do not substitute a lookalike asset.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
