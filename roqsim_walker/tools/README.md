# Walker import tools

**`import_actor.py` is a wrapper.** The tool itself is a subcommand, so it needs no path:

```bash
roqsim walker --help                       # every tool, one line each
roqsim walker import-actor --help          # the tool's options
python -m pydoc roqsim_walker.cli.import_actor   # the reasoning behind it
```

`import_actor.py` beside this README is three lines onto `roqsim_walker.cli.import_actor`, kept so the
tool still runs when you are standing in this folder. Prefer the command: it works from anywhere and
does not depend on where the repository sits.

## `roqsim walker import-actor` — rigged COLLADA actor → walker blueprint

Turns a Gazebo/Open-RMF animated `<actor>` (a rigged, textured `.dae`) into a `people/<Name>/`
blueprint the `walker` plugin can spawn. Only the **skin** is imported — the mesh, textures,
per-rig skeleton and authored skin weights; **locomotion comes from this package's own clips**
(`models/anims/`), so the source's animation tracks are ignored. It is the offline counterpart to
the CARLA character export pipeline.

```bash
pip install 'roqsim_walker[import]'   # pulls pycollada

roqsim walker import-actor <actor.dae> --name FemaleVisitorWalk --anim-set female \
    --credits-source <fuel-url> --credits-licence "CC-BY 4.0" --credits-author "..."
```

Writes `<name>.obj` + `.mtl` + textures, `<name>.walker.json` (skeleton/collision/sole), and the
authored `<name>.weights.npz`, plus a `CREDITS.txt`. Like the `roqsim_assets` prop tools, it
**only produces files** — committing them and honouring the licence is on you (CC0 / CC-BY /
CC-BY-SA only; see the package [`THIRD_PARTY.md`](../THIRD_PARTY.md)).

### What it does, and the traps it handles

- **Rig mapping is semantic, not positional.** The source's finer skeleton (31 joints: extra spine
  link, fingers, thumbs) is mapped onto our fixed 17 via `JOINT_MAP` (geometry) and `WEIGHT_FOLD`
  (skin weight). An unmappable joint is a hard error, never a silent drop.
- **Bind pose vs rest pose.** The source is T-posed, but our clips are authored against an arms-down
  rest and `skin.add_skin(tpose=True)` rotates the bind arms out to meet the mesh. So the emitted
  offsets are canonical arms-down (`elbow/wrist/knee/ankle` = `[0,0,-len]`), with lengths taken from
  the source bind — **not** the raw T-pose positions.
- **Facing.** CARLA faces −X; the Open-RMF actors face +X. `--flip` sets −X; the default is +X.
  This is why `skin.py` separates `flip` from `tpose` (CARLA couples them).
- **Anim set.** `blueprint._infer_gender` needs a CARLA-style `…F02` name and cannot read
  `FemaleVisitorWalk`; pass `--anim-set female|adult|kid` explicitly or the gait defaults to `adult`.
- **Held props.** `--exclude-material <group>` (repeatable) drops a material group and its texture —
  e.g. `--exclude-material Handbag-material` to remove the female visitor's bag. Excluded geometry is
  left out of the collision/sole measurements too, so a dropped bag doesn't fatten the hand's capsule.

### What it measures

Everything `walker.json` says of the mesh's shape is measured on the mesh turned as the skin turns it,
against the bones MuJoCo poses the skin with (`skin._bind_bones`), which are what the foot grounding
adds a sole offset to and what each collision capsule rides -- not the source rig's own bones, which
the walker's straight limbs do not keep (`import_actor.measure`): the height, the collision radii,
the heel and toe points (the lowest of the shoe's vertices the ankle and the toe carry) with
`foot_tip`, and each shoe's hull, on which the walker is grounded. The pelvis height and the bone
lengths are the source rig's.

### The bundled visitors, made again

Both are written by the tool from their Gazebo Fuel sources (download each model's zip from the page
its `CREDITS.txt` names and unpack it into a folder of the name used below); nothing in their folders is edited by hand:

```bash
roqsim walker import-actor FemaleVisitorWalk/meshes/FemaleVisitorWalk.dae --name FemaleVisitorWalk \
    --anim-set female --exclude-material Handbag-material \
    --credits-source "https://fuel.gazebosim.org/1.0/Luca/models/FemaleVisitorWalk (version 1)" \
    --credits-licence "CC-BY 4.0 (http://creativecommons.org/licenses/by/4.0/)" \
    --credits-author "Wan Yi Seow <wanyi@openrobotics.org>; Fuel owner: Luca"
roqsim walker import-actor "Male visitor/meshes/MaleVisitorWalk.dae" --name MaleVisitorWalk \
    --anim-set adult \
    --credits-source "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Male visitor (version 2)" \
    --credits-licence "CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)" \
    --credits-author "Wan Yi Seow <wanyi@openrobotics.org>; Fuel owner: OpenRobotics"
```

Without `--out` it writes into this package's `models/people/`. The thumbnails
(`<Name>.thumb.png`) are `roqsim render <obj> --size 480x480`.

### The Open-RMF actor family

One rig, so one importer unlocks the set (all on Fuel): `Luca/FemaleVisitorWalk` (CC-BY 4.0),
`OpenRobotics/Male visitor` → `MaleVisitorWalk` (CC0), and the hospital actors
(`DoctorFemaleWalk`, `NurseFemaleWalk`, `OpScrubsWalk`, `PatientWalkingCane`, `VisitorKidWalk`).
Note `anims/` ships only `adult` and `female` today — a `kid` actor falls back to `adult`.
