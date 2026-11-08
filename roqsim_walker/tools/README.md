# Walker import tools

**These files are wrappers.** The tool itself is a subcommand, so nothing here needs a path:

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

### The Open-RMF actor family

One rig, so one importer unlocks the set (all on Fuel): `Luca/FemaleVisitorWalk` (CC-BY 4.0),
`OpenRobotics/Male visitor` → `MaleVisitorWalk` (CC0), and the hospital actors
(`DoctorFemaleWalk`, `NurseFemaleWalk`, `OpScrubsWalk`, `PatientWalkingCane`, `VisitorKidWalk`).
Note `anims/` ships only `adult` and `female` today — a `kid` actor falls back to `adult`.

## `roqsim walker derive-hivis-worker` -- the industrial worker, from an imported actor

No redistributable rigged character with a high-visibility vest was found, so `WorkerHiVisWalk` is
dressed from the Open-RMF *OpScrubsWalk* actor (CC-BY 4.0): import it, then derive.

```bash
roqsim walker import-actor OpScrubsWalk.dae --name OpScrubsWalk --out /tmp/people \
    --credits-source "https://fuel.gazebosim.org/1.0/Luca/models/OpScrubsWalk (version 1)" \
    --credits-licence "CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)" \
    --credits-author "Wan Yi Seow <wanyi@openrobotics.org>; Fuel owner: Luca"
roqsim walker derive-hivis-worker /tmp/people/OpScrubsWalk \
    roqsim_walker/src/roqsim_walker/models/people/WorkerHiVisWalk
```

It keeps the actor's skeleton, skin weights, body and long trousers; removes the surgical mask, its
ties and the name badge and closes the mouth beneath; subdivides the mesh once (Loop) so its outline
is round up close; and repaints the texture by the 3-D position each texel lands on: a dark work
shirt with a yellow vest and two reflective bands over the torso, charcoal trousers, dark safety
shoes, hair for the scrub cap. It is written for that source's layout and refuses any other, naming
the UV island it cannot place. The thumbnail is `roqsim render <obj> --size 480x480`.

