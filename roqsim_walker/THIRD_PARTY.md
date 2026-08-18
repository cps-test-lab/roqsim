# Third-party assets & provenance

Every blueprint under `src/roqsim_walker/models/people/<Name>/` carries **its own licence + attribution
in a `CREDITS.txt` next to the files**. Look there for the terms of a specific character; this file
records the provenance chains those one-liners are too short for, and the attribution this package's
own redistribution has to carry.

Only assets whose licence permits redistribution are committed: **CC0** (public domain) or **CC-BY /
CC-BY-SA** (redistributable *with* attribution). Nothing under non-commercial (CC-*-NC) or
no-derivatives (CC-*-ND) terms is added — the same bar as `roqsim_assets`.

**Every locomotion clip in `models/anims/` is CC-BY, so attribution is a condition of
redistribution, not a courtesy.** They derive from CARLA. Keep this file (or an equivalent notice)
with any redistribution of the package or its assets.

## CARLA — *all* locomotion clips (committed, CC-BY 4.0)

- **Upstream:** CARLA Simulator — https://carla.org
- **Source of the assets:** the CARLA content repository,
  https://bitbucket.org/carla-simulator/carla-content/src/master/ — the `AS_*.FBX` animation
  sequences come from there.
- **Copyright:** Computer Vision Center (CVC) at the Universitat Autònoma de Barcelona (UAB)
- **Licence:** CC-BY 4.0 — https://creativecommons.org/licenses/by/4.0/
- **Cite:** *CARLA: An Open Urban Driving Simulator*, Dosovitskiy et al., CoRL 2017.

CARLA's Unreal (UE 4.26) content was extracted from that repository and converted to MuJoCo-ready
form by us; this package ships the results. **The attribution above is complete as it stands** — it
names the upstream, the copyright holder, the licence and the citation directly, so a downstream user
never has to chase a second-hand record to satisfy the CC-BY condition. The CARLA source files
(`AS_*.FBX` and the rest of the content repository) are **not** redistributed here; only the
retargeted clips are.

| Committed here | Derived from (CARLA) | How |
| --- | --- | --- |
| `models/anims/adult/*.npz` | `AS_GEN2_idle`, `AS_male2_WalkCicle0E`, `AS_joggingG3`, `AS_ShortWalkingG3`, `AS_cturnLG3`, `AS_cturnR02G3` | one `AS_*.FBX` action per clip, retargeted onto our skeleton (our `carla_walker_to_clip.py` retargeting) |
| `models/anims/female/*.npz` | `AS_female_Idle0C`, `AS_female_WalkCicle0C`, `AS_female_runCicle0E`, `AS_female_shortCicle`, `AS_female_LTurnCicle`, `AS_female_RTurnCicle` | same, the dedicated female set |

CARLA's "generic adult" locomotion *is* its male-bodied set, which is why `adult/` resolves from
`AS_male2` / `AS_GEN2` / `*G3` sources; see `blueprint.py` for the body-type + gender resolution. The
`kid/` set exists upstream (`AS_Girl_*`) but is **not** bundled here, so `anim_set: kid` resolves to
`adult` by fallback.

**The clips are retargeted, not authored.** They are our derivative works of CARLA's animations, so
the CC-BY obligation travels with them: a downstream package that ships `models/anims/` ships CARLA
attribution too. (The README says the clips "come from this package rather than the imported source" —
that is about *which* artifact supplies them when you import a foreign character, not a claim of
originality.)

## Open-RMF / Gazebo Fuel actors (committed, CC0 + CC-BY 4.0)

Imported with `roqsim walker import-actor` from rigged Gazebo/Open-RMF `<actor>` COLLADA. Each folder's
`CREDITS.txt` carries the Fuel URL, version, licence and author verbatim — that file is the
attribution of record.

| Blueprint | Source | Licence |
| --- | --- | --- |
| `MaleVisitorWalk` | [Fuel: OpenRobotics / *Male visitor* (v2)](https://fuel.gazebosim.org/1.0/OpenRobotics/models/Male%20visitor) | CC0 1.0 Universal |
| `FemaleVisitorWalk` | [Fuel: Luca / *FemaleVisitorWalk* (v1)](https://fuel.gazebosim.org/1.0/Luca/models/FemaleVisitorWalk) | CC-BY 4.0 (author: Wan Yi Seow, Open Robotics) |

Only the **skin** (mesh, textures, per-rig skeleton, authored skin weights) comes from these sources.
Their own `<animation>` tracks are ignored; locomotion is the CARLA-derived set above — so a world
using `FemaleVisitorWalk` still depends on CARLA attribution.

## What is this package's own (Apache-2.0)

The code (`src/roqsim_walker/**.py`), the 17-joint humanoid topology, the navigation stack (A\*,
behaviour tree, ORCA wiring), the world YAML, and the measurement pipeline that produces a
blueprint's collision radii and sole offsets from geometry. The optional `rvo2` dependency
(`[avoidance]`) is not vendored — it is fetched from its own upstream and carries its own licence.

## Adding an actor

Keep the `CREDITS.txt` `roqsim walker import-actor` writes beside the files, honour whatever it states,
and add a row above if the source is a new provenance chain rather than another Fuel actor. An asset
with no recorded licence does not get committed.
