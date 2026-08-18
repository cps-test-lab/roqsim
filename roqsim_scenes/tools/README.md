# `roqsim_scenes` tools

**These files are wrappers.** The tools themselves are subcommands, so nothing here needs a path:

```bash
roqsim scenes --help                       # every tool, one line each
roqsim scenes sdf-to-scene --help          # one tool's options
python -m pydoc roqsim_scenes.cli.sdf_to_scene   # the reasoning behind it
```

Each `<name>.py` beside this README is three lines onto `roqsim_scenes.cli.<name>`, kept so a tool still
runs when you are standing in this folder. Prefer the command: it works from anywhere and does not
depend on where the repository sits.

## The routes through the pipeline

A world arrives in one of four shapes, and the stage-1 front-end depends on which:

```
published format   sdf-to-scene / usd-to-scene   ->  scene.json + world-space OBJs
occupancy grid     gridmap-to-world              ->  world YAML + map.pgm/map.yaml   (obstacles)
                   gridmap-to-floorplan          ->  floorplan.json                  (architecture)
a drawn plan       dxf-to-floorplan              ->  floorplan.json
only a picture     mapimage-to-floorplan         ->  floorplan.json

then:  floorplan.json --floorplan-to-world-->  scene + world
       scene.json     --scene-to-mjcf------->  the MJCF a world YAML loads
```

Reading a finished world back out: `scene-to-map` for the Nav2 grid a planner needs,
`scene-to-floorplan` and `floorplan-to-png` for the plan view, `cad-to-png` to see which CAD layers
hold the walls before converting.

## Adding one

Write it standalone, then link it in — the recipe is `docs/developer_guide.rst`, "Adding a tool".
`make test` fails while a tool is unregistered, so the tree cannot quietly fall behind the folder.
