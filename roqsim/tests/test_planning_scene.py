"""The world's static geometry as a MoveIt planning scene.

The failure this guards against is silent in both directions. A prop MISSING from the scene is a
plan that goes through it and a simulator that resolves the contact afterwards; a prop that should
not be there -- a robot link, a box the trial is about to pick up, a decorative mesh the simulator
never collides -- is a planner that refuses motions the robot can make, or refuses to plan at all
because it thinks the arm starts inside its own bench. So the assertions here are about MEMBERSHIP
and about POSE, and every exclusion has a test saying it was a decision rather than an oversight.
"""

from __future__ import annotations

import mujoco
import pytest
import yaml

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.export_moveit import SCENE_FILE, main
from roqsim.planning_scene import BOX, CYLINDER, SPHERE, scene_objects, touching

# A world built by hand, so every case below is one geom and the model says exactly what it is.
_SHAPES = """
<mujoco>
  <asset>
    <mesh name="wedge" vertex="0 0 0  0.2 0 0  0 0.2 0  0 0 0.2"/>
  </asset>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1"/>
    <geom name="rail" type="capsule" size="0.05 0.4" pos="1 0 0.5"/>
    <geom name="blob" type="ellipsoid" size="0.1 0.2 0.3" pos="2 0 0.3"/>
    <geom name="signage" type="mesh" mesh="wedge" pos="4 0 0"/>
    <geom name="decor" type="box" size="0.1 0.1 0.1" pos="3 0 0.1" contype="0" conaffinity="0"/>
    <body name="crate" pos="0 1 0">
      <geom name="crate_body" type="box" size="0.2 0.3 0.4" pos="0 0 0.4"/>
      <geom name="crate_knob" type="sphere" size="0.05" pos="0.2 0 0.6"/>
    </body>
    <body name="cart" pos="0 -1 0">
      <freejoint/>
      <geom name="cart_body" type="box" size="0.2 0.2 0.2" mass="1"/>
    </body>
    <body name="ghost" pos="0 -2 0" mocap="true">
      <geom name="ghost_body" type="box" size="0.2 0.2 0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def shapes():
    return mujoco.MjModel.from_xml_string(_SHAPES)


def _ids(scene) -> set[str]:
    return {obj.id for obj in scene.objects}


def _object(scene, oid: str):
    return next(obj for obj in scene.objects if obj.id == oid)


# -- what an object IS ---------------------------------------------------------------------------


def test_a_static_body_is_one_object_carrying_every_shape_it_is_built_from(shapes):
    """A prop is one name, not one name per geom.

    That is the whole point of the named route: a trial allows, pads or removes THE BENCH. Split
    across five ids, a build has to know that a bench is a top and four legs before it can do any of
    those, and it will get one of them wrong.
    """
    crate = _object(scene_objects(shapes, robot_bodies=set()), "crate")
    assert crate.pos == pytest.approx((0.0, 1.0, 0.0)), "the object sits at its body's pose"
    assert [s.type for s in crate.shapes] == [BOX, SPHERE]
    assert crate.shapes[0].dimensions == pytest.approx((0.4, 0.6, 0.8)), "MuJoCo states HALF sizes"
    # The shapes are placed in the OBJECT's frame, so moving the object moves the whole prop.
    assert crate.shapes[0].pos == pytest.approx((0.0, 0.0, 0.4))
    assert crate.shapes[1].pos == pytest.approx((0.2, 0.0, 0.6))


def test_a_geom_on_the_world_body_is_its_own_object(shapes):
    """The world body is not a thing: it is where everything unattached ends up. One object holding
    every wall in the building could not be padded a wall at a time."""
    assert "rail" in _ids(scene_objects(shapes, robot_bodies=set()))


def test_a_capsule_is_written_exactly_as_a_cylinder_and_two_spheres(shapes):
    """A CollisionObject holds several primitives, so the union is exact.

    A bare cylinder would be short by a hemisphere at each end, and a planner would then sweep the
    gripper through the rounded end of a railing it is meant to clear.
    """
    rail = _object(scene_objects(shapes, robot_bodies=set()), "rail")
    assert [s.type for s in rail.shapes] == [CYLINDER, SPHERE, SPHERE]
    assert rail.shapes[0].dimensions == pytest.approx((0.8, 0.05)), (
        "(height, radius), in that order"
    )
    assert rail.shapes[1].dimensions == pytest.approx((0.05,))
    assert [s.pos[2] for s in rail.shapes] == pytest.approx([0.0, 0.4, -0.4]), "caps at each end"


# -- the exclusions, each of them a decision -----------------------------------------------------


def test_what_a_solid_primitive_cannot_hold_is_NAMED_rather_than_dropped(shapes):
    """Silence here is the whole bug this export exists for: an obstacle missing from the scene is a
    plan that looks fine. Each of these is reported by name, with the reason, so a build knows
    exactly which shape move_group will plan through."""
    scene = scene_objects(shapes, robot_bodies=set())
    skipped = {s.geom: s.why for s in scene.skipped}
    assert set(skipped) == {"ground", "blob", "signage"}
    assert "infinite plane" in skipped["ground"], "and the robot stands on it"
    assert "ellipsoid" in skipped["blob"]
    assert "convex hull" in skipped["signage"], "MuJoCo's mesh shape is not MoveIt's"
    assert not _ids(scene) & {"ground", "blob", "signage"}


def test_visual_only_geometry_is_left_out_and_is_not_a_limitation(shapes):
    """``contype``/``conaffinity`` both zero is this substrate's spelling for decoration. The
    simulator does not collide it, so a planner that did would refuse motions the robot can make --
    which makes this an exclusion rather than something the export failed to express."""
    scene = scene_objects(shapes, robot_bodies=set())
    assert "decor" not in _ids(scene)
    assert "decor" not in {s.geom for s in scene.skipped}


def test_anything_that_moves_is_reported_rather_than_written(shapes):
    """A free prop and a driven (mocap) obstacle both have a pose that is only true right now.

    Writing one states something false the moment the trial starts, and a stale obstacle standing
    where the part no longer is costs exactly what a missing one does.
    """
    scene = scene_objects(shapes, robot_bodies=set())
    assert not _ids(scene) & {"cart", "ghost"}
    assert set(scene.movable) == {"cart", "ghost"}


def test_the_robots_own_bodies_stay_out_of_the_scene(shapes):
    """A link that is also a world object collides with itself, and move_group then refuses every
    request from a start state it calls invalid."""
    crate = mujoco.mj_name2id(shapes, mujoco.mjtObj.mjOBJ_BODY, "crate")
    assert "crate" not in _ids(scene_objects(shapes, robot_bodies={crate}))


def test_two_things_that_would_share_an_id_are_refused():
    """A scene addresses an object by its id, so one id for two obstacles means a trial that allows,
    pads or removes one of them gets the other."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <geom name="crate" type="box" size="0.1 0.1 0.1" pos="2 0 0.1"/>
          <body name="crate" pos="0 0 0"><geom type="box" size="0.1 0.1 0.1"/></body>
        </worldbody></mujoco>
        """
    )
    with pytest.raises(ValueError, match="both be the collision object 'crate'"):
        scene_objects(model, robot_bodies=set())


# -- the frame -----------------------------------------------------------------------------------


def test_poses_are_written_in_the_frame_the_planner_works_in(shapes):
    """MoveIt plans in the URDF's root link -- the arm's own base, not the world origin -- so a scene
    written in world coordinates puts every prop wherever the arm happens to stand."""
    crate = mujoco.mj_name2id(shapes, mujoco.mjtObj.mjOBJ_BODY, "crate")
    in_world = _object(scene_objects(shapes, robot_bodies=set()), "rail")
    in_crate = _object(scene_objects(shapes, robot_bodies=set(), frame_body=crate), "rail")
    assert in_world.pos == pytest.approx((1.0, 0.0, 0.5))
    assert in_crate.pos == pytest.approx((1.0, -1.0, 0.5)), "the crate's frame is 1 m along +y"


# -- a compiled cell, through the CLI ------------------------------------------------------------


def _arm(**extra) -> dict:
    arm = {
        "model": "ur5e",
        "prefix": "ur5e_",
        "pos": [0.0, 0.0, 0.76],
        "end_effector": {
            "model": "robotiq_2f85",
            "site": "attachment_site",
            "pos": [0.0, 0.0, 0.011],
        },
    }
    arm.update(extra)
    return {"spawn_arm": arm, "name": "ur5e"}


def _prop(model: str, name: str, x: float, z: float, motion: str = "static") -> dict:
    return {
        "spawn_model": {
            "model": model,
            "prefix": f"{name}_",
            "motion": motion,
            "pose": {"position": {"x": x, "y": 0.0, "z": z}},
        },
        "name": name,
    }


def _export(tmp_path, world: dict, *extra):
    (tmp_path / "cell.yaml").write_text(yaml.safe_dump(world), encoding="utf-8")
    out = tmp_path / "gen"
    # Far below the Setup Assistant's 10000: the collision matrix is not what is under test here.
    code = main(
        ["--world", str(tmp_path / "cell.yaml"), "--out", str(out), "--samples", "40", *extra]
    )
    assert code == 0
    return out


def _scene_of(out):
    return yaml.safe_load((out / SCENE_FILE).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cell(tmp_path_factory):
    """An arm on a bench, in the default walled room, with a part it is about to pick up."""
    world = {
        "sim": {},
        "plugins": [
            _prop("industrial_table", "bench", 0.0, 0.0),
            _prop("graspable_box", "part", 0.3, 0.8, motion="physics"),
            _arm(),
        ],
    }
    return _export(tmp_path_factory.mktemp("cell"), world, "--tip-site", "pinch", "--scene")


def test_the_scene_names_the_cells_props_with_their_shapes_and_poses(cell):
    """Every number comes off the compiled world. A build copying them by hand is the drift the
    whole export exists to prevent."""
    scene = _scene_of(cell)
    objects = {obj["id"]: obj for obj in scene["world"]["collision_objects"]}
    assert "bench_industrial_table" in objects
    bench = objects["bench_industrial_table"]
    assert [p["type"] for p in bench["primitives"]] == [BOX] * 5, "a top and four legs"
    top, leg = bench["primitives"][0], bench["primitives"][1]
    assert top["dimensions"] == pytest.approx([2.9, 1.56, 0.04])
    assert leg["dimensions"] == pytest.approx([0.08, 0.08, 0.72])
    # The arm is bolted to the benchtop at z=0.76 and plans in its own base link, so the bench --
    # whose own frame is on the floor -- sits 0.76 m BELOW the frame the planner works in, and its
    # top stays 0.74 m up in the bench's own frame.
    assert bench["pose"]["position"]["z"] == pytest.approx(-0.76)
    assert bench["primitive_poses"][0]["position"]["z"] == pytest.approx(0.74)
    assert objects["wall_px"]["pose"]["position"]["z"] == pytest.approx(1.0 - 0.76)


def test_the_scene_is_a_diff_that_states_nothing_about_the_robot(cell):
    """A non-diff scene replaces everything in it, the robot state included, so a bring-up applying
    one hands the planner a robot at all-zero joints."""
    scene = _scene_of(cell)
    assert scene["is_diff"] is True
    assert scene["robot_state"]["is_diff"] is True
    for obj in scene["world"]["collision_objects"]:
        assert obj["operation"] == 0, "ADD"
        assert obj["header"]["frame_id"] == "base_link", "the frame move_group plans in"
        assert len(obj["primitives"]) == len(obj["primitive_poses"]), "read pairwise by MoveIt"


def test_the_arms_own_links_are_not_objects_in_its_scene(cell):
    """They are the URDF's job. A link that is also a world object collides with itself."""
    ids = {obj["id"] for obj in _scene_of(cell)["world"]["collision_objects"]}
    assert not any(oid.startswith("ur5e_") or oid.startswith("robotiq") for oid in ids)
    assert ids == {"wall_px", "wall_nx", "wall_py", "wall_ny", "bench_industrial_table"}


def test_the_part_the_trial_picks_up_is_not_in_the_scene(cell):
    """It is a free body: its pose at export time is not where it will be, and a collision object on
    the thing being grasped makes every approach a collision and every grasp pose unplannable."""
    ids = {obj["id"] for obj in _scene_of(cell)["world"]["collision_objects"]}
    assert not any(oid.startswith("part_") for oid in ids)


def test_a_world_with_no_props_writes_an_empty_scene_and_says_so(tmp_path, caplog):
    """Not an error -- a bare cell is a legitimate world -- but never silent: a scene with nothing in
    it means move_group plans as if the robot stood in free space."""
    (tmp_path / "bare.xml").write_text(
        "<mujoco><worldbody/></mujoco>",
        encoding="utf-8",
    )
    world = {"sim": {"world": str(tmp_path / "bare.xml")}, "plugins": [_arm()]}
    with caplog.at_level("WARNING"):
        out = _export(tmp_path, world, "--tip-site", "pinch", "--scene")
    assert _scene_of(out)["world"]["collision_objects"] == []
    assert "planning scene is EMPTY" in caplog.text


def test_a_prop_the_arm_is_already_touching_is_reported(tmp_path, caplog):
    """CheckStartStateCollision refuses a request whose start state is in collision, so move_group
    plans NOTHING -- which reads like a planner that will not work rather than like a scene saying
    the arm is inside its own furniture. MuJoCo reports no contact for the pair (both are welded to
    the world), so only a distance query finds it."""
    world = {
        "sim": {},
        "plugins": [_prop("graspable_box", "clutter", 0.0, 0.85), _arm()],
    }
    with caplog.at_level("WARNING"):
        out = _export(tmp_path, world, "--tip-site", "pinch", "--scene")
    assert "clutter_graspable_box" in caplog.text
    assert "CheckStartStateCollision" in caplog.text
    # The file says so too: whoever reads the scene later did not see the export's log.
    assert "TOUCH the robot" in (out / SCENE_FILE).read_text(encoding="utf-8")


def test_two_arms_in_one_description_write_the_scene_in_their_common_root(tmp_path):
    """That root is a pure frame the compiled model puts at the world origin, so the props' poses
    are their world poses -- and neither arm's own base is the frame, which would put the scene in
    the wrong place for the other."""

    def arm(name: str, y: float) -> dict:
        return {
            "spawn_arm": {
                "model": "ur5e",
                "prefix": f"{name}_",
                "namespace": name,
                "pos": [0.0, y, 0.76],
            },
            "name": name,
            "components": [{"arm_controller": {"joint_prefix": f"{name}_"}}],
        }

    world = {
        "sim": {},
        "plugins": [
            _prop("industrial_table", "bench", 0.0, 0.0),
            arm("left", -0.9),
            arm("right", 0.9),
        ],
    }
    out = _export(tmp_path, world, "--arm", "left,right", "--scene")
    objects = {obj["id"]: obj for obj in _scene_of(out)["world"]["collision_objects"]}
    assert objects["bench_industrial_table"]["pose"]["position"] == pytest.approx(
        {"x": 0.0, "y": 0.0, "z": 0.0}
    )
    assert objects["wall_px"]["header"]["frame_id"] == "base_link"


def test_an_arm_with_no_prefix_is_refused_rather_than_given_an_empty_scene(tmp_path):
    """The URDF export selects the robot's bodies by name prefix, so with none it takes every body in
    the world and the scene would come out empty for a reason nothing in it explains."""
    world = {"sim": {}, "plugins": [_prop("industrial_table", "bench", 0.0, 0.0), _arm(prefix="")]}
    with pytest.raises(SystemExit, match="needs the arm to have an MJCF `prefix:`"):
        _export(tmp_path, world, "--scene")


def test_a_robot_that_is_not_welded_down_is_refused(tmp_path):
    """Its base rides a free joint, so MoveIt plans in a frame TF provides. A prop's pose is fixed in
    the WORLD and the offset between the two is a run-time quantity, so a scene written here would be
    right only until the base moved."""
    world = {
        "sim": {},
        "plugins": [
            {
                "spawn_robot": {"model": "husky_a200", "pose": {"position": {"x": 0.0, "y": 0.0}}},
                "name": "h",
            },
            {
                "spawn_arm": {
                    "model": "ur10e",
                    "prefix": "ur10e_",
                    "mount": {"robot": "h", "body": "base_link"},
                    "pos": [0.25, 0.0, 0.2587],
                },
                "name": "arm",
            },
        ],
    }
    with pytest.raises(SystemExit, match="base is not welded"):
        _export(tmp_path, world, "--scene")


def test_the_robot_geoms_a_touch_is_measured_against_are_the_ones_the_urdf_collides(cell):
    """A visual-only geom becomes a ``<visual>`` in the URDF and never a ``<collision>``, so MoveIt
    never checks it. Measuring against it would report a start-state collision move_group does not
    have -- and the UR5e's base is exactly that case, its visual shell resting on the benchtop while
    the geom it collides with clears it."""
    engine = Engine(
        load_config_from_dict(
            {
                "sim": {},
                "plugins": [_prop("industrial_table", "bench", 0.0, 0.0), _arm()],
            },
            base_dir=cell,
        )
    )
    engine.setup()
    model = engine.ctx.model
    robot = {
        b
        for b in range(1, model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("ur5e_")
    }
    scene = scene_objects(model, robot_bodies=robot)
    assert touching(model, engine.ctx.data, robot_bodies=robot, objects=scene.objects) == []
