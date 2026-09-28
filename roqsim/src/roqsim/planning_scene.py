"""The world's static geometry as a MoveIt planning scene, read off the compiled model.

What this is for
================
``roqsim export moveit`` describes the **robot**: the URDF's links, the SRDF's groups, the action a
trajectory is executed against. The world the robot stands in is in none of those files, so
``move_group`` plans straight through the bench the arm is bolted to and the wall beside it, and the
simulator resolves the contact afterwards. There is no error and no warning -- the symptom is a plan
that looks fine and an arm that drives into furniture. This writes the other half: the static
collidable shapes the compiled world carries, as ``moveit_msgs/CollisionObject`` entries in a
``moveit_msgs/PlanningScene``, from the same compiled model the URDF is read off, so the two cannot
drift apart the way a hand-kept pair does.

Two routes carry the world to the planner, and they answer different questions
==============================================================================
A depth sensor plus MoveIt's octomap updater is the other one, and it already works here: a
``realsense_d435`` with ``points: true`` publishes a cloud the ``PointCloudOctomapUpdater`` turns
into occupied voxels. That route reports what is **in view**, whatever its shape, and it follows
things that move. What it cannot give is a **named** object -- and a name is what MoveIt needs to
attach a thing to the gripper, to allow a link against it, to pad it, or to remove it when the trial
has picked it up. This file gives the names; the octomap gives what nothing declared.

They compose rather than compete. MoveIt's sensor filter removes the robot's own links and anything
attached to them from the incoming cloud; it does not remove world collision objects, so a prop that
is both declared here and in view is carried twice -- as an object and as voxels. That costs
checking time and makes the scene slightly conservative where the two disagree at the surface; it
does not make it wrong.

What is written
===============
One ``CollisionObject`` per **static body**, at that body's pose, carrying every collidable shape the
body is built from -- so an ``industrial_table`` is one object called ``industrial_table`` holding
its top and its four legs, and a whole prop can be allowed, padded or removed under one name. Geoms
hanging directly off the world body are each their own object, named after the geom, because the
world body is not a thing: it is where everything unattached ends up, and one object holding every
wall in the building could not be padded a wall at a time.

Static means MuJoCo's own sense of it -- welded into the same weld tree as the world body, so nothing
in the simulation can move it. That is the property this export needs, because a pose written into a
scene at export time is only true for as long as nothing moves the thing.

What is NOT written, and why
============================
Every one of these is reported by name rather than dropped quietly, and the file itself records them:

* **Anything with a degree of freedom** -- a free prop physics moves, a ``motion: driven`` mocap
  obstacle, a pedestrian, another robot's links. Its pose at export time is not where it will be, so
  writing it states something false. A trial that needs one in the scene publishes it while it runs.
* **Visual-only geometry** -- ``contype`` and ``conaffinity`` both zero, this substrate's convention
  for decoration. The simulator does not collide it, so a planner reasoning about it would refuse
  motions the robot can make.
* **Mesh geoms.** A ``shape_msgs/Mesh`` is explicit vertices and triangles, and MuJoCo collides a
  mesh geom as its **convex hull** -- so neither the triangles nor the hull is a shape the two
  engines agree on, and the disagreement is largest exactly where a hull fills a span the prop exists
  to leave open. A prop meant for a planning scene carries primitive collision geoms behind its
  visual mesh, which ``roqsim assets collision`` measures.
* **Plane geoms.** A MuJoCo plane is infinite, and the robot stands on it: as a half-space in the
  scene it puts the start state in collision and every request is then refused before it is planned.
* **Ellipsoid, height-field and SDF geoms** -- ``shape_msgs/SolidPrimitive`` has a box, a sphere, a
  cylinder and a cone, and none of these is any of those.

A capsule is not in that list: it is exactly a cylinder plus two spheres, and a ``CollisionObject``
holds several primitives, so it is written exactly rather than approximated.

The frame
=========
Poses are written in the frame ``move_group`` plans in, which for a robot welded to the world is the
URDF's root link -- the arm's own base, not the world origin -- and for several arms in one
description is the common root the parts hang off, which the compiled model puts at the world origin.
A description whose base is **not** welded has a ``virtual_joint`` and plans in a frame TF provides
(``odom``), and nothing in the compiled model says where that frame is: such a scene is refused here
rather than written in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import yaml

from .export_mesh import _quat_to_mat
from .export_urdf import body_reference_pose

#: ``shape_msgs/SolidPrimitive`` type codes, as numbers because that is what the message carries: a
#: bring-up node filling a ``PlanningScene`` with ``rosidl_runtime_py`` reads them straight out of
#: the emitted file, so a friendlier spelling here would have to be translated there.
BOX = 1
SPHERE = 2
CYLINDER = 3

#: ``moveit_msgs/CollisionObject.ADD``.
ADD = 0

#: Separation at or below which a scene object counts as touching the robot. Not a clearance
#: requirement -- a margin the caller wants belongs in MoveIt's link padding -- just enough to catch
#: the arm that is bolted to a bench, whose base geom and the benchtop meet at exactly zero.
TOUCH = 1e-6

_KIND_UNWRITABLE = {
    int(mujoco.mjtGeom.mjGEOM_MESH): (
        "a mesh: MoveIt takes explicit triangles and MuJoCo collides the convex hull, so neither "
        "is a shape both engines agree on. Give the prop primitive collision geoms behind its "
        "visual mesh (`roqsim assets collision` measures the difference)"
    ),
    int(mujoco.mjtGeom.mjGEOM_PLANE): (
        "an infinite plane, and the robot stands on it: a half-space here puts the start state in "
        "collision and every request is refused before it is planned"
    ),
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): (
        "an ellipsoid, which is not a box, a sphere, a cylinder or a cone -- the four shapes "
        "shape_msgs/SolidPrimitive has"
    ),
    int(mujoco.mjtGeom.mjGEOM_HFIELD): ("a height field, which has no SolidPrimitive spelling"),
    int(mujoco.mjtGeom.mjGEOM_SDF): (
        "a signed-distance geom, which has no SolidPrimitive spelling"
    ),
}


@dataclass(frozen=True)
class Shape:
    """One ``shape_msgs/SolidPrimitive`` and where it sits in its object's frame.

    ``quat`` is MuJoCo's order (w, x, y, z); the message's order is applied where it is written out.
    """

    type: int
    dimensions: tuple[float, ...]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class SceneObject:
    """One ``moveit_msgs/CollisionObject``: a prop, where it is, and the shapes it is built from."""

    id: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    shapes: list[Shape] = field(default_factory=list)
    #: The MuJoCo geoms this object was read from, so a caller can ask the model about them again.
    geoms: tuple[int, ...] = ()


@dataclass(frozen=True)
class Skipped:
    """A collidable shape the world has that a ``CollisionObject`` cannot hold."""

    geom: str
    why: str


@dataclass(frozen=True)
class Scene:
    """What one world yielded: the objects, and everything it could not write."""

    objects: list[SceneObject]
    skipped: list[Skipped] = field(default_factory=list)
    #: Names of the bodies that move -- a free prop, a driven obstacle, another robot -- reported at
    #: the root of the thing that moves rather than per link.
    movable: list[str] = field(default_factory=list)


def _name(model: mujoco.MjModel, objtype, i: int, fallback: str) -> str:
    return mujoco.mj_id2name(model, objtype, i) or fallback


def _mat_to_quat(rot: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rot, dtype=float).reshape(9))
    return quat


def geom_shapes(model: mujoco.MjModel, g: int) -> list[Shape]:
    """The SolidPrimitives geom ``g`` *is*, in the geom's own frame; ``[]`` where it is none of them.

    A capsule becomes three: MuJoCo's capsule is a cylinder of half-length ``size[1]`` capped by two
    spheres of radius ``size[0]``, and that union is exact rather than an approximation, which is
    what a bare cylinder would be -- short by a hemisphere at each end, so a planner would sweep the
    gripper through the rounded ends of a railing it is meant to avoid.
    """
    gtype = int(model.geom_type[g])
    size = [float(v) for v in model.geom_size[g]]
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        return [Shape(BOX, (2 * size[0], 2 * size[1], 2 * size[2]))]
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return [Shape(SPHERE, (size[0],))]
    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return [Shape(CYLINDER, (2 * size[1], size[0]))]
    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, half = size[0], size[1]
        return [
            Shape(CYLINDER, (2 * half, radius)),
            Shape(SPHERE, (radius,), (0.0, 0.0, half)),
            Shape(SPHERE, (radius,), (0.0, 0.0, -half)),
        ]
    return []


def _in_frame(
    frame_pos: np.ndarray, frame_rot: np.ndarray, pos: np.ndarray, quat: np.ndarray
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """A world pose expressed in ``frame``."""
    local_pos = frame_rot.T @ (np.asarray(pos, dtype=float) - frame_pos)
    local_rot = frame_rot.T @ _quat_to_mat(quat)
    return tuple(float(v) for v in local_pos), tuple(float(v) for v in _mat_to_quat(local_rot))


def _shape_in_geom(model: mujoco.MjModel, g: int, shape: Shape) -> Shape:
    """``shape``, given in the geom's frame, moved into the frame of the geom's body."""
    quat = np.asarray(model.geom_quat[g], dtype=float)
    pos = np.asarray(model.geom_pos[g], dtype=float) + _quat_to_mat(quat) @ np.asarray(
        shape.pos, dtype=float
    )
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, quat, np.asarray(shape.quat, dtype=float))
    return Shape(
        shape.type,
        shape.dimensions,
        tuple(float(v) for v in pos),
        tuple(float(v) for v in out),
    )


def _moving_root(model: mujoco.MjModel, body: int) -> int:
    """The highest ancestor of ``body`` that is not welded to the world.

    So a free prop reports as itself and an arm's forearm reports as the arm, rather than a world
    with one pedestrian in it listing seventeen limb names.
    """
    root, cur = body, body
    while cur > 0:
        if int(model.body_weldid[cur]) != 0:
            root = cur
        cur = int(model.body_parentid[cur])
    return root


def scene_objects(model: mujoco.MjModel, *, robot_bodies: set[int], frame_body: int = 0) -> Scene:
    """Read the world's static collision geometry off the compiled model.

    ``robot_bodies`` is every body of every robot the description covers -- their geometry is the
    URDF's job, and a link that is also a world object collides with itself. ``frame_body`` is the
    body whose frame the poses are written in, i.e. the body the URDF's root link stands for; the
    world body (0) is right for a description whose root is the world origin.
    """
    frame_pos, frame_quat = body_reference_pose(model, frame_body)
    frame_rot = _quat_to_mat(frame_quat)

    objects: dict[str, SceneObject] = {}
    owner: dict[str, str] = {}
    skipped: list[Skipped] = []
    movable: list[str] = []
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        if body in robot_bodies:
            continue
        mocap = int(model.body_mocapid[body]) >= 0
        if mocap or int(model.body_weldid[body]) != 0:
            # Its pose at export time is not where it will be. Reported once per moving thing.
            root = body if mocap else _moving_root(model, body)
            name = _name(model, mujoco.mjtObj.mjOBJ_BODY, root, f"body_{root}")
            if name not in movable:
                movable.append(name)
            continue
        if not (int(model.geom_contype[g]) or int(model.geom_conaffinity[g])):
            continue
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g, f"geom_{g}")
        shapes = geom_shapes(model, g)
        if not shapes:
            why = _KIND_UNWRITABLE.get(
                int(model.geom_type[g]), f"a {mujoco.mjtGeom(int(model.geom_type[g])).name} geom"
            )
            skipped.append(Skipped(geom_name, why))
            continue

        if body == 0:
            # A geom on the world body is its own object: the world body is not a thing, it is where
            # everything unattached ends up, so grouping by it would make one object of every wall.
            oid, source = geom_name, f"geom {g}"
            base_pos = np.asarray(model.geom_pos[g], dtype=float)
            base_quat = np.asarray(model.geom_quat[g], dtype=float)
        else:
            oid = _name(model, mujoco.mjtObj.mjOBJ_BODY, body, f"body_{body}")
            source = f"body {body}"
            base_pos, base_quat = body_reference_pose(model, body)
            shapes = [_shape_in_geom(model, g, s) for s in shapes]

        if oid in objects and owner[oid] != source:
            raise ValueError(
                f"two different things in this world would both be the collision object {oid!r} "
                f"({source} and {owner[oid]} of the compiled model). A scene addresses an object by "
                "its id, so one id for two obstacles means a trial that allows, pads or removes one "
                "of them gets the other. Rename one of them in the world."
            )
        obj = objects.get(oid)
        if obj is None:
            pos, quat = _in_frame(frame_pos, frame_rot, base_pos, base_quat)
            obj = SceneObject(id=oid, pos=pos, quat=quat)
            objects[oid] = obj
            owner[oid] = source
        obj.shapes.extend(shapes)
        obj.geoms = (*obj.geoms, g)

    return Scene(objects=list(objects.values()), skipped=skipped, movable=movable)


def touching(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    robot_bodies: set[int],
    objects: list[SceneObject],
    tolerance: float = TOUCH,
) -> list[tuple[str, str, float]]:
    """``(object id, robot link, separation)`` for every object the robot rests on or overlaps.

    Asked of geometry rather than of contacts on purpose: MuJoCo generates no contact between two
    bodies in the same weld tree, and an arm bolted to a bench is exactly that pair -- so the one
    case that matters most would report nothing. MoveIt has no such exemption. It runs
    ``CheckStartStateCollision`` against the scene it is given and refuses every request whose start
    state is in collision, which reads as a planner that will not plan rather than as a scene that
    says the arm is inside its own bench. What to do about it is the caller's decision -- allow the
    pair, pad the object back, or leave the bench out -- so this only reports.

    The robot side is its **collidable** geoms, which is what the URDF emits as ``<collision>`` and
    therefore all MoveIt checks. A visual shell resting on a benchtop is not a collision the planner
    has, and reporting it would send a build looking for one.
    """
    mujoco.mj_kinematics(model, data)
    robot = [
        g
        for g in range(model.ngeom)
        if int(model.geom_bodyid[g]) in robot_bodies
        and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))
    ]
    found: dict[tuple[str, str], float] = {}
    for obj in objects:
        for g in obj.geoms:
            for r in robot:
                dist = float(mujoco.mj_geomDistance(model, data, g, r, tolerance + 1e-3, None))
                if dist > tolerance:
                    continue
                link = _name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[r]),
                    f"body_{int(model.geom_bodyid[r])}",
                )
                key = (obj.id, link)
                found[key] = min(dist, found.get(key, dist))
    return [(oid, link, dist) for (oid, link), dist in sorted(found.items())]


def _num(value) -> float:
    """A plain float a YAML dump can carry, with -0.0 normalised away."""
    return float(round(float(value), 9)) or 0.0


def _pose(pos, quat) -> dict:
    """``geometry_msgs/Pose`` from a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in quat)
    return {
        "position": {"x": _num(pos[0]), "y": _num(pos[1]), "z": _num(pos[2])},
        "orientation": {"x": _num(x), "y": _num(y), "z": _num(z), "w": _num(w)},
    }


def planning_scene_yaml(
    scene: Scene,
    *,
    frame: str,
    name: str,
    robot_model_name: str = "",
    contacts: list[tuple[str, str, float]] = (),
) -> str:
    """The scene as a ``moveit_msgs/PlanningScene``, and a header saying what it leaves out.

    ``is_diff`` is true, and ``robot_state.is_diff`` with it: applying this ADDS the world's objects
    to the scene ``move_group`` already has. A non-diff scene replaces everything in it, the robot
    state included, so a bring-up that applied one would hand the planner a robot at all-zero joints.
    """
    objects = [
        {
            "id": obj.id,
            "header": {"frame_id": frame},
            "operation": ADD,
            "pose": _pose(obj.pos, obj.quat),
            "primitives": [
                {"type": s.type, "dimensions": [_num(d) for d in s.dimensions]} for s in obj.shapes
            ],
            "primitive_poses": [_pose(s.pos, s.quat) for s in obj.shapes],
        }
        for obj in scene.objects
    ]
    body = {
        "name": name,
        "robot_model_name": robot_model_name or name,
        "is_diff": True,
        "robot_state": {"is_diff": True},
        "world": {"collision_objects": objects},
    }

    notes = ""
    if scene.skipped:
        notes += "#\n# NOT in this scene -- geometry the world has that a collision object cannot hold:\n"
        notes += "".join(f"#   {s.geom}: {s.why}\n" for s in scene.skipped)
    if scene.movable:
        notes += (
            "#\n# NOT in this scene -- these move, so a pose written now would be false by the time\n"
            "# it is read. Whatever moves them publishes them while the trial runs:\n"
        )
        notes += "".join(f"#   {m}\n" for m in scene.movable)
    if contacts:
        notes += (
            "#\n# These objects TOUCH the robot at the posture the simulator starts in.\n"
            "# CheckStartStateCollision refuses a request whose start state is in collision, so\n"
            "# move_group will plan nothing at all until each pair is allowed, padded back, or the\n"
            "# object is left out:\n"
        )
        notes += "".join(
            f"#   {oid} <-> {link} ({dist * 1000:.1f} mm)\n" for oid, link, dist in contacts
        )

    return (
        "# GENERATED by `roqsim export moveit --scene` from the world the simulator loads. "
        "Do not edit.\n"
        "#\n"
        f"# {len(objects)} collision object(s) in {frame!r}, the frame move_group plans in. Each is a\n"
        "# static body of the compiled world, carrying the primitives it is built from, so a whole\n"
        "# prop can be allowed, padded or removed under one name.\n"
        "#\n"
        "# Apply it as a DIFF (moveit_msgs/ApplyPlanningScene, or a PlanningScene publisher): it adds\n"
        "# these objects and states nothing about the robot. A depth sensor's octomap is the other\n"
        "# route into the scene and answers a different question -- it reports what is in view,\n"
        "# whatever its shape, while these are named objects a trial can attach, allow or pad.\n"
        "#\n"
        "# SolidPrimitive type: 1 box (x, y, z), 2 sphere (radius), 3 cylinder (height, radius).\n"
        "# CollisionObject operation: 0 add.\n" + notes + yaml.safe_dump(body, sort_keys=False)
    )
