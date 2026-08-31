"""``segmentation_camera``: what the label image says, and what the boxes measure.

A top-down camera over a small scene, so every assertion is about a mask whose expected shape is
obvious: two parcels side by side on the floor, a two-link "robot" beside them, and a plate hanging
over a third parcel that is therefore invisible from above.

The load-bearing checks are the ones a mask metric would silently get wrong: an unlabelled body must
stay background (0), a robot's links must be ONE instance rather than one per link, a box must
measure the VISIBLE extent, and an entity made absent must contribute no pixels -- absence is a geom
group and an alpha, and an id pass does not have to respect either.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from roqsim_sensors.plugins.segmentation_camera import SegmentationCameraPlugin

from roqsim.config import load_config_from_dict
from roqsim.context import Entity, SimContext
from roqsim.plugin import Plugin
from roqsim.presence import set_present

#: Big enough that a parcel is hundreds of pixels, small enough to render fast in a test.
WIDTH, HEIGHT = 160, 120


class _Scene(Plugin):
    """Two parcels, a hidden third under a plate, a two-link robot, and a camera looking down."""

    provides_entity = True

    def build(self, spec: mujoco.MjSpec, ctx: SimContext) -> None:
        # Straight down: a MuJoCo camera looks along its own -z, so the default orientation at
        # height is a plan view and the mask's geometry is the scene's own layout.
        cam = spec.worldbody.add_camera(name="camera", pos=[0.0, 0.0, 3.0])
        cam.resolution = [WIDTH, HEIGHT]

        for i, x in enumerate((-0.6, 0.6)):
            body = spec.worldbody.add_body(name=f"parcel_{i}", pos=[x, 0.0, 0.1])
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.1], mass=1.0)

        # A third parcel with a plate above it: visible to nothing looking down.
        hidden = spec.worldbody.add_body(name="parcel_hidden", pos=[0.0, 1.2, 0.1])
        hidden.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1], mass=1.0)
        plate = spec.worldbody.add_body(name="plate", pos=[0.0, 1.2, 0.5])
        plate.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.3, 0.02], mass=1.0)

        # The "robot": a base with a child link, to check that a subtree is one instance.
        base = spec.worldbody.add_body(name="base_link", pos=[0.0, -1.2, 0.15])
        base.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.15, 0.1], mass=5.0)
        wheel = base.add_body(name="wheel", pos=[0.25, 0.0, -0.05])
        wheel.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=1.0)

    def configure(self, ctx: SimContext) -> None:
        ctx.entities.add(
            Entity(
                name=self.name, kind="robot", body="base_link", meta={"prefix": "", "namespace": ""}
            )
        )


PARCELS = {"class_id": 1, "name": "parcel", "bodies": ["parcel_*"]}
ROBOT = {"class_id": 2, "name": "robot", "entities": ["robot"]}


def _engine(*, classes=None, capture=True, **config):
    """An engine with the segmentation camera nested under the scene entity, captured once."""
    from roqsim.engine import Engine

    entry = {"segmentation_camera": {"classes": classes or [PARCELS, ROBOT], **config}}
    cfg = load_config_from_dict(
        {
            "sim": {},
            "components": [{f"{__name__}:_Scene": {}, "name": "robot", "components": [entry]}],
        }
    )
    engine = Engine(cfg)
    engine.setup()
    engine.reset()
    if capture:
        engine.step()
    return engine


def _recapture(engine) -> np.ndarray:
    """Step until the camera's rate gate opens again, and return the new class image.

    One step is 2 ms of sim time against a 10 Hz camera, so a single step after a change leaves the
    PREVIOUS frame in place -- which reads exactly like a change that did not take effect.
    """
    plugin = _plugin(engine)
    for _ in range(int(1.0 / plugin.rate_hz / engine.ctx.model.opt.timestep) + 1):
        engine.step()
    return plugin._labels


def _plugin(engine) -> SegmentationCameraPlugin:
    return next(p for p in engine.plugins if isinstance(p, SegmentationCameraPlugin))


def _endpoint(engine, name):
    return next((e for e in engine.ctx.interface.all() if e.name == name), None)


def _body_id(engine, name: str) -> int:
    return mujoco.mj_name2id(engine.ctx.model, mujoco.mjtObj.mjOBJ_BODY, name)


# -- the class image -------------------------------------------------------------------------


def test_the_class_image_labels_only_what_the_world_declared():
    labels = _plugin(_engine())._labels
    assert labels.shape == (HEIGHT, WIDTH) and labels.dtype == np.uint8
    # The floor and the walls of empty_room are in view and were never declared, so they are
    # background -- a mask metric that scored them would score the room.
    assert set(np.unique(labels).tolist()) == {0, 1, 2}
    assert (labels == 1).sum() > 100, "the two parcels should cover a good part of a plan view"
    assert (labels == 2).sum() > 100


def test_an_undeclared_body_stays_background():
    """The plate is in view and unlabelled: 'unlabelled' and 'absent' must look the same here."""
    labels = _plugin(_engine(classes=[PARCELS]))._labels
    assert set(np.unique(labels).tolist()) == {0, 1}


def test_the_first_matching_class_wins():
    """A specific body above a glob keeps its own class -- the reason matching is ordered."""
    classes = [
        {"class_id": 7, "name": "special", "bodies": ["parcel_0"]},
        {"class_id": 1, "name": "parcel", "bodies": ["parcel_*"]},
    ]
    labels = _plugin(_engine(classes=classes))._labels
    assert 7 in np.unique(labels).tolist() and 1 in np.unique(labels).tolist()


def test_a_pattern_that_matches_nothing_is_an_error():
    """Silently it would be an all-background class, i.e. 'the detector never saw it'."""
    with pytest.raises(RuntimeError, match="matches no body"):
        _engine(classes=[{"class_id": 1, "name": "nope", "bodies": ["not_here_*"]}])


# -- instances -------------------------------------------------------------------------------


def test_instance_ids_are_body_ids_and_a_subtree_is_one_instance():
    engine = _engine(instances=True)
    instances = _plugin(engine)._instance_img
    present = set(instances[instances != 0].tolist())
    assert present == {
        _body_id(engine, "parcel_0"),
        _body_id(engine, "parcel_1"),
        _body_id(engine, "base_link"),
    }
    # The wheel is a link of the robot, not an object: its pixels carry the base's id.
    assert _body_id(engine, "wheel") not in present


def test_the_instance_image_is_only_published_when_asked_for():
    assert _endpoint(_engine(capture=False), "instances") is None
    ep = _endpoint(_engine(instances=True, capture=False), "instances")
    assert ep is not None and ep.backend["ros2"]["encoding"] == "16UC1"


# -- boxes -----------------------------------------------------------------------------------


def test_boxes_are_measured_from_the_mask():
    engine = _engine(instances=True)
    plugin = _plugin(engine)
    boxes = {instance: (cx, cy, w, h) for _, _, instance, cx, cy, w, h in plugin._boxes}
    instances = plugin._instance_img
    for instance, (cx, cy, w, h) in boxes.items():
        mask = instances == instance
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        assert (w, h) == (cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1)
        assert (cx, cy) == ((cols[0] + cols[-1]) / 2.0, (rows[0] + rows[-1]) / 2.0)
    # Two equal parcels at mirrored positions: same size, centres either side of the image centre.
    p0, p1 = boxes[_body_id(engine, "parcel_0")], boxes[_body_id(engine, "parcel_1")]
    assert p0[2:] == p1[2:]
    assert (p0[0] - WIDTH / 2) == pytest.approx(-(p1[0] - WIDTH / 2), abs=1.0)


def test_an_occluded_object_is_not_detected():
    """The visible extent is the only one a detector could have produced (see the docstring)."""
    engine = _engine()
    detected = {instance for _, _, instance, *_ in _plugin(engine)._boxes}
    assert _body_id(engine, "parcel_hidden") not in detected
    assert _body_id(engine, "parcel_0") in detected


def test_min_pixels_drops_an_instance_that_is_barely_in_frame():
    engine = _engine(instances=True, min_pixels=10**6)
    assert _plugin(engine)._boxes == []
    # The image itself is unaffected: the threshold is about what counts as a detection.
    assert (_plugin(engine)._instance_img != 0).any()


def test_the_detections_carry_the_class_name_and_the_instance():
    engine = _engine()
    by_instance = {instance: (cid, name) for cid, name, instance, *_ in _plugin(engine)._boxes}
    assert by_instance[_body_id(engine, "parcel_0")] == (1, "parcel")
    assert by_instance[_body_id(engine, "base_link")] == (2, "robot")


# -- absence ---------------------------------------------------------------------------------


def test_an_absent_entity_contributes_no_label_pixels():
    """`DeleteEntity` hides geoms by group and alpha; an id pass respects neither for free."""
    engine = _engine()
    ctx = engine.ctx
    robot = ctx.entities.get("robot")
    assert (_plugin(engine)._labels == 2).sum() > 0
    assert set_present(ctx, robot, False) is True
    labels = _recapture(engine)
    assert (labels == 2).sum() == 0, "a deleted robot must not be in the label image"
    # And the parcels are untouched -- absence is per entity, not a global mask.
    assert (labels == 1).sum() > 100
    assert set_present(ctx, robot, True) is True
    assert (_recapture(engine) == 2).sum() > 0


# -- wiring ----------------------------------------------------------------------------------


def test_the_colour_stream_is_off_by_default_and_can_be_turned_on():
    """A label camera shares its MJCF camera with the RGB sensor that already publishes those pixels."""
    engine = _engine(capture=False)
    assert _endpoint(engine, "image") is None
    # camera_info is still published: it describes the label image too.
    assert _endpoint(engine, "camera_info") is not None
    with_color = _engine(color=True, capture=False)
    assert _endpoint(with_color, "image") is not None


def test_the_label_endpoint_gates_the_renderer():
    """A consumer that wants only labels must still get frames (camera_common's depth lesson)."""
    engine = _engine(capture=False)
    gated = {ep.name for ep in _plugin(engine)._gate_endpoints()}
    assert "labels" in gated and "detections" in gated


def test_the_endpoints_advertise_the_topics_and_types_a_consumer_expects():
    engine = _engine(instances=True, capture=False)
    hints = {name: _endpoint(engine, name).backend["ros2"] for name in ("labels", "detections")}
    assert hints["labels"]["type"] == "sensor_msgs.msg.Image"
    assert hints["labels"]["topic"] == "segmentation/class_image"
    assert hints["labels"]["encoding"] == "mono8"
    assert hints["detections"]["type"] == "vision_msgs.msg.Detection2DArray"
    assert hints["detections"]["topic"] == "segmentation/detections"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "'classes' is required"),
        ({"classes": [{"name": "x", "bodies": ["a"]}]}, "needs a 'class_id'"),
        ({"classes": [{"class_id": 0, "bodies": ["a"]}]}, "0 is reserved"),
        ({"classes": [{"class_id": 300, "bodies": ["a"]}]}, "must be 1..255"),
        (
            {"classes": [{"class_id": 1, "bodies": ["a"]}, {"class_id": 1, "bodies": ["b"]}]},
            "already declared",
        ),
        ({"classes": [{"class_id": 1, "name": "x"}]}, "must name 'bodies'"),
        ({"classes": [{"class_id": 1, "bodies": "a"}]}, "must be a list"),
        ({"classes": [PARCELS], "min_pixels": 0}, "'min_pixels' must be >= 1"),
    ],
)
def test_config_errors_are_reported_by_name(config, expected):
    errors = SegmentationCameraPlugin(config, entity="robot", label="seg").validate_config(config)
    assert any(expected in e for e in errors), errors
