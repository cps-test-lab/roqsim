"""The ``vision_msgs/Detection2DArray`` converter: which field carries the class, which the instance.

Both exist in the message, and a producer that fills only one is the reason a tracker merges two
objects into one. The rule this pins: ``Detection2D.id`` is the INSTANCE, the hypothesis's
``class_id`` is the CLASS NAME. Two parcels in view are one class and two ids.
"""

from roqsim_ros_bridge.registry import get_converter, to_time_msg

#: (class_id, class_name, instance_id, cx, cy, w, h) -- what segmentation_camera puts on the endpoint.
TWO_PARCELS = [
    (1, "parcel", 12, 40.0, 30.0, 20.0, 16.0),
    (1, "parcel", 17, 100.0, 30.0, 20.0, 16.0),
]


def _fill(payload, hints=None):
    from vision_msgs.msg import Detection2DArray

    msg = Detection2DArray()
    get_converter("vision_msgs.msg.Detection2DArray")(msg, payload, to_time_msg(2.0), hints or {})
    return msg


def test_the_class_and_the_instance_land_in_their_own_fields():
    msg = _fill(TWO_PARCELS, {"frame_id": "camera_optical_frame"})
    assert [d.id for d in msg.detections] == ["12", "17"]
    assert {d.results[0].hypothesis.class_id for d in msg.detections} == {"parcel"}
    assert msg.header.frame_id == "camera_optical_frame"
    assert all(d.header.frame_id == "camera_optical_frame" for d in msg.detections)
    assert msg.header.stamp.sec == 2


def test_the_box_is_centre_and_size_in_pixels():
    det = _fill(TWO_PARCELS).detections[0]
    assert (det.bbox.center.position.x, det.bbox.center.position.y) == (40.0, 30.0)
    assert (det.bbox.size_x, det.bbox.size_y) == (20.0, 16.0)


def test_ground_truth_boxes_score_one():
    """A mask covers a pixel or it does not; a made-up confidence would be thresholdable noise."""
    assert _fill(TWO_PARCELS).detections[0].results[0].hypothesis.score == 1.0


def test_an_empty_frame_is_an_empty_array_not_a_missing_message():
    """Nothing in view is a result: a consumer must be able to tell it from a dead publisher."""
    assert _fill([]).detections == []


def test_the_frame_id_is_namespaced_like_every_other_endpoint():
    msg = _fill(TWO_PARCELS, {"frame_id": "camera_optical_frame", "frame_prefix": "robot_b"})
    assert msg.header.frame_id == "robot_b/camera_optical_frame"
