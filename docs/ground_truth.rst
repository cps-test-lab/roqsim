Ground truth
============

The simulator knows every body's exact pose. An analysis reads it from the recording, a plugin or a
client from the core pose data while the run goes, and a ROS stack that expects simulator ground
truth on a topic gets it from ``pose_publisher``.

The recording
-------------

An analysis reads the true path from the run itself. ``roqsim sim --record`` keeps the complete
state, and its ``poses`` channel (:ref:`sim-poses`) has one entry per sample per named body: the
world pose as a quaternion and the world twist, on exact simulated time. A
stepped run publishes nothing, so for it this is the only pose series there is. Because it comes from
the solver rather than from a transport, it has no arrival-time jitter.

The core pose data
------------------

While a run is going, every entity whose body is in the model has an ``out`` endpoint
``sim/entities/<name>/pose`` (owner ``sim``, :mod:`roqsim.entity_pose`). It holds the body's world
position, its ``(w, x, y, z)`` orientation, and the linear and angular velocity of the body origin,
taken from ``data.xpos``, ``data.xquat`` and ``cvel``::

    from roqsim import entity_pose

    ep = ctx.interface.find(entity_pose.OWNER, entity_pose.endpoint_name("robot"))
    pose = ep.read()            # EntityPose, or None while the entity is deleted

It is computed only when it is read. It carries no backend hint, so no bridge publishes it unless
asked; a client reads it over the control socket (:doc:`control`).

Any other frame of an entity -- a body, a site, a declared or device frame -- is named by its path
(:ref:`paths`) and read with :func:`roqsim.frames.frame_pose`, in the world or relative to another
frame. It takes an entity's root from the pose endpoint above and the rest from the same physics
state, so both agree to the bit::

    from roqsim.frames import frame_pose

    mouse = frame_pose(ctx, "robot/mouse", relative_to="robot/base_link")   # a Transform

On a topic (``pose_publisher``)
-------------------------------

A ROS stack that reads ground truth off a topic, as a vendor simulator's adapter does, gets it from
``pose_publisher`` (:doc:`plugins`): the frames it names, each in the world or relative to another,
on one topic at one rate. The TurtleBot 4's manifest and the Create 3 world's dock carry one each for
the streams the Create 3 stack reads (:doc:`create3_stack`).

roqsim publishes no ``<model>_base_link_gt`` TF frame. A consumer that needs the truth reads the
recording.
