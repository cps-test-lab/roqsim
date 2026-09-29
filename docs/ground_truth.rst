Ground truth
============

The simulator knows every body's exact pose. There are two ways to get it, and neither is a
plugin in the world.

The recording
-------------

An analysis reads the true path from the run itself. ``roqsim sim --record`` keeps the complete
state, and the pose series beside it (``sim_poses.csv``, :ref:`sim-poses`) has one row per sample
per named body: the world pose as a quaternion and the world twist, on exact simulated time. A
stepped run publishes nothing, so for it this is the only pose series there is. Because it comes from
the solver rather than from a transport, it has no arrival-time jitter.

The core pose endpoint
----------------------

While a run is going, every entity whose body is in the model has an ``out`` endpoint
``sim/entities/<name>/pose`` (owner ``sim``, :mod:`roqsim.entity_pose`). It holds the body's world
position, its ``(w, x, y, z)`` orientation, and the linear and angular velocity of the body origin,
taken from ``data.xpos``, ``data.xquat`` and ``cvel``::

    from roqsim import entity_pose

    ep = ctx.interface.find(entity_pose.OWNER, entity_pose.endpoint_name("robot"))
    pose = ep.read()            # EntityPose, or None while the entity is deleted

It is computed only when it is read. It carries no backend hint, so no bridge publishes it unless
asked. A plugin that must put a true pose on the wire in a stack's own shape reads this endpoint and
declares that stream itself. ``create3_pose_publisher`` (``roqsim_mobile``) is the example: it
publishes the TurtleBot 4's and its dock's poses, and their sites' poses relative to the body, on the
topics the Create 3 simulator stack reads (:doc:`create3_stack`).

roqsim publishes no ``<model>_base_link_gt`` TF frame. A consumer that needs the truth reads the
recording.
