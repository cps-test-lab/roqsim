ground truth
============

The simulator knows every body's exact pose. Two mechanisms expose that truth to ROS, for two
different consumers.

Ground-truth topic namespace (``ros2_bridge`` ``gt:``)
------------------------------------------------------

The :doc:`ros2_bridge <interfaces>` can divert *output* topics under a ground-truth prefix so a
consumer can tell a true pose from real perception. Configure it on the bridge plugin::

    plugins:
      - ros2_bridge:
          gt:
            prefix: /gt              # published outputs move to /gt/... (e.g. /gt/tf)
            exempt: [odom, scan]     # ...except these, which mirror a real topic and stay canonical

The rule: **prefix** a pure ground-truth stream that has no real-sensor equivalent (an object's true
pose); **exempt** a stream that mirrors a real topic (robot telemetry, a sensor's own message), so it
keeps its canonical name. With no ``gt`` block every topic is canonical.

Ground-truth base pose (``ground_truth_pose`` plugin)
-----------------------------------------------------

``ground_truth_pose`` (in ``roqsim_sensors``) publishes a robot base body's *true* world pose as a TF
transform ``<frame_id> -> <child_frame>``, default ``map -> <model>_base_link_gt``. It reads
``data.xpos``/``data.xquat`` directly, so it is robot-family-agnostic (TurtleBot, Husky, Spot, a
humanoid — same plugin).

The frame is a **disconnected leaf**: it is deliberately *not* wired into the odometry/localization
chain. nav2 localizes with AMCL off the drifting wheel odometry (``diff_drive`` integrates wheel
velocities, so ``/odom`` and ``odom -> base_link`` drift like real hardware); ``map -> odom`` is
AMCL's correction. ``<model>_base_link_gt`` sits beside that tree purely so an evaluator can diff the
navigated path against the truth.

This mirrors the Gazebo navigation stack, where ``gazebo_tf_publisher`` republishes
``SceneBroadcaster`` poses as ``<robot>_base_link_gt`` on ``/tf``. Recording ``/tf`` against either
simulator therefore yields the same ground-truth frame and the same offline analysis applies
unchanged — which is what lets roqsim stand in for Gazebo.

Add it to a world (or a robot manifest)::

    plugins:
      - spawn_robot: {model: turtlebot4, name: robot}
      - ros2_bridge: {}
      - ground_truth_pose: {robot: robot}     # -> /tf: map -> turtlebot4_base_link_gt

Config keys: ``robot`` (entity to read, default ``robot``), ``body`` (base-body override; default the
entity's registered base body), ``frame_id`` (parent, default ``map``), ``child_frame`` (default
``<model>_base_link_gt``), ``rate_hz`` (default ``30``), and the standard ``topics:`` hardwire map
(``pose`` role; default relative ``tf`` → ``/tf``, or ``/gt/tf`` under the bridge ``gt`` prefix).
