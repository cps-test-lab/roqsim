Talking to a running simulation
===============================

``roqsim sim`` serves every endpoint of the world it runs -- every sensor reading, every command a
plugin takes, every stream a controller follows -- over a local control socket, together with
pause, resume, step and reset. No ROS is needed on either side. The socket is on by default and
needs the ``ipc`` extra (``pip install 'roqsim[ipc]'``, which is pyzmq).

Examples
--------

Start a simulation. It says where it can be reached::

   $ roqsim sim roqsim_mobile:husky_demo --headless
   control: ipc:///home/me/runs/roqsim-control.sock

From another shell, in the same directory or anywhere else on the machine::

   $ roqsim ls
   ipc:///home/me/runs/roqsim-control.sock  pid 41213  up 12 s  roqsim_mobile:husky_demo

   $ roqsim endpoints
   robot/diff_drive/cmd_vel      stream             Body-frame velocity command, applied once per step.
   robot/diff_drive/odom         out      50 Hz
   robot/lidar2d_0/lidar/scan    out      41.6667 Hz
   sim/run_control/pause         command            Stop stepping. Commands sent while paused still run ...
   sim/run_control/step          command            Take N steps (default 1) while paused ...
   ...

   $ roqsim describe robot/diff_drive/cmd_vel
   robot/diff_drive/cmd_vel
     kind: stream
     Body-frame velocity command, applied once per step.
     takes: Twist -- A body-frame velocity.
     parameter vx: float [m/s] -- forward speed
     parameter vy: float [m/s] -- sideways speed; a differential drive drops it = 0.0
     parameter wz: float [rad/s] -- yaw rate = 0.0
     on ros2: {"topic": "/cmd_vel", "type": "geometry_msgs.msg.Twist", "qos": {...}}
     example: roqsim call robot/diff_drive/cmd_vel '{"vx": 0.0, "vy": 0.0, "wz": 0.0}'

   $ roqsim describe robot/diff_drive/odom
   robot/diff_drive/odom
     kind: out, 50 Hz (from odom_rate_hz)
     Wheel odometry, integrated from the wheels' own motion.
     value: Odometry -- A pose in the odometry frame and the body-frame twist.
       position: array [m] -- body origin, odometry frame
       ...

   $ roqsim read robot/lidar2d_0/lidar/scan          # arrays summarised; --full prints them
   $ roqsim read sim/entities/robot/pose --field position
   $ roqsim read sim/run_control/state --field sim_time
   $ roqsim call robot/diff_drive/cmd_vel '{"vx": 0.5}'
   $ roqsim sub robot/diff_drive/odom --count 5      # values as they are published
   $ roqsim ctl pause
   $ roqsim ctl step 100                             # returns once the 100 steps ran
   $ roqsim ctl resume

A command that names a confirmation replies with it. ``model_override``'s ``override`` is confirmed
by its report, so a fault reports whether it landed::

   $ roqsim call grip_fault/override true
   {"applied": true, "result": null, "verified": true,
    "confirmation": {"active": true, "since": 4.2, "changes": 1, "verified": "landed"}}

From Python::

   from roqsim.control_client import Client

   with Client() as sim:
       sim.pause()
       print(sim.step(10)["sim_time"])
       scan = sim.read("robot/lidar2d_0/lidar/scan")    # ranges is a numpy array
       sim.call("robot/diff_drive/cmd_vel", {"vx": 0.3, "wz": 0.1})
       sim.resume()
       with sim.subscribe("robot/diff_drive/odom") as odom:
           path, t, value = odom.get(timeout=1.0)

A scenario reaches it too: ``osc.roqsim``'s actions, run against a simulator in another process,
talk to its control socket, and ``entity_call`` sends any command a world declares::

   entity_call(entity: 'grip_fault', command: 'override', value: 'true')

An MCP client reaches the same calls through ``roqsim mcp serve``: ``list_endpoints``,
``describe_endpoint``, ``read_endpoint``, ``call_endpoint``, ``pause``, ``resume`` and ``step``.

Finding the simulator
---------------------

``roqsim sim --control <uri>`` (or ``ROQSIM_CONTROL``) chooses the address; ``--control none``
serves nothing:

* ``ipc://<path>`` -- a Unix socket. The default is ``roqsim-control.sock`` in the run directory
  (``RUN_OUTPUT_DIR``, else ``OUTPUT_DIR``, else the working directory). A run directory too deep
  for a Unix socket path moves it to the runtime directory, with a warning naming both.
* ``tcp://[host]:<port>`` -- ``tcp://:5555`` binds ``127.0.0.1``. Naming another address opens the
  simulator's control to that network.

Subscriptions travel on a second socket derived from the first: ``<path>.pub``, or port + 1.

A client (the commands above, :class:`roqsim.control_client.Client`, the MCP tools) takes
``--control``/``control`` too, and without it looks in this order: ``ROQSIM_CONTROL``, the run
directory's ``roqsim-control.sock`` when a running simulator serves it, then the only simulator
running. Each running
simulator registers itself in a per-user runtime directory, which is what ``roqsim ls`` lists; with
several running and none named, a client refuses and lists them. A second simulator asked to serve
an address one is already serving is refused at start-up.

When pyzmq is not installed, ``roqsim sim`` runs without a control socket and says so once; asking
for one by name (``--control``, ``ROQSIM_CONTROL``) is then an error naming the extra.
``--no-communication`` does not remove it -- it strips the middleware an experiment publishes on,
and the control socket reaches only this process. The scenario-execution adapter never serves one:
a stepped run is in-process, and the scenario is its client.

Paths and kinds
---------------

An endpoint's path is the address of the plugin that registered it, with its dots as slashes, then
the endpoint's name: the ``lidar`` nested under ``robot`` publishing ``scan`` is ``robot/lidar/scan``;
a ``model_override`` named ``grip_fault`` at the top of a world is ``grip_fault/override``. Run
control is ``sim/run_control/{pause, resume, step, reset, state}``; every entity's ground-truth
pose is the core's ``sim/entities/<name>/pose``, and ``sim/entities/set_state`` and
``sim/entities/set_presence`` place an entity and make it present or absent (what a scenario's
``set_entity_state``, ``spawn_entity`` and ``delete_entity`` do). A navigator's route is
``<entity>/<navigator>/navigate_through_poses``, followed by ``route_status`` and stopped by
``cancel_route``. Two endpoints that would share a path are refused
when the simulation starts, naming both.

Every endpoint is served, carrying its payload as it is, unless it opts out with ``ipc=None`` on its
decorator (``backend={"ipc": None}`` on a hand-built one). Three kinds:

``out``
   ``read`` returns its current value; ``sub`` delivers it as it is published, at its rate.
``command``
   ``call`` writes it -- the parameters by name, as a JSON object, for an endpoint that declares
   them (``describe`` lists them with their types and units; a missing, unknown or mistyped one is
   refused before anything is queued, naming the nearest known name) -- and waits for the outcome:
   what the plugin's method returned, or the plugin's own exception text when it refused. A command that names an ``out`` endpoint that confirms it
   (``Endpoint.confirm``) replies with that endpoint's value as recorded in the step that applied
   the command. **While the simulation is paused no step runs**, so such a reply is ``"verified":
   false`` with a note saying so -- it neither steps the simulation nor reports the verdict from
   before the change. A command that gets no outcome within the timeout (5 s unless the request
   names one) is an error, never a success.
``stream``
   ``call`` checks the parameters the same way, puts them in the stream's slot and returns at
   once; the newest value is applied at the next step, or while paused at once.

``describe`` returns the whole tree, or one endpoint: its kind, rate, docstring, payload type,
parameters and result with their types and units, the attribute or config key its rate, its laziness
(not read while nobody subscribes) and its presence come from, what confirms it, and what the other
transports call it (the ROS topic, service or action the ROS bridge resolved, with its type and
QoS). An unknown path is refused with the nearest known path and its siblings.

Values
------

Messages are JSON; a numpy array travels as a raw binary frame beside it and arrives as an array of
the same dtype and shape. A dataclass or a named tuple arrives as an object of its fields. The
command-line tools summarise an array of more than 16 numbers (``--max-items``, ``--full``), the MCP
tools one of more than 64.

With the ROS bridge
-------------------

Both bridges can serve one world: ``roqsim sim --ros`` still serves the control socket. Commands from
either run in the order they arrive, on the physics thread. A stream written by both within a second
is logged once as a WARNING naming both transports; the latest value wins, so they overwrite each
other. An IPC subscription counts as a subscriber for a producer that renders only when someone is
listening (a camera under ROS); without the ROS bridge such a producer renders as it always does.

Performance
-----------

Nothing runs per physics step while nobody asks. A ``read`` or a ``call`` costs one command on the
physics thread when it arrives; requests are answered on background threads, never in the loop. The
publishing side reads and sends only the endpoints under a prefix some client subscribed to -- ZeroMQ
reports subscriptions to the simulator -- so with no subscriber the bridge's per-step work is two
empty checks. Each array is copied once, on the physics thread (a producer may reuse its buffer the
next step), and sent without a further copy.

Pausing resets the pacer: the first step after ``resume`` is paced from when it is taken, so a pause
is not counted as falling behind in the run's pacing report.
