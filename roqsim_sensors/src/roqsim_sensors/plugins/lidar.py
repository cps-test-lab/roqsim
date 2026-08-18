"""Sensor plugin: 2D lidar via batched ray-casting (``mj_multiRay``, no GL).

Ported from our earlier in-house nav prototype's ``Lidar``. Casts a horizontal fan from the robot's ``lidar`` site each
``post_step``, optionally applies sensor noise (Gaussian range noise and/or random point dropout),
and exposes the latest :class:`LaserScan` via a ``scan`` output endpoint for a transport plugin.

Config::

    lidar:
      robot: robot
      namespace: ""              # transport scope (default: inherited from spawn_robot's namespace)
      site: lidar
      frame_id: lidar            # ROS frame the scan is stamped in + child of the static mount TF;
                                 # defaults to `site`. Set it when the robot's real description names
                                 # the frame differently (turtlebot4: rplidar_link).
      rays: 360
      angle_min: 0.0
      angle_max: 6.283185307     # 2*pi
      range_min: 0.164
      max_range: 12.0
      rate_hz: 10.0              # scan rate: rays are cast (and published) at this rate, not every step
      exclude_body: base_link    # robot's own body, so rays skip the chassis
      range_stddev: 0.0          # optional Gaussian noise on finite ranges (0 = off)
      dropout_percent: 0.0       # optional: % of points dropped (-> no return) per scan (0..100)
      rate_hz: 10.0              # scan publication rate for the bridge (ray-cast runs every step)
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin
from roqsim.presence import visible_geomgroup_mask

from .payloads import LaserScan

#: Built once: the mask is a constant, and rebuilding it per scan would allocate on every tick
#: of every lidar in the world.
_VISIBLE_GROUPS = visible_geomgroup_mask()


class LidarPlugin(Plugin):
    parallel_safe = True  # post_step only reads data + writes its own scan buffer

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.robot = self.config.get("robot", "robot")
        self.site = self.config.get("site", "lidar")
        # ROS frame the scan is stamped in, and the child of the static mount TF (one value, so the
        # two cannot disagree). Defaults to the site the rays are actually cast from; a model whose
        # real description names the frame differently declares it in its manifest (e.g. the
        # TurtleBot 4's URDF calls it `rplidar_link`). It used to be hardwired to `rplidar_link` for
        # every robot, which published a Husky's scan in a TurtleBot's frame.
        self.frame_id = self.config.get("frame_id", self.site)
        self.num_rays = int(self.config.get("rays", 360))
        self.angle_min = float(self.config.get("angle_min", 0.0))
        self.angle_max = float(self.config.get("angle_max", 2.0 * math.pi))
        self.range_min = float(self.config.get("range_min", 0.164))
        self.range_max = float(self.config.get("max_range", 20.0))
        # Scan rate: rays are cast (and the scan published) at this rate. Ray-casting every physics
        # step (e.g. 500 Hz) is ~50x more mj_multiRay work than the consumer needs; gate it here.
        self.rate_hz = float(self.config.get("rate_hz", 10.0))
        self._last_cast = float("-inf")
        self.exclude_body = self.config.get("exclude_body", "base_link")
        self.range_stddev = float(self.config.get("range_stddev", 0.0))
        self.dropout_percent = float(self.config.get("dropout_percent", 0.0))
        # Scan publication rate (Hz) for the bridge; the ray-cast itself runs every post_step.
        self.rate_hz = float(self.config.get("rate_hz", 10.0))
        # Publish base_link -> scan-frame as a static TF (from the site, so it always matches where
        # the rays are cast). On by default; disable when an external robot_state_publisher owns it.
        self.emit_static_tf = bool(self.config.get("emit_static_tf", True))
        self._site_id = -1
        self._bodyexclude = -1
        self._local_dirs: np.ndarray | None = None
        self._geomid = self._dist = None
        self._scan: LaserScan | None = None  # latest scan, read by the scan endpoint

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if int(config.get("rays", 360)) <= 0:
            errors.append("'rays' must be > 0")
        if float(config.get("max_range", 20.0)) <= 0:
            errors.append("'max_range' must be > 0")
        if float(config.get("rate_hz", 10.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if float(config.get("range_stddev", 0.0)) < 0:
            errors.append("'range_stddev' must be >= 0")
        if not 0.0 <= float(config.get("dropout_percent", 0.0)) <= 100.0:
            errors.append("'dropout_percent' must be in [0, 100]")
        return errors

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        # Transport scope for the scan endpoint: own config wins, else inherited from the spawn.
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        m = ctx.model
        self._site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, prefix + self.site)
        if self._site_id < 0:
            raise RuntimeError(f"lidar: site {prefix + self.site!r} not found")
        self._bodyexclude = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, prefix + self.exclude_body
        )

        angles = self.angle_min + np.arange(self.num_rays) * (
            (self.angle_max - self.angle_min) / self.num_rays
        )
        self._angle_increment = (self.angle_max - self.angle_min) / self.num_rays
        self._local_dirs = np.stack(
            [np.cos(angles), np.sin(angles), np.zeros(self.num_rays)], axis=1
        )
        self._geomid = np.full(self.num_rays, -1, dtype=np.int32)
        self._dist = np.full(self.num_rays, -1.0, dtype=np.float64)

        ros2_hints = {
            "type": "sensor_msgs.msg.LaserScan",
            "topic": self.topic_override("scan") or "scan",
            "frame_id": self.frame_id,
        }
        if self.emit_static_tf:
            ros2_hints["static_tf"] = self._mount_tf(m, prefix)

        # Declare the scan as a backend-neutral output endpoint (no ROS import here). The bridge
        # resolves the type string and publishes at rate; ``namespace`` scopes topic and frames.
        ctx.interface.add(
            Endpoint(
                name="scan",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._scan,
                rate_hz=self.rate_hz,
                backend={"ros2": ros2_hints},
            )
        )

    def on_reset(self, ctx: SimContext) -> None:
        # sim_time restarts at 0 on reset; clear the gate so the first post-reset step casts again.
        self._last_cast = float("-inf")

    def _mount_tf(self, m, prefix: str) -> dict:
        """Static mount transform (base body -> lidar site) as plain numbers, for a bridge to publish.

        Computed from the model on a throwaway ``MjData`` at the reference pose. The base<-site
        transform is rigid, so it is independent of where the robot stands; deriving it from the same
        site the rays are cast from keeps the published frame consistent with the scan by construction.
        No ROS types here -- ``roqsim`` stays ROS-free; the bridge turns this into a TransformStamped.
        """
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        base_pos = d0.xpos[self._bodyexclude]
        base_mat = d0.xmat[self._bodyexclude].reshape(3, 3)
        site_pos = d0.site_xpos[self._site_id]
        site_mat = d0.site_xmat[self._site_id].reshape(3, 3)
        rel_pos = base_mat.T @ (site_pos - base_pos)
        rel_quat = np.zeros(4)
        mujoco.mju_mat2Quat(rel_quat, np.ascontiguousarray(base_mat.T @ site_mat).reshape(-1))
        return {
            "parent": self.exclude_body,  # bare "base_link"; the bridge applies any namespace prefix
            "translation": [float(v) for v in rel_pos],
            "rotation": [float(v) for v in rel_quat],  # (w, x, y, z)
        }

    def post_step(self, ctx: SimContext) -> None:
        # Cast at the scan rate, not every physics step; the endpoint reads the latest self._scan.
        if ctx.sim_time - self._last_cast < 1.0 / self.rate_hz:
            return
        self._last_cast = ctx.sim_time
        m, d = ctx.model, ctx.data
        origin = d.site_xpos[self._site_id].copy()
        rot = d.site_xmat[self._site_id].reshape(3, 3)
        world_dirs = np.ascontiguousarray((self._local_dirs @ rot.T).reshape(-1))
        mujoco.mj_multiRay(
            m,
            d,
            origin,
            world_dirs,
            # NOT None (= every group): an absent entity lives in a reserved group, and this
            # raycaster tests the real triangles while ignoring contype/conaffinity -- so
            # without the mask an obstacle nothing can collide with is still a lidar return,
            # which is precisely what a navigation stack reacts to.
            _VISIBLE_GROUPS,
            1,
            self._bodyexclude,
            self._geomid,
            self._dist,
            None,
            self.num_rays,
            self.range_max,
        )
        # mj_multiRay's cutoff is a culling hint, not a clamp -- it can still report hits beyond
        # it. Enforce max_range here: a beyond-range hit is no return (inf), like a miss (-1).
        ranges = np.where(
            (self._dist >= 0.0) & (self._dist <= self.range_max),
            np.maximum(self._dist, self.range_min),
            np.inf,
        )
        if self.range_stddev > 0.0 or self.dropout_percent > 0.0:
            # One generator per (sensor, step), not per draw: counter-based, so the same noise is
            # reproducible from a recording without replaying the run. Keyed on this plugin's own
            # name so two lidars on one robot get independent streams.
            rng = ctx.rng_for(self.name or "lidar")
            ranges = ranges.copy()
            if self.range_stddev > 0.0:
                finite = np.isfinite(ranges)
                ranges[finite] += rng.normal(0.0, self.range_stddev, size=int(finite.sum()))
            if self.dropout_percent > 0.0:
                # Randomly drop this percentage of points per scan (a dropped ray -> no return).
                n_drop = int(round(self.num_rays * self.dropout_percent / 100.0))
                if n_drop > 0:
                    drop = rng.choice(self.num_rays, size=n_drop, replace=False)
                    ranges[drop] = np.inf
        self._scan = LaserScan(
            ranges=np.asarray(ranges),
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            angle_increment=self._angle_increment,
            range_min=self.range_min,
            range_max=self.range_max,
        )
