"""Sensor plugin: Livox Mid-360 3D lidar via batched ray-casting (``mj_multiRay``, no GL).

The Mid-360 is a 360deg (horizontal) x 59deg (vertical, -7deg..+52deg) dome-FoV 3D lidar that emits a
point cloud rather than a planar scan, so this is a sibling of the 2D :class:`~.lidar.LidarPlugin`
rather than a mode of it: it casts a spherical grid of rays each frame and exposes the hits as a
``cloud`` (``sensor_msgs/PointCloud2``) output endpoint for a transport plugin.

Fidelity note (a substrate artifact, not a spec of the device): the real Mid-360 sweeps a proprietary
*non-repetitive* rosette pattern that fills the FoV densely over time. ``mj_multiRay`` is a
deterministic caster, so this plugin samples a uniform spherical grid over the same FoV bounds each
frame instead. The default grid (360 x 56 = 20160 rays at 10 Hz ~= 200k points/s) matches the device's
point rate and FoV, not its instantaneous pattern; for algorithms that depend on the true scan
trajectory this is an approximation, and that is deliberate and documented here rather than hidden.

Datasheet (https://www.livoxtech.com/mid-360): range 0.1 m (blind zone) .. 40 m @ 10% reflectivity
(70 m @ 80%), range precision 1sigma ~= 2 cm @ 10 m, 905 nm, 10 Hz frame rate, ~200k points/s.
Range noise is off by default (package convention -- opt in via ``range_stddev: 0.02`` for the 2 cm
figure), matching how ``lidar`` treats noise.

Config::

    livox_mid360:
      robot: robot
      namespace: ""              # transport scope (default: inherited from spawn's namespace)
      site: lidar                # site the rays are cast from
      frame_id: livox_frame      # ROS frame the cloud is stamped in + child of the static mount TF;
                                 # defaults to `site`. Livox's own driver publishes `livox_frame`.
      horizontal_rays: 360       # azimuth samples across the 360deg FoV (wraps, so no duplicate at 2pi)
      vertical_rays: 56          # elevation samples across the vertical FoV (inclusive endpoints)
      h_fov_min: 0.0
      h_fov_max: 6.283185307     # 2*pi (full 360deg)
      v_fov_min: -0.122173048    # -7 deg
      v_fov_max: 0.907571211     # +52 deg
      range_min: 0.1             # blind zone; nearer returns are dropped
      max_range: 40.0
      rate_hz: 10.0              # cloud rate: rays are cast (and published) at this rate, not per step
      exclude_body: base_link    # robot's own body, so rays skip the chassis
      range_stddev: 0.0          # optional Gaussian range noise (0 = off); device 1sigma ~= 0.02
      dropout_percent: 0.0       # optional: % of points dropped (-> no return) per frame (0..100)
      emit_static_tf: true       # publish base_link -> frame_id as a static TF (from the site)
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from roqsim.context import Endpoint, SimContext
from roqsim.plugin import Plugin

from .payloads import PointCloud

# Vertical FoV of the Mid-360, in radians -- see the datasheet reference in the module docstring.
_V_FOV_MIN = math.radians(-7.0)
_V_FOV_MAX = math.radians(52.0)


class LivoxMid360Plugin(Plugin):
    parallel_safe = True  # post_step only reads data + writes its own cloud buffer

    #: Azimuth spans a full 360deg dome and wraps, so the last sample is one step short of ``h_fov_max``
    #: (no duplicate ray at 2*pi). A bounded, forward-facing FoV subclass (see
    #: :class:`~roqsim_sensors.plugins.seyond_robin_w1g.SeyondRobinW1GPlugin`) sets this ``False`` to
    #: use inclusive azimuth endpoints instead, like elevation.
    AZIMUTH_WRAPS = True
    #: Default cloud topic when the world declares no ``topics: {cloud: ...}`` override. Livox's own ROS
    #: driver publishes ``livox/lidar``.
    DEFAULT_TOPIC = "livox/lidar"
    #: Name used in this plugin's own error/log messages (so a subclass reports under its own name).
    PLUGIN_LABEL = "livox_mid360"

    def __init__(self, config=None, *, name=None):
        super().__init__(config, name=name)
        self.robot = self.config.get("robot", "robot")
        self.site = self.config.get("site", "lidar")
        # ROS frame the cloud is stamped in, and the child of the static mount TF (one value, so the
        # two cannot disagree). Defaults to the site the rays are cast from (see LidarPlugin for the
        # rationale); Livox's own ROS driver names it `livox_frame`, so a model declares that.
        self.frame_id = self.config.get("frame_id", self.site)
        self.h_rays = int(self.config.get("horizontal_rays", 360))
        self.v_rays = int(self.config.get("vertical_rays", 56))
        self.h_fov_min = float(self.config.get("h_fov_min", 0.0))
        self.h_fov_max = float(self.config.get("h_fov_max", 2.0 * math.pi))
        self.v_fov_min = float(self.config.get("v_fov_min", _V_FOV_MIN))
        self.v_fov_max = float(self.config.get("v_fov_max", _V_FOV_MAX))
        self.range_min = float(self.config.get("range_min", 0.1))
        self.range_max = float(self.config.get("max_range", 40.0))
        # Cloud rate: rays are cast (and the cloud published) at this rate. Casting ~20k rays every
        # physics step would be ~50x more mj_multiRay work than the consumer needs; gate it here.
        self.rate_hz = float(self.config.get("rate_hz", 10.0))
        self._last_cast = float("-inf")
        self.exclude_body = self.config.get("exclude_body", "base_link")
        self.range_stddev = float(self.config.get("range_stddev", 0.0))
        self.dropout_percent = float(self.config.get("dropout_percent", 0.0))
        # Publish base_link -> cloud-frame as a static TF (from the site, so it always matches where
        # the rays are cast). On by default; disable when an external robot_state_publisher owns it.
        self.emit_static_tf = bool(self.config.get("emit_static_tf", True))
        self._site_id = -1
        self._bodyexclude = -1
        self._local_dirs: np.ndarray | None = None  # (nray, 3) unit directions in the site frame
        self._geomid = self._dist = None
        self._cloud: PointCloud | None = None  # latest cloud, read by the cloud endpoint

    @property
    def num_rays(self) -> int:
        return self.h_rays * self.v_rays

    def validate_config(self, config: dict) -> list[str]:
        errors = self.validate_topics(config)
        if int(config.get("horizontal_rays", 360)) <= 0:
            errors.append("'horizontal_rays' must be > 0")
        if int(config.get("vertical_rays", 56)) <= 0:
            errors.append("'vertical_rays' must be > 0")
        if float(config.get("max_range", 40.0)) <= 0:
            errors.append("'max_range' must be > 0")
        if float(config.get("range_min", 0.1)) < 0:
            errors.append("'range_min' must be >= 0")
        if float(config.get("rate_hz", 10.0)) <= 0:
            errors.append("'rate_hz' must be > 0")
        if float(config.get("range_stddev", 0.0)) < 0:
            errors.append("'range_stddev' must be >= 0")
        if not 0.0 <= float(config.get("dropout_percent", 0.0)) <= 100.0:
            errors.append("'dropout_percent' must be in [0, 100]")
        if float(config.get("v_fov_min", _V_FOV_MIN)) > float(config.get("v_fov_max", _V_FOV_MAX)):
            errors.append("'v_fov_min' must be <= 'v_fov_max'")
        return errors

    def configure(self, ctx: SimContext) -> None:
        entity = ctx.entities.get(self.robot)
        prefix = entity.meta.get("prefix", "") if entity else ""
        # Transport scope for the cloud endpoint: own config wins, else inherited from the spawn.
        ns = self.config.get("namespace") or (entity.meta.get("namespace", "") if entity else "")
        m = ctx.model
        self._site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, prefix + self.site)
        if self._site_id < 0:
            raise RuntimeError(f"{self.PLUGIN_LABEL}: site {prefix + self.site!r} not found")
        self._bodyexclude = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, prefix + self.exclude_body
        )

        self._local_dirs = self._build_directions()
        self._geomid = np.full(self.num_rays, -1, dtype=np.int32)
        self._dist = np.full(self.num_rays, -1.0, dtype=np.float64)

        ros2_hints = {
            "type": "sensor_msgs.msg.PointCloud2",
            "topic": self.topic_override("cloud") or self.DEFAULT_TOPIC,
            "frame_id": self.frame_id,
        }
        if self.emit_static_tf:
            ros2_hints["static_tf"] = self._mount_tf(m, prefix)

        # Declare the cloud as a backend-neutral output endpoint (no ROS import here). The bridge
        # resolves the type string and publishes at rate; ``namespace`` scopes topic and frames.
        ctx.interface.add(
            Endpoint(
                name="cloud",
                direction="out",
                owner=self.robot,
                namespace=ns,
                read=lambda: self._cloud,
                rate_hz=self.rate_hz,
                backend={"ros2": ros2_hints},
            )
        )

    def _build_directions(self) -> np.ndarray:
        """Unit ray directions in the site frame: a spherical grid over the (azimuth, elevation) FoV.

        For the full 360deg dome (``AZIMUTH_WRAPS``) azimuth wraps (like the 2D lidar), so the last
        sample is one step short of ``h_fov_max`` -- no duplicate ray at 2*pi. For a bounded FoV
        (a forward-facing subclass) azimuth uses inclusive endpoints, like elevation. Elevation is
        always a bounded band, so its endpoints are inclusive.
        """
        if self.AZIMUTH_WRAPS:
            az = self.h_fov_min + np.arange(self.h_rays) * (
                (self.h_fov_max - self.h_fov_min) / self.h_rays
            )
        elif self.h_rays == 1:
            az = np.array([(self.h_fov_min + self.h_fov_max) * 0.5])
        else:
            az = np.linspace(self.h_fov_min, self.h_fov_max, self.h_rays)
        if self.v_rays == 1:
            el = np.array([(self.v_fov_min + self.v_fov_max) * 0.5])
        else:
            el = np.linspace(self.v_fov_min, self.v_fov_max, self.v_rays)
        EL, AZ = np.meshgrid(el, az, indexing="ij")  # (v_rays, h_rays)
        cos_el = np.cos(EL)
        dirs = np.stack([cos_el * np.cos(AZ), cos_el * np.sin(AZ), np.sin(EL)], axis=-1).reshape(
            -1, 3
        )
        return np.ascontiguousarray(dirs)

    def on_reset(self, ctx: SimContext) -> None:
        # sim_time restarts at 0 on reset; clear the gate so the first post-reset step casts again.
        self._last_cast = float("-inf")

    def _mount_tf(self, m, prefix: str) -> dict:
        """Static mount transform (base body -> site) as plain numbers, for a bridge to publish.

        Identical rationale to LidarPlugin._mount_tf: computed on a throwaway ``MjData`` at the
        reference pose, from the same site the rays are cast from, so the published frame stays
        consistent with the cloud by construction. No ROS types here -- ``roqsim`` stays ROS-free.
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
            "parent": self.exclude_body,  # the bridge applies any namespace prefix
            "translation": [float(v) for v in rel_pos],
            "rotation": [float(v) for v in rel_quat],  # (w, x, y, z)
        }

    def post_step(self, ctx: SimContext) -> None:
        # Cast at the cloud rate, not every physics step; the endpoint reads the latest self._cloud.
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
            None,
            1,
            self._bodyexclude,
            self._geomid,
            self._dist,
            None,
            self.num_rays,
            self.range_max,
        )
        dist = self._dist
        hit = dist >= self.range_min  # a miss is -1; a return inside the blind zone is invalid
        if self.range_stddev > 0.0 or self.dropout_percent > 0.0:
            # One generator per (sensor, step), not per draw: counter-based, so the same noise is
            # reproducible from a recording without replaying the run. Keyed on this plugin's own
            # name so two lidars on one robot get independent streams.
            rng = ctx.rng_for(self.name or "livox")
            dist = dist.copy()
            if self.range_stddev > 0.0:
                dist[hit] += rng.normal(0.0, self.range_stddev, size=int(hit.sum()))
            if self.dropout_percent > 0.0:
                # Randomly drop this percentage of the potential returns per frame.
                n_drop = int(round(self.num_rays * self.dropout_percent / 100.0))
                if n_drop > 0:
                    drop = rng.choice(self.num_rays, size=n_drop, replace=False)
                    hit = hit.copy()
                    hit[drop] = False
        # Points in the sensor frame: direction * range for each valid return. Frame-independent, so
        # the cloud needs no world transform -- the static TF places the sensor frame in the tree.
        points = self._local_dirs[hit] * dist[hit, None]
        self._cloud = PointCloud(points=np.ascontiguousarray(points, dtype=np.float32))
