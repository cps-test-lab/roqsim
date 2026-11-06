# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""A renderer in another process stands behind the render service unchanged.

``render_stub_server.py`` serves roqsim's render socket from a process of its own and answers with
synthetic images; the world names the ``socket`` renderer, which starts it, sends it the scene
once and each tick's batch, and fails the tick loudly once it is gone.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim.config import load_config_from_dict
from roqsim.engine import Engine
from roqsim.plugin import Plugin, PluginError
from roqsim.presence import set_present
from roqsim.render_api import RenderError
from roqsim.render_socket import recv_message, send_message
from roqsim.rendering import FrameRenderer, camera_intrinsics

STUB = Path(__file__).with_name("render_stub_server.py")


class Cams(Plugin):
    """Two model cameras over a box, each registered with the render service as a view due every
    tick, the way a camera plugin registers one. With ``lens: true`` the front camera is a real
    lens: a ``sensorsize`` with its principal point off the image centre."""

    def build(self, spec, ctx):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.2])
        front = spec.worldbody.add_camera(name="front", pos=[1, 0, 0.3], xyaxes=[0, 1, 0, 0, 0, 1])
        if self.config.get("lens"):
            front.sensor_size = [0.0064, 0.0048]
            front.focal_length = [0.004, 0.006 if self.config["lens"] == "anisotropic" else 0.004]
            front.principal_length = [0.0004, -0.0002]
            front.resolution = [32, 24]
        spec.worldbody.add_camera(name="top", pos=[0, 0, 2])

    def configure(self, ctx):
        images = tuple(self.config.get("images", ("rgb", "depth")))
        self.views = [
            ctx.render.view(
                f"stub/{name}",
                camera=name,
                width=w,
                height=h,
                images=images,
                wants=lambda i=images: i,
                geomgroup=self.config.get("geomgroup"),
                bodies=self.config.get("bodies") or "all",
            )
            for name, (w, h) in (("front", (32, 24)), ("top", (16, 16)))
        ]


def _engine(
    tmp_path,
    images=("rgb", "depth"),
    settings=None,
    geomgroup=None,
    lens=False,
    address=None,
    bodies="all",
) -> Engine:
    address = address or f"unix:{tmp_path / 'render.sock'}"
    socket_settings = {
        "address": address,
        "command": [sys.executable, str(STUB), "{address}", str(tmp_path / "report.json")],
        "images": ["rgb", "depth"],
        "connect_timeout_s": 60,
        "timeout_s": 30,
        **(settings or {}),
    }
    cfg = load_config_from_dict(
        {
            "sim": {
                "timestep": 0.002,
                "renderer": "socket",
                "renderers": {"socket": socket_settings},
            },
            "components": [
                {
                    f"{__name__}:Cams": {
                        "images": list(images),
                        "geomgroup": geomgroup,
                        "lens": lens,
                        "bodies": bodies,
                    }
                }
            ],
        }
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    return engine


def test_a_world_draws_through_a_renderer_in_another_process(tmp_path):
    engine = _engine(tmp_path)
    engine.setup()
    engine.reset()
    try:
        for _ in range(3):
            engine.step()
        renderer = engine.ctx.render.renderer()
        assert renderer.capabilities_for(renderer.settings).in_process is False
        assert renderer._proc is not None and renderer._proc.poll() is None  # its own process
        front, top = (v.frame for v in engine.plugins[0].views)
        step = round(engine.ctx.sim_time / engine.ctx.dt)
        for frame, (w, h), view_id in (
            (front, (32, 24), "stub/front"),
            (top, (16, 16), "stub/top"),
        ):
            assert frame.sim_time == pytest.approx(engine.ctx.sim_time) and frame.step == step
            assert frame.rgb.shape == (h, w, 3) and frame.depth.shape == (h, w)
            assert (frame.rgb[..., 0] == step % 256).all()
            assert (frame.rgb[..., 1] == len(view_id)).all()
            assert (frame.rgb[0, :, 2] == np.arange(w)).all()
            assert frame.depth[0, 0] == pytest.approx(1.0 + 0.001 * step)
    finally:
        engine.shutdown()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["cameras"] == ["stub/front", "stub/top"]
    assert report["bodies"] == engine.ctx.model.nbody
    assert report["glb_magic"] == "glTF" and report["glb_bytes"] > 0  # the scene, once, as GLB
    # Each tick carries the poses and the presence the server draws.
    assert report["render_blobs"] == [
        "model/flex_group",
        "model/flex_rgba",
        "model/geom_group",
        "model/geom_rgba",
        "xpos",
        "xquat",
    ]


def test_a_renderer_process_that_died_fails_the_next_tick(tmp_path):
    engine = _engine(tmp_path)
    engine.setup()
    engine.reset()
    try:
        engine.step()
        proc = engine.ctx.render.renderer()._proc
        proc.kill()
        proc.wait()
        with pytest.raises(RenderError, match="failed during 'render'"):
            engine.step()
    finally:
        engine.shutdown()


def test_an_image_the_server_does_not_declare_is_refused_at_load(tmp_path):
    engine = _engine(tmp_path, images=("rgb", "segmentation"))
    with pytest.raises(PluginError, match="image 'segmentation'"):
        engine.setup()


def test_a_server_nobody_listens_at_is_said(tmp_path):
    engine = _engine(tmp_path, settings={"command": None, "connect_timeout_s": 0.3})
    engine.setup()
    engine.reset()
    try:
        with pytest.raises(RenderError, match="no renderer accepted a connection"):
            engine.step()
    finally:
        engine.shutdown()


def test_a_server_that_never_listens_is_stopped_with_the_failed_open(tmp_path):
    pid_file = tmp_path / "server.pid"
    never_listens = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    )
    engine = _engine(
        tmp_path,
        settings={"command": [sys.executable, "-c", never_listens], "connect_timeout_s": 2},
    )
    engine.setup()
    engine.reset()
    try:
        with pytest.raises(RenderError, match="no renderer accepted a connection"):
            engine.step()
    finally:
        engine.shutdown()
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):  # terminated and reaped, not left running
        os.kill(pid, 0)


#: A soft block, spawned as the entity ``prop``.
SOFT_BLOCK = """<mujoco><worldbody><body name="soft_block">
  <flexcomp name="soft" type="grid" count="3 3 3" spacing=".02 .02 .02" dim="3" radius=".002"
            mass="0.1" rgba="1 0 0 1"><edge equality="false"/>
    <elasticity young="5e4" poisson="0.3" damping="0.002"/></flexcomp>
</body></worldbody></mujoco>"""


def test_an_absent_flex_is_not_drawn_by_a_renderer_in_another_process(tmp_path):
    block = tmp_path / "soft_block.xml"
    block.write_text(SOFT_BLOCK, encoding="utf-8")
    address = f"unix:{tmp_path / 'render.sock'}"
    cfg = load_config_from_dict(
        {
            "sim": {
                "timestep": 0.002,
                "renderer": "socket",
                "renderers": {
                    "socket": {
                        "address": address,
                        "command": [
                            sys.executable,
                            str(STUB),
                            "{address}",
                            str(tmp_path / "r.json"),
                        ],
                        "images": ["rgb", "depth"],
                    }
                },
            },
            "components": [
                {"spawn_model": {"model": str(block), "prefix": "p_"}, "name": "prop"},
                {f"{__name__}:Cams": {}},
            ],
        },
        base_dir=tmp_path,
    )
    engine = Engine(cfg)
    engine.ctx.seed = 1
    engine.setup()
    engine.reset()
    try:
        engine.step()
        set_present(engine.ctx, engine.ctx.entities.get("prop"), False)
        engine.step()
    finally:
        engine.shutdown()
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["flexes_drawn"] == [1, 0]


def test_a_camera_drawing_some_geom_groups_is_refused_by_a_server_that_draws_every_group(tmp_path):
    engine = _engine(tmp_path, settings={}, geomgroup=[2])
    with pytest.raises(PluginError, match="only some geom groups"):
        engine.setup()


def test_a_server_that_declares_geom_groups_is_sent_the_cameras_groups(tmp_path):
    engine = _engine(tmp_path, settings={"geom_groups": True}, geomgroup=[2])
    engine.setup()
    engine.reset()
    try:
        engine.step()
    finally:
        engine.shutdown()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["geomgroups"] == [[2], [2]]


def test_a_camera_asking_for_the_moving_bodies_alone_is_refused_by_a_server_that_draws_all(
    tmp_path,
):
    engine = _engine(tmp_path, settings={}, bodies="moving")
    with pytest.raises(PluginError, match="the moving bodies alone"):
        engine.setup()


def test_a_server_that_declares_moving_only_is_sent_which_bodies_a_camera_draws(tmp_path):
    engine = _engine(tmp_path, settings={"moving_only": True}, bodies="moving")
    engine.setup()
    engine.reset()
    try:
        engine.step()
    finally:
        engine.shutdown()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["bodies_asked"] == ["moving", "moving"]


def test_its_settings_are_validated_at_load():
    with pytest.raises(PluginError, match="'address' is required"):
        load_config_from_dict(
            {"sim": {"renderer": "socket", "renderers": {"socket": {}}}, "components": []}
        )


def test_a_camera_s_intrinsics_reach_the_server(tmp_path):
    engine = _engine(tmp_path, settings={"principal_point": True}, lens=True)
    engine.setup()
    engine.reset()
    try:
        engine.step()
    finally:
        engine.shutdown()
    report = json.loads((tmp_path / "report.json").read_text())
    # 32x24 on a 6.4x4.8 mm sensor: 5 px/mm. The principal point is 0.4 mm right of and 0.2 mm
    # below the sensor's centre, so the axis is 2 px left of and 1 px below the image centre.
    fx, fy, cx, cy = report["intrinsics"]["stub/front"]
    assert (fx, fy) == pytest.approx((20.0, 20.0))
    assert (cx, cy) == pytest.approx((15.5 - 2.0, 11.5 + 1.0))
    top = report["intrinsics"]["stub/top"]  # a fovy camera: the symmetric pinhole
    assert top[2:] == pytest.approx([7.5, 7.5]) and top[0] == pytest.approx(top[1])
    assert report["request_intrinsics"] == report["intrinsics"]


def test_a_lens_camera_is_refused_by_a_server_that_draws_only_a_symmetric_pinhole(tmp_path):
    engine = _engine(tmp_path, lens=True)
    with pytest.raises(PluginError, match="off-centre principal point"):
        engine.setup()


def test_non_square_pixels_are_refused_by_a_server_that_draws_only_square_ones(tmp_path):
    engine = _engine(tmp_path, settings={"principal_point": True}, lens="anisotropic")
    with pytest.raises(PluginError, match=r"non-square pixels \(fx != fy\)"):
        engine.setup()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_tick_over_tcp_does_not_wait_for_a_delayed_acknowledgement(tmp_path):
    """A message is several writes; Nagle's algorithm would hold the last for an acknowledgement."""
    port = _free_port()
    engine = _engine(tmp_path, address=f"127.0.0.1:{port}")
    engine.setup()
    engine.reset()
    try:
        engine.step()  # opens the server
        times = []
        for _ in range(20):
            t = time.perf_counter()
            engine.step()
            times.append(time.perf_counter() - t)
    finally:
        engine.shutdown()
    assert sorted(times)[len(times) // 2] < 0.02


def test_a_server_restarted_on_the_port_of_one_that_just_ended_binds(tmp_path):
    port = _free_port()
    for run in range(2):
        engine = _engine(tmp_path / str(run), address=f"127.0.0.1:{port}")
        (tmp_path / str(run)).mkdir()
        engine.setup()
        engine.reset()
        try:
            engine.step()
        finally:
            engine.shutdown()


def test_an_async_worker_waits_for_the_renderer_s_own_open_timeout(tmp_path):
    engine = _engine(tmp_path, settings={"connect_timeout_s": 300, "timeout_s": 30})
    engine.setup()
    try:
        service = engine.ctx.render
        assert service.timeout_s == 60.0
        assert service.open_timeout_s("socket") == 330.0
        assert service.open_timeout_s("mujoco") == 60.0
    finally:
        engine.shutdown()


def test_a_server_slow_to_open_is_waited_for_open_timeout_s_not_a_tick_s(tmp_path):
    def slow(path, open_timeout_s):
        path.mkdir()
        command = [sys.executable, str(STUB), "{address}", str(path / "report.json"), "1.5"]
        return _engine(
            path,
            settings={"command": command, "timeout_s": 0.5, "open_timeout_s": open_timeout_s},
        )

    engine = slow(tmp_path / "waited", 10)
    engine.setup()
    engine.reset()
    try:
        engine.step()
        assert engine.plugins[0].views[0].frame is not None
    finally:
        engine.shutdown()
    engine = slow(tmp_path / "not_waited", None)
    engine.setup()
    engine.reset()
    try:
        with pytest.raises(RenderError, match="failed during 'open'"):
            engine.step()
    finally:
        engine.shutdown()


def _lens_model(anisotropic: bool = False) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.2])
    spec.worldbody.add_camera(name="plain", pos=[1, 0, 0.3], xyaxes=[0, 1, 0, 0, 0, 1])
    lens = spec.worldbody.add_camera(name="lens", pos=[1, 0, 0.3], xyaxes=[0, 1, 0, 0, 0, 1])
    lens.sensor_size = [0.0064, 0.0048]
    lens.focal_length = [0.004, 0.006 if anisotropic else 0.004]
    lens.principal_length = [0.0004, -0.0002]
    lens.resolution = [32, 24]
    return spec.compile()


@pytest.mark.parametrize(
    ("anisotropic", "settings", "refused"),
    [
        (False, {}, "off-centre principal point"),
        (True, {"principal_point": True}, r"non-square pixels \(fx != fy\)"),
    ],
)
def test_a_tool_s_frame_is_refused_what_the_server_does_not_draw(
    tmp_path, anisotropic, settings, refused
):
    """A tool's one view (`roqsim render`) never goes through the service's check at load."""
    model = _lens_model(anisotropic)
    renderer = FrameRenderer(
        model,
        32,
        24,
        camera="lens",
        renderer="socket",
        settings={
            "address": f"unix:{tmp_path / 'render.sock'}",
            "command": [sys.executable, str(STUB), "{address}"],
            "connect_timeout_s": 60,
            **settings,
        },
    )
    try:
        with pytest.raises(RenderError, match=refused):
            renderer.render(mujoco.MjData(model))
    finally:
        renderer.close()


def test_a_tool_s_view_carries_the_intrinsics_of_the_camera_it_moved_to():
    model = _lens_model()
    renderer = FrameRenderer(model, 32, 24, camera="plain")
    try:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        renderer.render(data)
        assert not renderer._view.spec.off_centre
        renderer.camera = "lens"
        renderer.render(data)
        spec = renderer._view.spec
        assert spec.off_centre
        lens = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "lens")
        assert spec.intrinsics == camera_intrinsics(model, lens, 32, 24, spec.fovy)
    finally:
        renderer.close()


def test_a_tool_s_view_gives_a_camera_without_a_lens_no_intrinsics():
    """As the service's own view does: an orthographic camera and a fovy of 0 project through fovy."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.2])
    spec.worldbody.add_camera(name="plain", pos=[1, 0, 0.3], xyaxes=[0, 1, 0, 0, 0, 1])
    ortho = spec.worldbody.add_camera(name="ortho", pos=[1, 0, 0.3], xyaxes=[0, 1, 0, 0, 0, 1])
    ortho.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
    ortho.fovy = 1.0
    zero = spec.worldbody.add_camera(name="zero", pos=[1, 0, 0.3], xyaxes=[0, 1, 0, 0, 0, 1])
    zero.fovy = 0.0
    renderer = FrameRenderer(spec.compile(), 32, 24, camera="plain")
    try:
        assert renderer._view_for("plain").spec.intrinsics is not None
        assert renderer._view_for("ortho").spec.intrinsics is None
        assert renderer._view_for("zero").spec.intrinsics is None
    finally:
        renderer.close()


def test_a_received_array_is_writable_memory_of_its_own():
    left, right = socket.socketpair()
    with left, right:
        send_message(left, {"op": "x"}, [("a", np.arange(6, dtype=np.uint8).reshape(2, 3))])
        _, blobs = recv_message(right)
    assert blobs["a"].flags.writeable and blobs["a"].flags.owndata
    assert blobs["a"].tolist() == [[0, 1, 2], [3, 4, 5]]


@pytest.mark.parametrize("dtype", ["|O", "O,i4"])
def test_a_blob_of_objects_is_refused_since_its_bytes_would_be_pointers(dtype):
    """A peer's bytes received into an object array are pointers it chose."""
    left, right = socket.socketpair()
    with left, right:
        body = json.dumps({"blobs": [{"name": "a", "dtype": dtype, "shape": [1]}]}).encode()
        left.sendall(len(body).to_bytes(4, "big") + body + b"A" * 8)
        with pytest.raises(ValueError, match="holds objects"):
            recv_message(right)
