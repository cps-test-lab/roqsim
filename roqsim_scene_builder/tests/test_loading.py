"""Loading + offscreen render: the tool loads the same targets as ``roqsim`` (model ref, MJCF)
and renders a frame through the shared FrameRenderer -- run headless with the egl backend."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen render needs a GL context; egl is headless


@pytest.fixture(scope="module")
def _egl_ok():
    """Skip these tests where no offscreen GL context can be created (e.g. no EGL in CI)."""
    import mujoco

    spec = mujoco.MjSpec.from_string(
        "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
    )
    model = spec.compile()
    try:
        r = mujoco.Renderer(model, 32, 32)
        r.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no offscreen GL context: {exc}")


def _render_one(target: str):
    from roqsim_scene_builder.scene_window import load_engine

    from roqsim import FrameRenderer

    engine, _view = load_engine(target)
    try:
        fr = FrameRenderer(engine.ctx.model, 64, 48)
        frame = fr.render(engine.ctx.data)
        fr.close()
        return frame
    finally:
        engine.shutdown()


def test_render_model_reference(_egl_ok):
    frame = _render_one("roqsim_assets:industrial_table")
    assert frame.shape == (48, 64, 3)
    assert frame.dtype.name == "uint8"


def test_a_world_whose_bridge_is_not_installed_still_loads(tmp_path):
    """A review is about geometry, so a `*_ros` world must open without ROS on the path.

    No GL needed: the point is that the *build* survives a plugin ref this environment cannot
    resolve (``ros2_bridge`` ships in a colcon package -- see roqsim.config.drop_transport_plugins).
    """
    from roqsim_scene_builder.scene_window import load_engine

    world = tmp_path / "w.yaml"
    world.write_text("plugins:\n  - ros2_bridge: {}\n  - sim_interfaces: {}\n")
    engine, _view = load_engine(str(world))
    try:
        assert engine.ctx.model is not None
    finally:
        engine.shutdown()


def test_render_mjcf(_egl_ok, tmp_path):
    xml = tmp_path / "scene.xml"
    xml.write_text(
        "<mujoco><worldbody>"
        "<light pos='0 0 3'/><geom type='plane' size='2 2 .1'/>"
        "<body pos='0 0 .5'><freejoint/><geom type='box' size='.2 .2 .2'/></body>"
        "</worldbody></mujoco>"
    )
    frame = _render_one(str(xml))
    assert frame.shape == (48, 64, 3)


def test_scene_window_selects_the_gl_backend_before_importing_mujoco():
    """Importing the window module must leave an OFFSCREEN backend bound, not glfw.

    MUJOCO_GL is read once, while ``import mujoco`` runs. This module imports mujoco directly rather
    than through roqsim, so it has to call ``select_offscreen_gl`` itself first -- and isort sorts
    first-party ``roqsim`` *below* third-party ``mujoco``, so the ordering is one re-sort away from
    breaking. When it broke, every review window died in ``check_gl_backend`` with "MuJoCo bound the
    glfw (on-screen) GL backend", which names neither this module nor the ordering.

    Run in a subprocess with MUJOCO_GL unset: the variable is bound at first import, so an in-process
    check would only observe whatever the test session already bound.
    """
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import roqsim_scene_builder.scene_window\n"
            "from roqsim.rendering import bound_gl_backend\n"
            "print(bound_gl_backend())",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert proc.stdout.strip().splitlines()[-1] != "glfw"
