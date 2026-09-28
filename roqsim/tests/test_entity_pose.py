"""Every entity has a core pose endpoint: ground truth, computed only when read, presence-aware."""

from __future__ import annotations

import mujoco
import numpy as np

from roqsim import entity_pose
from roqsim.bridge import BridgeBase
from roqsim.config import load_config_from_dict
from roqsim.context import Entity
from roqsim.engine import Engine
from roqsim.plugin import Plugin
from roqsim.plugins.dummy import DummyPlugin
from roqsim.presence import set_present


class Late(Plugin):
    """Registers an entity after the bridge bound, on the dummy's body."""

    def configure(self, ctx) -> None:
        ctx.entities.add(Entity(name="late", kind="object", body="box_box"))


class Bridge(BridgeBase):
    BACKEND = "test"

    def _make_output(self, ep, hints):
        return object()

    def _make_input(self, ep, hints, on_payload):
        pass


def _engine(*extra):
    cfg = load_config_from_dict({"sim": {}, "plugins": []})
    return Engine(cfg, plugins=[DummyPlugin({}, name="box"), *extra], preview=True)


def _pose(engine, name="box"):
    return engine.ctx.interface.find(entity_pose.OWNER, entity_pose.endpoint_name(name))


def test_the_pose_is_the_bodys_world_state():
    with _engine() as engine:
        for _ in range(50):
            engine.step()
        ep = _pose(engine)
        assert (ep.owner, ep.name, ep.direction) == ("sim", "entities/box/pose", "out")
        d, m = engine.ctx.data, engine.ctx.model
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box_box")
        pose = ep.read()
        assert np.array_equal(pose.position, d.xpos[bid])
        assert np.array_equal(pose.orientation, d.xquat[bid])
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
        assert np.array_equal(pose.linear_velocity, vel[3:])
        assert np.array_equal(pose.angular_velocity, vel[:3])
        # A copy: the next step does not change what a reader already holds.
        held = pose.position.copy()
        engine.step()
        assert np.array_equal(pose.position, held)


def test_the_result_type_is_described():
    with _engine() as engine:
        result = _pose(engine).result
        fields = {f.name: f.type for f in result.fields}
        assert result.name == "EntityPose"
        assert (fields["position"].unit, fields["position"].shape) == ("m", (3,))
        assert fields["orientation"].shape == (4,)
        assert fields["angular_velocity"].unit == "rad/s"


def test_nothing_is_computed_without_a_reader(monkeypatch):
    calls = []
    real = mujoco.mj_objectVelocity
    monkeypatch.setattr(
        entity_pose.mujoco, "mj_objectVelocity", lambda *a: calls.append(1) or real(*a)
    )
    with _engine() as engine:
        for _ in range(20):
            engine.step()
        assert calls == []
        _pose(engine).read()
        assert calls == [1]


def test_a_deleted_entity_reads_nothing_until_it_is_spawned_again():
    with _engine() as engine:
        entity = engine.ctx.entities.get("box")
        set_present(engine.ctx, entity, False)
        assert _pose(engine).read() is None
        set_present(engine.ctx, entity, True)
        assert _pose(engine).read() is not None


def test_an_entity_registered_after_the_bridge_still_gets_its_pose():
    with _engine(Bridge({}), Late({}, label="late")) as engine:
        assert _pose(engine, "late").read() is not None


def test_an_entity_without_a_body_in_the_model_has_no_pose():
    class Ghost(Plugin):
        def configure(self, ctx) -> None:
            ctx.entities.add(Entity(name="ghost", kind="object", body="no_such_body"))
            ctx.entities.add(Entity(name="idea", kind="object"))

    with _engine(Ghost({}, label="ghost")) as engine:
        assert _pose(engine, "ghost") is None and _pose(engine, "idea") is None


def test_each_entity_has_exactly_one():
    with _engine(Late({}, label="late")) as engine:
        poses = [e.name for e in engine.ctx.interface.all() if e.owner == entity_pose.OWNER]
        assert sorted(poses) == ["entities/box/pose", "entities/late/pose"]
