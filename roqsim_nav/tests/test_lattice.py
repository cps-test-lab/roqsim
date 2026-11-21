"""The footprint planner: a way over position and heading on which every pose of a body and what it
holds clears the world, in height layers.

Each case a disc's way gets wrong is a test world in ``data/``:

* ``carry_post_straight``: a post beside a straight stretch, where a disc passes and a load carried
  across the body does not;
* ``carry_post_behind``: a dead end the body cannot turn round in, so it steps back; and a post
  where the turn after a short step back sweeps what it holds;
* ``carry_three_bends``: three bends of 30 degrees close together;
* ``carry_shelf_over_bench`` / ``_shelf``: a load wider than the passage passes over a bench beside
  it, and not under a shelf board at its own height.

In each, the disc's way walked as a body walks it -- facing along each leg, turning on the spot at
its corners -- meets something; the planner's way, checked pose by pose with the exact footprint,
meets nothing, or the planner says there is no way and names what is in it.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from roqsim_nav import lattice as L
from roqsim_nav.footprint import Footprint, Piece, capsule_piece, geom_piece
from roqsim_nav.lattice import NoWay, check_way, plan_way
from roqsim_nav.obstacles import wall_polygons

DATA = Path(__file__).parent / "data"
#: The most one of these cases may take to plan (s); each takes well under half of it.
PLAN_SECONDS = 2.0


def _ellipse(a, b, n=12):
    t = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.stack([a * np.cos(t), b * np.sin(t)], 1)


def _box(cx, cy, sx, sy):
    return np.array(
        [
            [cx - sx / 2, cy - sy / 2],
            [cx + sx / 2, cy - sy / 2],
            [cx + sx / 2, cy + sy / 2],
            [cx - sx / 2, cy + sy / 2],
        ]
    )


def _footprint(width=0.6, margin=0.0, one_band=False):
    """A body (legs to the knee, the trunk above them) and a tote carried in front of it across the
    body, its long side ``width``, at 0.85-1.05 m."""
    legs = (0.0, 0.5) if not one_band else (0.0, 1.8)
    trunk = (0.5, 1.6) if not one_band else (0.0, 1.8)
    load = (0.85, 1.05) if not one_band else (0.0, 1.8)
    return Footprint(
        [
            Piece(_ellipse(0.12, 0.15), legs, "legs"),
            Piece(_ellipse(0.12, 0.2), trunk, "trunk"),
            Piece(_box(0.42, 0.0, 0.4, width), load, "tote"),
        ],
        margin=margin,
    )


def _world(name):
    model = mujoco.MjModel.from_xml_path(str(DATA / f"{name}.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _disc_walked(model, data, fp, start, goal):
    """The disc's way as a body walks it: facing along each leg, turning on the spot at each
    corner and at the end to the goal's heading."""
    bounds = L._bounds(model, data, [start, goal], fp.radius)
    layers = L.Layers(model, data, fp.bands, bounds, fp.resolution)
    way = L._disc_way(layers, start, goal, 2)
    assert way is not None, "the disc has a way in every one of these worlds"
    segs, th = [], start[2]
    for (x0, y0), (x1, y1) in zip(way[:-1], way[1:], strict=False):
        if math.hypot(x1 - x0, y1 - y0) < 1e-6:
            continue
        h = math.atan2(y1 - y0, x1 - x0)
        segs.append(L.Segment("turn", [(x0, y0, th), (x0, y0, h)], angle=L._wrap(h - th)))
        segs.append(L.Segment("walk", [(x0, y0, h), (x1, y1, h)]))
        th = h
    segs.append(L.Segment("turn", [(goal[0], goal[1], th), goal], angle=L._wrap(goal[2] - th)))
    return L.Way(segs, 0.0, 0, 0.0, 0.0, False, [start, goal])


def _met(hits):
    return sorted({n for h in hits for n in h[3]})


CASES = {
    "carry_post_straight": ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), 0.6, ["post"]),
    "carry_post_behind": ((0.0, 0.0, 0.0), (-1.8, 1.2, math.pi), 0.6, ["side_s"]),
    "carry_three_bends": ((-1.0, 0.0, 0.0), (5.5, 0.92, 0.0), 0.6, ["wall_l0", "wall_r4"]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_a_case_the_disc_way_walked_meets_and_the_planned_way_clears(name):
    start, goal, width, disc_meets = CASES[name]
    model, data = _world(name)
    fp = _footprint(width)
    met = _met(check_way(model, data, fp, _disc_walked(model, data, fp, start, goal)))
    assert all(n in met for n in disc_meets), met  # the case is real: the disc's way meets them
    way = plan_way(model, data, fp, start, goal, back_max=0.5)
    assert check_way(model, data, fp, way) == []
    assert way.seconds < PLAN_SECONDS, way.summary()
    end = way.poses[-1]
    assert math.hypot(end[0] - goal[0], end[1] - goal[1]) < 0.04
    assert abs(L._wrap(end[2] - goal[2])) < math.radians(3)


def test_a_post_on_a_straight_stretch_is_walked_round_not_past():
    """The way along the straight stretch keeps what is carried off the post: it leaves the disc's
    straight line where the post stands."""
    model, data = _world("carry_post_straight")
    way = plan_way(model, data, _footprint(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    ys = [p[1] for p in way.poses if 1.5 < p[0] < 2.5]
    assert ys and min(ys) < -0.05, ys  # away from the post at y = +0.33


def test_a_dead_end_is_left_by_stepping_back_no_further_than_back_max_at_a_time():
    model, data = _world("carry_post_behind")
    fp = _footprint()
    start, goal = (0.0, 0.0, 0.0), (-1.8, 1.2, math.pi)
    way = plan_way(model, data, fp, start, goal, back_max=0.5)
    backs = [s for s in way.segments if s.kind == "back"]
    assert backs and all(0.0 < s.distance <= 0.5 + 1e-9 for s in backs), way.segments
    # two steps back always have a move forward between them
    kinds = [s.kind for s in way.segments]
    assert all(not (a == "back" and b == "back") for a, b in zip(kinds, kinds[1:], strict=False)), (
        kinds
    )
    # the first move is a step back: in the dead end nothing else is clear
    assert way.segments[0].kind == "back"
    assert check_way(model, data, fp, way) == []
    assert way.seconds < PLAN_SECONDS


def test_a_dead_end_without_a_step_back_has_no_way_and_says_what_is_in_it():
    model, data = _world("carry_post_behind")
    with pytest.raises(NoWay) as err:
        plan_way(
            model,
            data,
            _footprint(),
            (0.0, 0.0, 0.0),
            (-1.8, 1.2, math.pi),
            back_max=0.0,
            what="tote_1",
            to="the bench",
        )
    msg = str(err.value)
    assert msg.startswith("no way holding tote_1 to the bench:"), msg
    assert "side_n" in msg or "side_s" in msg or "end_wall" in msg, msg
    assert "back_max 0.00 m, long_way false" in msg, msg


def test_a_body_that_holds_nothing_is_refused_without_naming_a_load():
    """A way for a body that holds nothing (``what`` empty) is refused as "no way to ...", not as
    holding a nameless something."""
    model, data = _world("carry_post_behind")
    with pytest.raises(NoWay) as err:
        plan_way(
            model,
            data,
            _footprint(),
            (0.0, 0.0, 0.0),
            (-1.8, 1.2, math.pi),
            back_max=0.0,
            what="",
            to="the bench",
        )
    assert str(err.value).startswith("no way to the bench:"), str(err.value)


def test_a_turn_past_half_a_turn_is_taken_only_with_long_way():
    """A post on the short way round, and walls just outside what the tote sweeps turning on the
    spot, so that it cannot shuffle round in a few turns: the only way to face three quarters of a
    turn to the left is the long way round, taken only with ``long_way``."""
    xml = """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .05"/>
      <geom name="post" type="box" pos="0.1 0.62 0.9" size="0.05 0.05 0.9"/>
      <geom name="wall_e" type="box" pos="0.77 0 0.9" size="0.05 1.5 0.9"/>
      <geom name="wall_w" type="box" pos="-0.77 0 0.9" size="0.05 1.5 0.9"/>
      <geom name="wall_n" type="box" pos="0 0.77 0.9" size="1.5 0.05 0.9"/>
      <geom name="wall_s" type="box" pos="0 -0.77 0.9" size="1.5 0.05 0.9"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fp = _footprint()
    # three quarters of a turn to the left, the post on it; right round is past half a turn
    start, goal = (0.0, 0.0, 0.0), (0.0, 0.0, 0.75 * math.pi)
    with pytest.raises(NoWay) as err:
        plan_way(model, data, fp, start, goal, back_max=0.0, long_way=False)
    assert "long_way false" in str(err.value)
    way = plan_way(model, data, fp, start, goal, back_max=0.0, long_way=True)
    assert way.long_turns == 1, way.segments
    long = next(s for s in way.segments if s.kind == "turn" and abs(s.angle) > math.pi)
    assert long.angle < 0.0  # right round, away from the post on the left
    assert check_way(model, data, fp, way) == []


def test_a_tether_at_a_point_of_the_body_holds_that_point_within_reach_at_every_heading():
    """``tether_at`` names the point of the body a tether holds, in the body's frame: a point to
    the body's right, held within reach of a spot to its right, stays there turning a little to
    the left, and is out of reach turned a half turn round -- which the root alone allows."""
    xml = """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .05"/>
      <geom name="wall_e" type="box" pos="2 0 0.9" size="0.05 2 0.9"/>
      <geom name="wall_w" type="box" pos="-2 0 0.9" size="0.05 2 0.9"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fp = _footprint()
    tether, at = [(0.0, -0.3, 0.35)], [(0.0, -0.25)]
    start = (0.0, 0.0, 0.0)
    plan_way(model, data, fp, start, (0.0, 0.0, 0.3), tether=tether, tether_at=at)
    plan_way(model, data, fp, start, (0.0, 0.0, math.pi), tether=tether, long_way=True)
    with pytest.raises(NoWay):
        plan_way(
            model, data, fp, start, (0.0, 0.0, math.pi), tether=tether, tether_at=at, long_way=True
        )


def test_a_load_passes_over_a_bench_beside_the_way_and_not_under_a_shelf_at_its_height():
    """The tote is wider than the passage: the way exists only because the tote, at its height,
    passes over the bench. One band for all of it (a disc's question in a polygon's shape) finds
    none, and a shelf board at the tote's height over the bench leaves none, naming the shelf."""
    start, goal = (0.3, -0.12, 0.0), (3.9, -0.12, 0.0)
    model, data = _world("carry_shelf_over_bench")
    fp = _footprint(width=0.7)
    way = plan_way(model, data, fp, start, goal)
    assert check_way(model, data, fp, way) == []
    assert way.seconds < PLAN_SECONDS
    # (a search that finds nothing runs to its cap; a lower one says the same sooner)
    with pytest.raises(NoWay):
        plan_way(model, data, _footprint(width=0.7, one_band=True), start, goal, max_states=20000)
    model, data = _world("carry_shelf_over_bench_shelf")
    with pytest.raises(NoWay) as err:
        plan_way(model, data, fp, start, goal, max_states=20000)
    assert "shelf" in str(err.value), err.value


def test_a_margin_keeps_the_way_that_much_further_off():
    """A body that follows its way with an error plans with that error round it: no pose of the way
    comes nearer what it would meet than the margin."""
    model, data = _world("carry_post_straight")
    margin = 0.05
    fp = _footprint(margin=margin)
    way = plan_way(model, data, fp, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    assert way.margin == margin and way.summary()["margin"] == margin
    # the exact footprint grown by the margin meets nothing anywhere on the way
    assert check_way(model, data, fp, way) == []


def test_what_is_held_is_not_in_its_own_way():
    """A free body in the world is an obstacle while it stands still; the same body named as held
    (its weld root in skip_roots) is part of the footprint and not in the world."""
    xml = """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .05"/>
      <body name="tote" pos="0.42 0 0.95"><freejoint/><geom type="box" size="0.2 0.3 0.1"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fp = _footprint()
    with pytest.raises(NoWay) as err:
        plan_way(model, data, fp, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert "tote" in str(err.value)
    root = int(model.body_weldid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tote")])
    way = plan_way(model, data, fp, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), skip_roots=[root])
    assert way.length > 1.9


def test_via_points_are_passed_through_in_order():
    model, data = _world("carry_post_straight")
    way = plan_way(model, data, _footprint(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), via=[(1.0, -0.3)])
    assert min(math.hypot(p[0] - 1.0, p[1] + 0.3) for p in way.poses) < 0.04


def test_a_tether_holds_the_way_within_its_reach_until_past_its_line():
    """Tethered to a segment by the start, the goal far off is out of reach; with a line it may
    leave the tether past, the way goes on beyond that line, and stays in reach up to it."""
    model, data = _world("carry_post_straight")
    seg = (0.0, 0.5, 0.6, 0.5, 1.0)
    with pytest.raises(NoWay):
        plan_way(model, data, _footprint(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), tether=[seg])
    way = plan_way(
        model,
        data,
        _footprint(),
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        tether=[(*seg, 1.2, 0.0, 1.0, 0.0)],
    )
    assert way.poses[-1][0] > 3.9
    before = [p for p in way.poses if p[0] < 1.2]
    assert before and all(
        math.hypot(p[0] - min(max(p[0], 0.0), 0.6), p[1] - 0.5) <= 1.0 + 1e-6 for p in before
    )


def test_a_via_point_does_not_start_a_second_step_back():
    """A step back is once between moves forward across a via point too: in a corridor too narrow
    to turn in, a via point straight behind does not let the way step back twice running, which
    would reverse further than ``back_max`` without a move forward between."""
    xml = """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .05"/>
      <geom name="side_n" type="box" pos="0 0.45 0.9" size="2 0.05 0.9"/>
      <geom name="side_s" type="box" pos="0 -0.45 0.9" size="2 0.05 0.9"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    way = plan_way(
        model,
        data,
        _footprint(),
        (0.0, 0.0, 0.0),
        (-0.6, 0.0, 0.0),
        via=[(-0.3, 0.0)],
        back_max=0.3,
    )
    run = 0.0
    for seg in way.segments:
        run = run + seg.distance if seg.kind == "back" else run if seg.kind == "turn" else 0.0
        assert run <= 0.3 + 1e-9, way.segments


def test_a_way_reports_what_it_cost():
    model, data = _world("carry_three_bends")
    way = plan_way(model, data, _footprint(), (-1.0, 0.0, 0.0), (5.5, 0.92, 0.0))
    summary = way.summary()
    assert set(summary) == {
        "seconds",
        "expanded",
        "length",
        "reverses",
        "long_turns",
        "margin",
        "widened",
    }
    assert summary["expanded"] > 0 and 6.4 < summary["length"] < 7.5, summary


def test_a_back_max_that_is_no_distance_is_refused():
    model, data = _world("carry_post_straight")
    for bad in (-0.1, float("nan")):
        with pytest.raises(ValueError, match="back_max"):
            plan_way(model, data, _footprint(), (0, 0, 0), (1, 0, 0), back_max=bad)


# -- the footprint -------------------------------------------------------------------------------
def test_a_bins_cells_cover_every_heading_in_the_bin_from_anywhere_in_the_cell():
    """Conservative by construction: the piece posed at random headings within a bin, from random
    roots within a cell, never covers a cell centre the bin's cells leave out."""
    fp = Footprint([Piece(_box(0.6, 0.0, 1.2, 0.05), (0.9, 1.1), "profile")])
    rng = np.random.default_rng(0)
    res = fp.resolution
    for k in (0, 5, 17, 40):
        cells = {tuple(c) for c in fp.cells(k)[(0.9, 1.1)]}
        for _ in range(200):
            th = fp.heading(k) + rng.uniform(-0.5, 0.5) * 2 * math.pi / fp.bins
            off = rng.uniform(-0.5, 0.5, 2) * res
            poly = fp.pieces[0].posed(off[0], off[1], th)
            # every cell centre strictly inside the posed piece is a covered cell
            lo = np.floor(poly.min(0) / res).astype(int)
            hi = np.ceil(poly.max(0) / res).astype(int)
            for i in range(lo[0], hi[0] + 1):
                for j in range(lo[1], hi[1] + 1):
                    if L._inside(poly, (i * res, j * res), 0.0):
                        assert (i, j) in cells, (k, i, j)


def test_footprint_pieces_from_geoms_and_capsules_are_in_the_body_frame():
    xml = """<mujoco><worldbody>
      <geom name="g" type="box" pos="1.5 2.0 0.5" size="0.1 0.2 0.3"/></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    piece = geom_piece(model, data, 0, (1.0, 2.0, math.pi / 2), "g")
    # 0.5 m ahead of a body at (1, 2) facing +y is... 0.5 m to its right (-y in its frame)
    centre = piece.polygon.mean(0)
    assert np.allclose(centre, [0.0, -0.5], atol=1e-6), centre
    assert piece.z == pytest.approx((0.2, 0.8))
    arm = capsule_piece((0.1, 0.2, 1.3), (0.4, 0.2, 1.0), 0.05, "arm_l")
    assert arm.z == pytest.approx((0.95, 1.35))


def test_a_footprint_refuses_what_is_no_footprint():
    with pytest.raises(ValueError):
        Footprint([])
    with pytest.raises(ValueError):
        Piece(np.zeros((2, 2)), (0.0, 1.0))
    with pytest.raises(ValueError):
        Piece(_box(0, 0, 1, 1), (1.0, 0.5))
    with pytest.raises(ValueError):
        Footprint([Piece(_box(0, 0, 1, 1), (0.0, 1.0))], margin=-0.1)


def test_wall_polygons_with_labels_names_each_footprint():
    model, data = _world("carry_shelf_over_bench_shelf")
    polys = wall_polygons(model, data, 0.9, 1.0, labels=True)
    names = {n for _p, n in polys}
    assert "shelf" in names and "bench" not in names, names  # the bench's top is at 0.75 m
    plain = wall_polygons(model, data, 0.9, 1.0)
    assert [p for p, _n in polys] == plain


def _posts(points, radius=0.04):
    """A floor with a post of ``radius`` at each point."""
    posts = "".join(
        f'<geom name="p{i}" type="cylinder" size="{radius} 1" pos="{x} {y} 1"/>'
        for i, (x, y) in enumerate(points)
    )
    xml = f'<mujoco><worldbody><geom name="floor" type="plane" size="8 8 .1"/>{posts}</worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _broken_joins(way, bins):
    """The moves of ``way`` that do not go on from where the move before ended: an arc that is no
    arc of the search's from the pose before it (nor a straight shot along its heading), a turn
    that moves, a step back that is not straight back."""
    dth = 2 * math.pi / bins
    out = []
    for seg in way.segments:
        a, b = seg.poses[0], seg.poses[-1]
        if seg.kind == "turn":
            if (
                math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6
                or abs(L._wrap(a[2] + seg.angle - b[2])) > 1e-6
            ):
                out.append(("turn", a, b))
        elif seg.kind == "back":
            x, y = a[0] - seg.distance * math.cos(a[2]), a[1] - seg.distance * math.sin(a[2])
            if math.hypot(x - b[0], y - b[1]) > 1e-6 or abs(L._wrap(a[2] - b[2])) > 1e-6:
                out.append(("back", a, b))
        else:
            for a, b in zip(seg.poses[:-1], seg.poses[1:], strict=False):
                arcs = []
                for db in (0, 1, -1):
                    x, y = a[0], a[1]
                    for q in range(1, 5):
                        t = a[2] + db * dth * (q - 0.5) / 4
                        x, y = x + L.STEP / 4 * math.cos(t), y + L.STEP / 4 * math.sin(t)
                    arcs.append((x, y, a[2] + db * dth))
                arc = any(
                    math.hypot(x - b[0], y - b[1]) < 1e-6 and abs(L._wrap(th - b[2])) < 1e-6
                    for x, y, th in arcs
                )
                along = math.atan2(b[1] - a[1], b[0] - a[0])
                straight = abs(L._wrap(a[2] - b[2])) < 1e-6 and abs(L._wrap(along - a[2])) < 1e-6
                if not (arc or straight):
                    out.append(("walk", a, b))
    return out


def test_every_move_of_a_way_goes_on_from_where_the_one_before_ended():
    """The search keeps one pose per closed-set cell, the latest it reached it with; a way must
    still join each move to the pose it was made from, not to a later one of the same cell."""
    model, data = _posts(
        [
            (2.073, 0.268),
            (2.552, 0.545),
            (3.079, 1.017),
            (2.841, 0.078),
            (3.533, -0.81),
            (0.999, -1.161),
            (2.406, -0.345),
            (1.283, -0.423),
            (3.142, 0.588),
            (3.412, -0.546),
            (0.679, 0.734),
            (3.057, -0.1),
            (2.048, 0.576),
            (3.964, -0.971),
        ]
    )
    fp = _footprint()
    way = plan_way(model, data, fp, (0.0, 0.0, 0.0), (5.0, 0.0, 0.0))
    assert _broken_joins(way, fp.bins) == []


def test_a_turn_by_the_start_is_checked_across_its_sweep_not_at_its_bins():
    """By the start and the goal a pose the raster says is blocked is checked exactly. A turn on the
    spot there is checked across its sweep, as the raster's bins are: a 1 cm post at 1 m between
    two bins' headings is met by a rod reaching past it, though neither heading meets it."""
    r, at = 1.0, math.radians(46.25)
    model, data = _posts([(r * math.cos(at), r * math.sin(at))], radius=0.005)
    fp = Footprint(
        [
            Piece(_ellipse(0.12, 0.15), (0.0, 0.5), "legs"),
            Piece(_box(0.7, 0.0, 1.0, 0.02), (0.85, 1.05), "rod"),
        ]
    )
    try:
        way = plan_way(model, data, fp, (0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2), back_max=0.0)
    except NoWay:
        return  # nothing turns the rod a quarter turn there without meeting the post
    assert check_way(model, data, fp, way) == [], way.segments


def test_a_turn_on_the_spot_of_the_tote_meets_no_thin_post_beside_it():
    """The tote's footprint turned on the spot by the start, between posts of 1 cm: every pose of the
    way, checked exactly at every degree, meets none of them."""
    model, data = _posts(
        [
            (-0.39, 0.738),
            (0.575, -0.373),
            (0.603, 0.728),
            (-0.388, 0.707),
            (-0.555, 0.605),
            (0.693, -0.042),
            (0.087, -0.524),
        ],
        radius=0.01,
    )
    fp = _footprint()
    try:
        way = plan_way(model, data, fp, (0.0, 0.0, 0.0), (0.0, 0.0, 1.86), back_max=0.0)
    except NoWay:
        return
    assert check_way(model, data, fp, way) == [], way.segments


def test_a_way_from_one_wall_to_dock_at_another_passes_the_last_check():
    """What the body stands against at the start it may go on standing against as it leaves, and at
    the goal it may stand against what it docks at: the exact check every way gets before it is
    returned allows each where the search does, on a straight shot from one to the other too."""
    xml = """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .05"/>
      <body name="back"><geom name="back" type="box" pos="0.16 0 0.9" size="0.05 1.0 0.9"/></body>
      <body name="bench"><geom name="bench" type="box" pos="-0.76 0 0.25" size="0.05 1.0 0.25"/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fp = Footprint([Piece(_ellipse(0.12, 0.15), (0.0, 0.5), "legs", body=True)])
    way = plan_way(
        model, data, fp, (0.0, 0.0, math.pi), (-0.6, 0.0, math.pi), dock=["bench"], back_max=0.0
    )
    assert way.length > 0.55


def test_a_way_planned_in_steps_hands_back_while_it_searches_and_is_the_same_way():
    """``plan_way_steps`` hands back every ``SLICE_STATES`` states its search expands, so a caller
    on the physics thread steps the simulation on meanwhile; it reads the world before it first
    hands back, so what moves after that changes nothing; and its way is :func:`plan_way`'s."""
    model, data = _world("carry_three_bends")
    fp = _footprint()
    start, goal = (0.0, 0.0, 0.0), (5.5, 0.92, 0.0)
    whole = plan_way(model, data, fp, start, goal)
    steps = L.plan_way_steps(model, data, fp, start, goal)
    handed, way = 0, None
    while True:
        try:
            next(steps)
        except StopIteration as done:
            way = done.value
            break
        handed += 1
        data.qpos[:] = 0.0  # the world as a simulation steps it on: not read again
    assert handed >= max(1, whole.expanded // L.SLICE_STATES), (handed, whole.expanded)
    assert [s.kind for s in way.segments] == [s.kind for s in whole.segments]
    assert way.poses == whole.poses
    assert way.seconds < whole.seconds * 3 + 0.5  # the time it ran, not the time handed back


def _model(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_a_last_straight_shorter_than_a_step_stops_short_of_a_wall_at_the_goal():
    """The straight onto the goal is checked as long as it is: a shot of 5 cm that brings the tote
    to 2 cm short of a wall is no fault of the planner's."""
    model, data = _model(
        """<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .05"/>
        <geom name="wall" type="box" pos="1.05 0 0.9" size="0.05 1.5 0.9"/></worldbody></mujoco>"""
    )
    fp = _footprint()  # the tote's front 0.62 m ahead of the root: 0.98 at the goal
    for x0 in (0.31, 0.2):
        way = plan_way(model, data, fp, (x0, 0.0, 0.0), (0.36, 0.0, 0.0), back_max=0.0)
        assert check_way(model, data, fp, way) == []


def test_a_via_point_whose_leg_takes_every_state_leaves_no_way_for_the_next():
    """States run out at a via point: the next leg says there is no way, rather than failing on
    a search it never ran."""
    model, data = _world("carry_post_straight")
    fp = _footprint()
    leg = plan_way(model, data, fp, (0.0, 0.0, 0.0), (1.0, -0.3, None))
    with pytest.raises(NoWay, match=f"it gave up after {leg.expanded} states"):
        plan_way(
            model,
            data,
            fp,
            (0.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            via=[(1.0, -0.3)],
            max_states=leg.expanded,
        )


def test_a_goal_the_body_would_stand_in_at_every_heading_is_refused_naming_it():
    """With no heading given, a goal where the body meets something whichever way it faces is
    refused before the search, naming what is there."""
    model, data = _world("carry_post_straight")
    with pytest.raises(NoWay, match=r"at \(2\.00, 0\.33\) facing any way it meets post"):
        plan_way(model, data, _footprint(), (0.0, 0.0, 0.0), (2.0, 0.33, None))


def test_a_sweep_keeps_the_bow_it_is_given_to_spare():
    """``sweep_free`` checks a move with ``bow`` to spare, how far the path the root takes may stray
    from the straight line between its ends (an arc from its chord): a post 3 cm beside a straight
    move is clear of it with no bow and met with a bow of 5 cm."""
    model, data = _model("""<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
      <geom name="post" type="box" pos="1.0 0.23 0.5" size="0.05 0.05 0.5"/></worldbody></mujoco>""")
    fp = Footprint([Piece(_box(0.0, 0.0, 0.2, 0.3), (0.0, 1.0), "body")])
    layers = L.Layers(
        model, data, fp.bands, L._bounds(model, data, [(0, 0), (2, 0)], fp.radius), fp.resolution
    )
    cs = L._CSpace(layers, fp, (0, 0, *layers.shape), np.ones(layers.shape, bool))
    cs._exact_setup()
    assert cs.sweep_free((0.8, 0.0, 0.0), (1.2, 0.0, 0.0), frozenset(), 0.0)
    assert not cs.sweep_free((0.8, 0.0, 0.0), (1.2, 0.0, 0.0), frozenset(), 0.05)


def test_a_goal_facing_a_given_way_near_the_start_may_stand_against_what_the_start_does():
    """Where the goal's zone and the start's overlap, the goal may stand against what each allows,
    with a heading given as without one."""
    model, data = _model("""<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
      <body name="bench" pos="0 0.22 0.25"><geom name="bench" type="box" size="0.6 0.1 0.25"/></body>
      <geom name="wall" type="box" pos="0 -2 0.5" size="3 0.05 0.5"/></worldbody></mujoco>""")
    fp = Footprint([Piece(_box(0.0, 0.0, 0.2, 0.3), (0.0, 1.0), "body", body=True)])
    for goal in ((0.10, 0.0, None), (0.10, 0.0, 0.0)):
        way = plan_way(model, data, fp, (0.0, 0.0, 0.0), goal)
        assert math.hypot(way.poses[-1][0] - 0.10, way.poses[-1][1]) < 0.05, (goal, way.poses[-1])
