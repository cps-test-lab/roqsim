"""Eyes on an roqsim scene -- an agent's and a human's.

Exposes an MCP server (:func:`roqsim_scene_builder.server.create_server`) with three tools over
whatever ``roqsim`` can load:

* :func:`sketch_floorplan_by_human` -- a 2D top-view window where a human draws a floorplan's walls
  and returns a structured sketch;
* :func:`review_scene_by_human` -- a native MuJoCo 3D window (walk/look, comment dots, prop moves)
  that blocks until a human returns a verdict;
* :func:`render_scene` -- no window and no human: a PNG of a world, a model, or a moment from a
  recorded run.

See ``docs/scene_builder.rst`` for the full contract.
"""

from roqsim_scene_builder.floorplan_sketch import sketch_floorplan_by_human
from roqsim_scene_builder.scene_render import render_scene
from roqsim_scene_builder.scene_review import review_scene_by_human

__all__ = ["render_scene", "review_scene_by_human", "sketch_floorplan_by_human"]
