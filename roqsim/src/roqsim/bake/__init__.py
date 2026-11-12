# Copyright (C) 2026 Frederik Pasch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Baked lighting: a world's light path-traced once (Blender's Cycles), kept with the world, drawn by
MuJoCo's classic renderer; the stored format is renderer-neutral.

The pieces, in the order they run:

* :mod:`.settings` -- the presets (``default`` for a CPU, ``high`` for the full quality) and the
  run-time light rig the classic renderer draws a bake under;
* :mod:`.export` -- the world as loaded, split into what is baked (static, drawn) and what moves, its
  lights resolved (:mod:`.lights`) and its fingerprint (:mod:`.fingerprint`), in one call by
  :mod:`.analysis`;
* :mod:`.store` -- the bake on disk, renderer-neutral: light maps without the surface colour, the
  lightmap UVs, ``bake.json``;
* :mod:`.compose` -- MuJoCo's classic renderer's part: the light map times the model's own colour,
  divided by the rig's light, composed in memory when the world loads;
* :mod:`.run` -- ``roqsim bake``: the export handed to Blender (:mod:`.blender_bake`, which runs
  *inside* Blender and is never imported here), its passes turned into the stored bake.

The plugin a world turns it on with is :mod:`roqsim.plugins.baked_lighting`.
"""
