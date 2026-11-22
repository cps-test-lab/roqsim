# SPDX-License-Identifier: Apache-2.0
"""roqsim_motion: one motion core for a humanoid body -- the walker's mocap skeleton, through the
interface any body offers (:mod:`.body`).

A humanoid's moves are one entity's procedures, as a machine's programs are; which moves a worker
makes, in which order and for what, is a scenario's (``docs/architecture.rst``, "Machines and
scenarios"). This package holds what does not depend on the body: the canonical description and its
forward kinematics (:mod:`.description`), the whole-body solver (:mod:`.ik`), hand targets
(:mod:`.targets`) and the hands' poses (:mod:`.hands`).
"""
