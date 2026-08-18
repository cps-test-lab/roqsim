"""Sensor-coverage estimation for rst worlds.

Given a compiled MuJoCo world and a set of sensors with known fields of view, this subpackage answers
*how much of the room and which objects are observed, by how many sensors (0..N)?* -- and provides the
pieces to search for placements that reach a target coverage.

Layered so the pieces compose without dragging in the whole stack:

* :mod:`~roqsim_sensors.coverage.fov` -- the single shared :class:`SensorFov` definition and the
  ``in_fov`` membership test. Knows nothing about any specific sensor.
* :mod:`~roqsim_sensors.coverage.adapters` -- the only module that knows sensor specifics: a
  per-type registry that builds a :class:`SensorFov` from a sensor's resolved parameters + world pose.
  A new/special sensor gets a new adapter here; the sensor *plugins* are never edited.
* :mod:`~roqsim_sensors.coverage.engine` -- :func:`coverage`, the visibility computation.
* :mod:`~roqsim_sensors.coverage.sampling` -- where to put the sample points (room volume + object
  surfaces), using only numpy + MuJoCo raycasts (no scipy/trimesh).
* :mod:`~roqsim_sensors.coverage.catalog` -- the sensor catalog with constrained-mount metadata.
* :mod:`~roqsim_sensors.coverage.report` / :mod:`~roqsim_sensors.coverage.viz` -- the two
  outputs (an agent-digestible JSON report and a human visualisation).

Two front doors onto this core: the ``sensor_coverage_probe`` runtime plugin (a world-YAML toggle) and
the ``roqsim sensors coverage`` CLI (:mod:`~roqsim_sensors.coverage.cli`).
"""

from __future__ import annotations

from .fov import FovKind, SensorFov, in_fov

__all__ = ["FovKind", "SensorFov", "in_fov"]
