# Copyright (C) 2026 Frederik Pasch
# SPDX-License-Identifier: Apache-2.0
"""One place where roqsim's command-line entry points configure logging.

Six entry points repeated the same three-line ``basicConfig`` call, which is six copies of
one decision -- and the decision is not obvious, because roqsim's log lines are read in two
very different places:

* **A terminal**, where ``roqsim sim`` is one command a person is watching. ``INFO roqsim.engine:
  seed 42`` is what belongs there: short, and no timestamp competing with the message.
* **An aggregated log**, where roqsim is one producer among many and every line has to be
  placeable in time and attributable. There a bare ``INFO roqsim.engine:`` cannot be ordered
  against anything else, because it says nothing about when it happened.

So the format is selectable, and the terminal case is the default: making roqsim's own output
worse for every standalone user, to suit an aggregator that may not even be present, would be
the wrong trade. :data:`STAMPED_FORMAT` is opt-in through ``ROQSIM_LOG_FORMAT=stamped``, and an
aggregator that wants it sets that variable itself.

Note that :data:`STAMPED_FORMAT` needs no clock call of its own: ``%(created)`` is the log
record's own epoch timestamp, taken when the event happened rather than when it was formatted.

``export_capture`` deliberately does **not** use this. Its logging format is bare
``%(message)s`` because that CLI's contract is one line of JSON on stdout, and its output is
plain by design -- so it is not an oversight to be tidied up into this helper later.
"""

import logging
import os

#: The default: what roqsim has always printed, unchanged.
PLAIN_FORMAT = "%(levelname)s %(name)s: %(message)s"

#: ``[LEVEL] [epoch] [logger]: message`` -- the shape ROS tooling writes, so a log aggregator
#: has one grammar to parse rather than one per producer.
STAMPED_FORMAT = "[%(levelname)s] [%(created).6f] [%(name)s]: %(message)s"

#: Environment variable selecting between them. Named for what it controls rather than for
#: who sets it: roqsim does not know or care which aggregator is reading.
FORMAT_ENV = "ROQSIM_LOG_FORMAT"

FORMATS = {"plain": PLAIN_FORMAT, "stamped": STAMPED_FORMAT}


def log_format(name: str = None) -> str:
    """The format named by *name*, or by ``ROQSIM_LOG_FORMAT``, defaulting to plain.

    An unrecognised name raises rather than falling back. A silent fallback here would mean
    an aggregator that set the variable and got plain output would look correctly configured
    while producing lines nothing downstream can place -- and a typo is exactly how that
    happens.
    """
    name = name or os.environ.get(FORMAT_ENV) or "plain"
    try:
        return FORMATS[name]
    except KeyError:
        raise ValueError(
            f"{FORMAT_ENV}={name!r} is not a known log format; "
            f"expected one of {', '.join(sorted(FORMATS))}"
        ) from None


def configure(*, verbose: bool = False, stream=None) -> None:
    """Configure logging for an roqsim CLI.

    *stream* is for the entry points whose **stdout is a machine payload** (``roqsim state``'s
    JSON/CSV, ``roqsim render``'s JSON record): they pass ``sys.stderr`` so log lines cannot
    corrupt it. Defaulting it to ``None`` keeps ``basicConfig``'s own default (stderr) rather
    than choosing one here, so the two cases stay visibly different at the call site.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=log_format(),
        stream=stream,
    )
