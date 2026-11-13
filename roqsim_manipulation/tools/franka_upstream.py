#!/usr/bin/env python3
"""Run-from-the-folder wrapper for `roqsim arm franka-upstream`; the logic is in roqsim_manipulation.cli.franka_upstream."""

from roqsim_manipulation.cli.franka_upstream import main

raise SystemExit(main())
