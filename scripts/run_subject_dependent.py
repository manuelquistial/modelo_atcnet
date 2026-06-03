#!/usr/bin/env python3
"""Deprecated: BCI IV-2a replaced by PhysioNet MI. Use scripts/run_holdout.py."""

import sys

print(
    "BCI IV-2a subject-dependent protocol removed.\n"
    "PhysioNet MI (left vs right hand) hold-out:\n"
    "  python scripts/run_holdout.py\n"
    "  python scripts/run_holdout.py --quick\n",
    file=sys.stderr,
)
sys.exit(1)
