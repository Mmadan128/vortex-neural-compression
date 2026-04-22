#!/usr/bin/env python3
"""Thin wrapper around cms_experiment_lg/download.py with local defaults.

If no args are provided, it runs create-bin mode into experiments/cms_experiment/data.
Any user-provided args are forwarded as-is to the underlying CMS converter.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    src = here.parent / "cms_experiment_lg" / "download.py"

    forwarded = sys.argv[1:]
    if not forwarded:
        forwarded = [
            "--create-bin",
            "--out-dir",
            str(here / "data"),
            "--bin-out",
            "cms_events.bin",
        ]

    cmd = [sys.executable, str(src), *forwarded]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
