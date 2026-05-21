#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def _add_src_to_path() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main() -> int:
    _add_src_to_path()
    from aod.aod_hallusionbench_train import main as impl_main

    return int(impl_main())


if __name__ == "__main__":
    raise SystemExit(main())

