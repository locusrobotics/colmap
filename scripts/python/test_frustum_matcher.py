#!/usr/bin/env python3

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    runpy.run_module("frustum_matcher.test_frustum_matcher", run_name="__main__")
