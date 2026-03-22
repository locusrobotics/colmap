#!/usr/bin/env python3

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from frustum_matcher.run_frustum_matching import main

if __name__ == "__main__":
    main()
