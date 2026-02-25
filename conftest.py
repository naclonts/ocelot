"""
Root-level pytest conftest.

Adds the project root to sys.path so that host-side Python modules under
sim/, tests/, etc. are importable without a pip install step.
"""
import sys
from pathlib import Path

# Ensure project root is on the path when running `pytest` from any directory.
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
