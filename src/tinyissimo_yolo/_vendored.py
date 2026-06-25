"""Ensure the vendored ultralytics package is importable.

The ultralytics/ directory at the project root is a vendored copy that
needs to be on sys.path for imports to resolve at runtime.
"""

import os
import sys


def _ensure_vendored_path() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_vendored_path()
