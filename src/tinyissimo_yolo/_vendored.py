"""Ensure the vendored ultralytics package is importable.

The ultralytics/ directory at the project root is a vendored copy that
needs to be on sys.path for imports to resolve at runtime.
"""

import sys
from pathlib import Path


def _ensure_vendored_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_vendored_path()
