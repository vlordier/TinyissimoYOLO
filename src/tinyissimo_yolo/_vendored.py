"""Ensure the vendored ultralytics package is importable."""

import sys
from pathlib import Path


def _ensure_vendored_path() -> None:
    project_root: str = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_vendored_path()
