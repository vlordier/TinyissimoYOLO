"""Ensure the vendored ultralytics package is importable.

Used as a fallback when ultralytics is not installed as a proper dependency
(e.g. running ``python src/...`` directly without ``uv run``).
"""

import sys
from pathlib import Path


def _ensure_vendored_path() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        project_root: str = str(Path(__file__).resolve().parents[2])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)


_ensure_vendored_path()
