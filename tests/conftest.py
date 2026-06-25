"""Shared test fixtures and helpers."""

import numpy as np
import pytest


@pytest.fixture
def dummy_image() -> np.ndarray:
    """Return a 100x200 black image (HWC, uint8)."""
    return np.zeros((100, 200, 3), dtype=np.uint8)


def gt_box(
    xc_norm: float,
    yc_norm: float,
    w_norm: float,
    h_norm: float,
    img_w: int = 200,
    img_h: int = 100,
) -> list[float]:
    """Convert normalized GT coords to pixel ``[x1, y1, x2, y2]``."""
    xc, yc = xc_norm * img_w, yc_norm * img_h
    w, h = w_norm * img_w, h_norm * img_h
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]
