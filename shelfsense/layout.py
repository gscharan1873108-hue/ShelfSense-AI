"""Shelf geometry: locates shelf-edge boards, shelf extents and price-tag regions."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .models import Box


@dataclass
class ShelfGeometry:
    index: int
    x0: int
    x1: int
    y0: int  # top of the product area
    y1: int  # top of the board (= bottom of the product area)
    board_y1: int
    px_per_cm: float

    @property
    def product_area(self) -> Box:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def board_area(self) -> Box:
        return (self.x0, self.y1, self.x1, self.board_y1)


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs


def detect_layout(img: np.ndarray, shelf_indices: list[int], shelf_width_cm: float,
                  cfg: Config) -> tuple[list[ShelfGeometry], list[str]]:
    h, w = img.shape[:2]
    warnings: list[str] = []
    mask = np.abs(img.astype(np.int16) - np.array(cfg.board_bgr, np.int16)).sum(axis=2) < cfg.board_tol
    bands = _runs(mask.mean(axis=1) > cfg.min_board_row_frac, max(8, int(h * 0.015)))

    if len(bands) != len(shelf_indices):
        warnings.append(
            f"Found {len(bands)} shelf boards, planogram has {len(shelf_indices)}; "
            "falling back to equal vertical split (check board colour calibration).")
        step = h / len(shelf_indices)
        band_h = int(step * 0.17)
        bands = [(int((i + 1) * step) - band_h, int((i + 1) * step)) for i in range(len(shelf_indices))]
        x0, x1 = int(w * 0.03), int(w * 0.97)
        xs = [(x0, x1)] * len(bands)
    else:
        xs = []
        for y0, y1 in bands:
            cols = np.flatnonzero(mask[y0:y1].mean(axis=0) > 0.1)
            xs.append((int(cols[0]), int(cols[-1]) + 1))

    x0, x1 = int(np.median([x[0] for x in xs])), int(np.median([x[1] for x in xs]))
    px_per_cm = (x1 - x0) / shelf_width_cm
    heights = [b[1] - b[0] for b in bands]
    spacing = int(np.median(np.diff([b[0] for b in bands]))) if len(bands) > 1 else int(h * 0.3)
    area_h = spacing - int(np.median(heights))

    geoms = []
    for i, ((by0, by1), idx) in enumerate(zip(bands, shelf_indices)):
        top = bands[i - 1][1] if i > 0 else max(0, by0 - area_h)
        geoms.append(ShelfGeometry(idx, x0, x1, top, by0, by1, px_per_cm))
    return geoms, warnings


def find_tags(img: np.ndarray, g: ShelfGeometry) -> list[Box]:
    """Bright, low-saturation rectangles on the shelf-edge board."""
    bx0, by0, bx1, by1 = g.board_area
    band = img[by0:by1, bx0:bx1]
    white = (band.min(axis=2) > 185).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(white)
    boxes = []
    for x, y, w, h, _ in stats[1:].tolist():
        if w >= 3 * g.px_per_cm and h >= 0.4 * (by1 - by0):
            boxes.append((bx0 + x, by0 + y, bx0 + x + w, by0 + y + h))
    return sorted(boxes)
