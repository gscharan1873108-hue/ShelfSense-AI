"""Product detection (segmentation) and SKU recognition against catalogue packs."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .layout import ShelfGeometry
from .models import Detection, Sku
from .synthetic import render_pack


class SkuClassifier:
    """Nearest-reference classifier on banded Lab colour features.

    References come from `ref_dir/<sku>.png|jpg` when present, otherwise from the
    procedural pack renderer. Replace with a CNN embedding for real packaging.
    """

    def __init__(self, catalog: dict[str, Sku], cfg: Config, ref_dir: str | Path | None = None):
        self.catalog, self.cfg = catalog, cfg
        self.refs: dict[str, np.ndarray] = {}
        for sku in catalog.values():
            ref = None
            if ref_dir:
                for ext in ("png", "jpg", "jpeg"):
                    p = Path(ref_dir) / f"{sku.sku}.{ext}"
                    if p.exists():
                        ref = cv2.imread(str(p))
                        break
            if ref is None:
                ref = render_pack(sku, int(sku.width_cm * 10) - 6, int(sku.height_cm * 10))
            self.refs[sku.sku] = self._features(ref)

    @staticmethod
    def _features(crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        my, mx = max(1, int(h * 0.06)), max(1, int(w * 0.1))
        c = cv2.resize(crop[my:h - my, mx:w - mx], (16, 40), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
        bands = lab.reshape(5, 8, 16, 3).mean(axis=(1, 2))
        bands[:, 0] *= 0.6
        return bands.ravel()

    def classify(self, crop: np.ndarray) -> tuple[str | None, float]:
        f = self._features(crop)
        best, dist = min(((s, float(np.linalg.norm(f - r))) for s, r in self.refs.items()),
                         key=lambda t: t[1])
        return (best if dist <= self.cfg.unknown_dist else None), dist


def _back_panel_colour(region: np.ndarray) -> np.ndarray:
    """Most common colour in the upper third of the shelf (above all but the tallest packs)."""
    px = region[: max(4, region.shape[0] // 3)].reshape(-1, 3)
    keys = (px // 16).astype(np.int32) @ np.array([256, 16, 1])
    vals, counts = np.unique(keys, return_counts=True)
    return np.median(px[keys == vals[counts.argmax()]], axis=0)


def detect_products(img: np.ndarray, g: ShelfGeometry, clf: SkuClassifier,
                    cfg: Config) -> list[Detection]:
    x0, y0, x1, y1 = g.product_area
    region = img[y0:y1, x0:x1]
    bg = _back_panel_colour(region)
    fg = (np.abs(region.astype(np.int16) - bg.astype(np.int16)).sum(axis=2) > cfg.fg_tol).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((9, 3), np.uint8))  # removes thin bridges along the board edge
    _, _, stats, _ = cv2.connectedComponentsWithStats(fg)

    boxes = [(x, y, x + w, y + h) for x, y, w, h, _ in stats[1:].tolist()
             if w >= cfg.min_product_w_cm * g.px_per_cm and h >= cfg.min_product_h_cm * g.px_per_cm]
    # Drop boxes nested inside another (inner regions of light-coloured packs).
    boxes = [b for b in boxes
             if not any(o != b and o[0] <= b[0] and o[1] <= b[1] and o[2] >= b[2] and o[3] >= b[3]
                        for o in boxes)]

    dets: list[Detection] = []
    for bx0, by0, bx1, by1 in sorted(boxes):
        sku_id, dist = clf.classify(region[by0:by1, bx0:bx1])
        parts = [(bx0, bx1)]
        if sku_id:
            unit = clf.catalog[sku_id].width_cm * g.px_per_cm
            n = round((bx1 - bx0) / unit)
            if n >= 2 and (bx1 - bx0) > 1.6 * unit:  # touching facings merged into one blob
                step = (bx1 - bx0) / n
                parts = [(int(bx0 + i * step), int(bx0 + (i + 1) * step)) for i in range(n)]
        for px0, px1 in parts:
            if len(parts) > 1:
                sku_id, dist = clf.classify(region[by0:by1, px0:px1])
            conf = max(0.0, 1 - dist / cfg.unknown_dist)
            dets.append(Detection((x0 + px0, y0 + by0, x0 + px1, y0 + by1), sku_id, dist, conf))
    return dets
