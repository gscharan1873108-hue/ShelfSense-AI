"""Shelf-label reading and label/price verification against the catalogue."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import cv2
import numpy as np

from .config import Config
from .layout import ShelfGeometry
from .models import Box, Issue, LabelCheck, Sku, SlotResult, TagRead
from .ocr import OcrEngine

_CURRENCY = re.compile(r"^\s*(r[s5]\.?|inr|₹|₨|\$)?\s*", re.I)
_NUMBER = re.compile(r"\d+(?:[.,]\d{1,2})?")


def parse_price(text: str) -> float | None:
    """Parse a whole-line price like 'Rs 40.00'; lines with other words return None."""
    t = _CURRENCY.sub("", text).strip().translate(str.maketrans("OoIl", "0011"))
    return float(t.replace(",", ".")) if _NUMBER.fullmatch(t) else None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_name(text: str, catalog: dict[str, Sku]) -> tuple[str | None, float]:
    t = _norm(text)
    if not t:
        return None, 0.0
    scored = [(max(SequenceMatcher(None, t, _norm(s.short_name)).ratio(),
                   SequenceMatcher(None, t, _norm(s.name)).ratio()), s.sku)
              for s in catalog.values()]
    score, sku = max(scored)
    return sku, score


def read_tag(img: np.ndarray, box: Box, ocr: OcrEngine, catalog: dict[str, Sku]) -> TagRead:
    x0, y0, x1, y1 = box
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    lines = ocr.read(crop)
    digit_lines = [ln for ln in lines if parse_price(ln[0]) is not None]
    price_line = max(digit_lines, key=lambda ln: ln[2], default=None)
    name_text = " ".join(t for t, _, _ in lines if price_line is None or t != price_line[0])
    sku, name_score = match_name(name_text, catalog)
    conf = min((s for _, s, _ in lines), default=0.0)
    raw = " | ".join(t for t, _, _ in lines)
    return TagRead(box, raw, parse_price(price_line[0]) if price_line else None, sku, name_score, conf)


def check_labels(shelf_index: int, groups: dict[int, tuple[str, Box]], tags: list[TagRead],
                 results: list[SlotResult], catalog: dict[str, Sku], cfg: Config
                 ) -> tuple[list[LabelCheck], list[Issue]]:
    """groups: group id -> (expected SKU, x-range box). Each tag is assigned to the group containing its centre."""
    by_group: dict[int, TagRead] = {}
    for t in tags:
        cx = (t.box[0] + t.box[2]) / 2
        for g, (_, gb) in groups.items():
            if gb[0] <= cx < gb[2] and (g not in by_group or t.ocr_conf > by_group[g].ocr_conf):
                by_group[g] = t

    checks, issues = [], []
    for g, (exp, gb) in groups.items():
        sku = catalog[exp]
        under = [r.found_sku for r in results if r.slot.group == g and r.found_sku]
        product_under = max(set(under), key=under.count) if under else None
        tag = by_group.get(g)
        pos = min(r.slot.position for r in results if r.slot.group == g)
        status, read_price = "ok", tag.price if tag else None

        def add(sev: str, typ: str, msg: str, act: str) -> None:
            issues.append(Issue(typ, sev, shelf_index, pos, exp, msg, act, box=tag.box if tag else gb))

        if tag is None:
            status = "missing"
            add("medium", "missing_label", f"No shelf label for {sku.name}.",
                f"Print and place a label for {sku.name} (Rs {sku.price:.2f}).")
        elif tag.matched_sku is None or tag.name_score < cfg.name_match_min \
                or tag.price is None or tag.ocr_conf < cfg.ocr_min_conf:
            status = "unreadable"
            add("low", "label_unreadable", f"Label under {sku.name} could not be read reliably "
                f"(OCR: '{tag.raw_text}', confidence {tag.ocr_conf:.2f}).", "Manually verify this label.")
        elif tag.matched_sku != exp:
            status = "wrong_label"
            wrong = catalog[tag.matched_sku]
            add("high", "label_mismatch", f"Label shows '{wrong.name}' but this position is planned for {sku.name}.",
                f"Replace label with {sku.name} (Rs {sku.price:.2f}).")
        elif abs(tag.price - sku.price) > cfg.price_tol:
            status = "wrong_price"
            add("high", "price_mismatch", f"{sku.name}: shelf price Rs {tag.price:.2f} vs catalogue Rs {sku.price:.2f}.",
                f"Correct the label to Rs {sku.price:.2f}.")
        checks.append(LabelCheck(shelf_index, g, exp, product_under, tag, status, sku.price, read_price))
    return checks, issues
