"""Annotated image rendering."""
from __future__ import annotations

import cv2
import numpy as np

from .models import Report

GREEN, ORANGE, RED, BLUE = (60, 180, 60), (0, 140, 255), (40, 40, 220), (200, 120, 20)


def _label(img, text, org, color, scale=0.45):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def annotate(img: np.ndarray, report: Report) -> np.ndarray:
    out = img.copy()
    overlay = out.copy()
    for r in report.slots:
        if r.status == "empty":
            x0, y0, x1, y1 = r.box
            cv2.rectangle(overlay, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), RED, -1)
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    for r in report.slots:
        x0, y0, x1, y1 = r.box
        if r.status == "empty":
            cv2.rectangle(out, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), RED, 2)
            _label(out, "EMPTY", (x0 + 6, y1 - 6), RED)
    for dets in report.detections.values():
        for d in dets:
            cv2.rectangle(out, d.box[:2], d.box[2:], GREEN, 2)
    for r in report.slots:
        if r.status == "misplaced":
            x0, y0, _, y1 = r.box
            cv2.rectangle(out, (x0 + 3, y0 + 3), (r.box[2] - 3, y1 - 1), ORANGE, 3)
            _label(out, f"WRONG: want {r.slot.sku[-4:]}", (x0 + 6, y0 + 22), ORANGE, 0.4)
    for c in report.labels:
        if c.tag is None:
            continue
        ok = c.status == "ok"
        col = GREEN if ok else RED
        cv2.rectangle(out, c.tag.box[:2], c.tag.box[2:], col, 2)
        txt = "OK" if ok else c.status.replace("_", " ").upper()
        _label(out, txt, (c.tag.box[0], c.tag.box[1] - 2), col, 0.4)
    s = report.scores
    _label(out, f"Planogram compliance {s['planogram_compliance']:.0f}%  |  Overall {s['overall']:.0f}/100",
           (10, 20), BLUE, 0.6)
    return out
