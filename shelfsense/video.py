"""Video support: sample frames, analyse each, keep issues seen in most frames."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from statistics import median

import cv2

from .models import Report
from .narrator import summarize
from .pipeline import ShelfSenseAnalyzer


def sample_frames(path: str, every_s: float = 1.0, max_frames: int = 30):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps * every_s))
    i, n = 0, 0
    try:
        while n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                n += 1
                yield frame
            i += 1
    finally:
        cap.release()


def analyze_video(analyzer: ShelfSenseAnalyzer, path: str, every_s: float = 1.0, max_frames: int = 30,
                  min_support: float = 0.5) -> tuple[Report, object, int]:
    """Returns (consensus report, representative frame, frames analysed)."""
    runs = [(f, analyzer.analyze(f, narrate=False)) for f in sample_frames(path, every_s, max_frames)]
    if not runs:
        raise ValueError("No frames could be read from the video.")
    key = lambda i: (i.type, i.shelf, i.position, i.sku)  # noqa: E731
    votes = Counter(key(i) for _, r in runs for i in r.issues)
    keep = {k for k, v in votes.items() if v / len(runs) >= min_support}
    mid = median(r.scores["overall"] for _, r in runs)
    frame, best = min(runs, key=lambda fr: abs(fr[1].scores["overall"] - mid))
    consensus = replace(best, issues=[i for i in best.issues if key(i) in keep])
    consensus.summary = summarize(consensus)
    return consensus, frame, len(runs)
