"""OCR engines. RapidOCR (ONNX Runtime, CPU) is used because it runs on edge devices."""
from __future__ import annotations

import os
from typing import Protocol

import numpy as np


class OcrEngine(Protocol):
    def read(self, img: np.ndarray) -> list[tuple[str, float, float]]:
        """Return (text, confidence, line_height_px) per detected text line, top to bottom."""


class RapidOcrEngine:
    def __init__(self, threads: int | None = None) -> None:
        from rapidocr_onnxruntime import RapidOCR
        # Tag crops are small: don't upscale them to the default 736 px detector input.
        # Unbounded ONNX threading is ~20x slower on many-core CPUs; edge boards use 1-4.
        threads = threads or int(os.environ.get("SHELFSENSE_THREADS", "4"))
        self._engine = RapidOCR(det_limit_type="max", det_limit_side_len=480, intra_op_num_threads=threads)

    def read(self, img: np.ndarray) -> list[tuple[str, float, float]]:
        result, _ = self._engine(img)
        lines = []
        for box, text, score in result or []:
            ys = [p[1] for p in box]
            lines.append((str(text), float(score), max(ys) - min(ys), min(ys)))
        lines.sort(key=lambda t: t[3])
        return [(t, s, h) for t, s, h, _ in lines]


def default_engine() -> OcrEngine | None:
    try:
        return RapidOcrEngine()
    except ImportError:
        return None
