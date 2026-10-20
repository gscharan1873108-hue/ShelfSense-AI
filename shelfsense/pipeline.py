from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

from .availability import availability
from .config import DEFAULT, Config
from .detector import SkuClassifier, detect_products
from .labels import check_labels, read_tag
from .layout import detect_layout, find_tags
from .models import Planogram, Report, Sku
from .narrator import summarize
from .ocr import OcrEngine, default_engine
from .planogram import match_slots, placement_issues, slot_box

_SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


class ShelfSenseAnalyzer:
    def __init__(self, catalog: dict[str, Sku], plan: Planogram, ocr: OcrEngine | str | None = "auto",
                 cfg: Config = DEFAULT, ref_dir=None):
        self.catalog, self.plan, self.cfg = catalog, plan, cfg
        self.ocr = default_engine() if ocr == "auto" else ocr
        self.clf = SkuClassifier(catalog, cfg, ref_dir)

    def analyze(self, img, narrate: bool = True) -> Report:
        timings: dict[str, float] = defaultdict(float)

        @contextmanager
        def timed(name: str):
            t = time.perf_counter()
            yield
            timings[name] += (time.perf_counter() - t) * 1000

        cfg, plan = self.cfg, self.plan
        with timed("layout"):
            geoms, warnings = detect_layout(img, [s.index for s in plan.shelves], plan.shelf_width_cm, cfg)
        report = Report(plan.store, (img.shape[1], img.shape[0]), {}, warnings=warnings, shelf_geometry=geoms)
        if self.ocr is None:
            warnings.append("No OCR engine installed (pip install rapidocr-onnxruntime): label checks skipped.")

        checks_all = []
        for shelf, g in zip(plan.shelves, geoms):
            slots = shelf.slots(self.catalog)
            with timed("detect"):
                dets = detect_products(img, g, self.clf, cfg)
            report.detections[shelf.index] = dets
            results, extras = match_slots(slots, dets, g, cfg)
            report.slots += results
            report.issues += placement_issues(results, extras, shelf, plan, self.catalog, cfg)
            rows, av_issues = availability(results, shelf, self.catalog, cfg)
            report.availability += rows
            report.issues += av_issues

            if self.ocr is not None:
                groups: dict[int, tuple[str, tuple[int, int, int, int]]] = {}
                for s in slots:
                    b = slot_box(s, g)
                    prev = groups.get(s.group)
                    groups[s.group] = (s.sku, b if prev is None else (prev[1][0], b[1], b[2], b[3]))
                with timed("ocr"):
                    tags = [read_tag(img, b, self.ocr, self.catalog) for b in find_tags(img, g)]
                checks, label_issues = check_labels(shelf.index, groups, tags, results, self.catalog, cfg)
                report.labels += checks
                checks_all += checks
                report.issues += label_issues

        total = len(report.slots)
        scores = {
            "planogram_compliance": 100 * sum(r.status == "ok" for r in report.slots) / total,
            "availability": 100 * (1 - sum(r.status == "empty" for r in report.slots) / total),
        }
        weights = {"planogram_compliance": cfg.w_planogram, "availability": cfg.w_availability}
        if checks_all:
            scores["label_accuracy"] = 100 * sum(c.status == "ok" for c in checks_all) / len(checks_all)
            weights["label_accuracy"] = cfg.w_labels
        scores["overall"] = sum(scores[k] * w for k, w in weights.items()) / sum(weights.values())
        for shelf in plan.shelves:
            rs = [r for r in report.slots if r.slot.shelf == shelf.index]
            scores[f"shelf_{shelf.index}_planogram"] = 100 * sum(r.status == "ok" for r in rs) / len(rs)
        report.scores = {k: round(v, 1) for k, v in scores.items()}

        report.issues.sort(key=lambda i: (_SEV_RANK[i.severity], -i.revenue_at_risk, i.shelf, i.position or 0))
        report.timings_ms = {k: round(v, 1) for k, v in timings.items()}
        report.timings_ms["total"] = round(sum(timings.values()), 1)
        if narrate:
            report.summary = summarize(report)
        return report
