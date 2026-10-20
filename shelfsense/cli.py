"""Command line: generate demo data, analyse an image/video, benchmark latency."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2

from . import DATA_DIR
from .models import load_catalog, load_planogram
from .pipeline import ShelfSenseAnalyzer
from .report import to_dict, to_html
from .synthetic import SCENARIOS, render_scene
from .video import analyze_video
from .visualize import annotate

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def _analyzer(args) -> ShelfSenseAnalyzer:
    catalog = load_catalog(args.catalog)
    return ShelfSenseAnalyzer(catalog, load_planogram(args.planogram, catalog))


def cmd_generate(args) -> None:
    catalog = load_catalog(args.catalog)
    plan = load_planogram(args.planogram, catalog)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in SCENARIOS:
        cv2.imwrite(str(out / f"{name}.jpg"), render_scene(catalog, plan, name, seed=1))
    frames = [render_scene(catalog, plan, "messy", seed=s) for s in range(8)]
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out / "messy.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"Wrote {len(SCENARIOS)} images and messy.mp4 to {out}")


def cmd_analyze(args) -> None:
    analyzer = _analyzer(args)
    path = Path(args.path)
    if path.suffix.lower() in VIDEO_EXT:
        report, img, n = analyze_video(analyzer, str(path))
        print(f"Analysed {n} frames")
    else:
        img = cv2.imread(str(path))
        if img is None:
            raise SystemExit(f"Cannot read image: {path}")
        report = analyzer.analyze(img)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    annotated = annotate(img, report)
    cv2.imwrite(str(out / "annotated.jpg"), annotated)
    (out / "report.json").write_text(json.dumps(to_dict(report), indent=2), encoding="utf-8")
    (out / "report.html").write_text(to_html(report, annotated), encoding="utf-8")
    print(report.summary)
    for i in report.issues:
        print(f"[{i.severity:8}] shelf {i.shelf} pos {i.position or '-':>2}: {i.message}")
    print(f"Outputs in {out}  (timings: {report.timings_ms} ms)")


def cmd_benchmark(args) -> None:
    cv2.setNumThreads(args.threads)
    analyzer = _analyzer(args)
    img = render_scene(analyzer.catalog, analyzer.plan, "messy")
    analyzer.analyze(img, narrate=False)  # warm-up
    times = []
    for _ in range(args.runs):
        t = time.perf_counter()
        analyzer.analyze(img, narrate=False)
        times.append((time.perf_counter() - t) * 1000)
    print(f"{args.runs} runs, {args.threads} cv thread(s): median {statistics.median(times):.0f} ms, "
          f"max {max(times):.0f} ms per {img.shape[1]}x{img.shape[0]} frame")


def main() -> None:
    p = argparse.ArgumentParser(prog="shelfsense")
    p.add_argument("--catalog", default=str(DATA_DIR / "catalog.json"))
    p.add_argument("--planogram", default=str(DATA_DIR / "planogram.json"))
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write demo shelf images and a video")
    g.add_argument("--out", default=str(DATA_DIR / "samples"))
    g.set_defaults(fn=cmd_generate)
    a = sub.add_parser("analyze", help="analyse an image or video")
    a.add_argument("path")
    a.add_argument("--out", default="output")
    a.set_defaults(fn=cmd_analyze)
    b = sub.add_parser("benchmark", help="measure per-frame latency")
    b.add_argument("--runs", type=int, default=10)
    b.add_argument("--threads", type=int, default=1)
    b.set_defaults(fn=cmd_benchmark)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
