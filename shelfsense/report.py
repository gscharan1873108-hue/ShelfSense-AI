"""JSON and self-contained HTML reports."""
from __future__ import annotations

import base64
import html
from dataclasses import asdict

import cv2
import numpy as np

from .models import Report

ASSUMPTIONS = [
    "Front-facing, roughly fronto-parallel images of one planogram bay; one image/frame per bay.",
    "Shelf-edge boards have a known, calibratable colour; the planogram gives the shelf width in cm.",
    "Slot positions are derived from SKU widths in the catalogue; products shifted by < 50% of a slot still match.",
    "Only front facings are visible: stock depth is assumed (4 units per facing) when estimating replenishment.",
    "Catalogue prices are the source of truth; a price tag is OK only if SKU and price both match.",
]


def to_dict(report: Report) -> dict:
    return {
        "store": report.store,
        "image_size": report.image_size,
        "scores": report.scores,
        "summary": report.summary,
        "issues": [asdict(i) for i in report.issues],
        "availability": report.availability,
        "labels": [{
            "shelf": c.shelf, "expected_sku": c.expected_sku, "product_under_label": c.product_under,
            "label_text": c.tag.raw_text if c.tag else None, "label_sku": c.tag.matched_sku if c.tag else None,
            "read_price": c.read_price, "catalogue_price": c.catalog_price, "status": c.status,
        } for c in report.labels],
        "slots": [{"shelf": r.slot.shelf, "position": r.slot.position, "expected_sku": r.slot.sku,
                   "found_sku": r.found_sku, "status": r.status} for r in report.slots],
        "warnings": report.warnings,
        "timings_ms": report.timings_ms,
        "assumptions": ASSUMPTIONS,
    }


def to_html(report: Report, annotated: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img64 = base64.b64encode(buf.tobytes()).decode()
    e = html.escape
    cards = "".join(
        f"<div class=card><b>{v:.0f}{'/100' if k == 'overall' else '%'}</b><span>{e(k.replace('_', ' '))}</span></div>"
        for k, v in report.scores.items() if not k.startswith("shelf_"))
    rows = "".join(
        f"<tr class={i.severity}><td>{i.severity}</td><td>{i.shelf}</td><td>{i.position or ''}</td>"
        f"<td>{e(i.message)}</td><td>{e(i.action)}</td><td>{i.revenue_at_risk:.0f}</td></tr>"
        for i in report.issues)
    av = "".join(
        f"<tr><td>{r['shelf']}</td><td>{e(r['name'])}</td>"
        f"<td>{r['planned_facings'] - r['empty_facings']}/{r['planned_facings']}</td>"
        f"<td>{r['status'].replace('_', ' ')}</td></tr>" for r in report.availability)
    return f"""<!doctype html><meta charset=utf-8><title>ShelfSense AI report</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1200px}}.cards{{display:flex;gap:12px;margin:12px 0}}
.card{{border:1px solid #ccc;border-radius:8px;padding:12px 18px}}.card b{{font-size:28px;display:block}}
table{{border-collapse:collapse;width:100%;margin:8px 0 24px}}td,th{{border:1px solid #ddd;padding:6px;text-align:left}}
tr.critical td:first-child{{background:#e53935;color:#fff}}tr.high td:first-child{{background:#fb8c00;color:#fff}}
tr.medium td:first-child{{background:#fdd835}}img{{max-width:100%}}</style>
<h1>ShelfSense AI - {e(report.store)}</h1><p>{e(report.summary)}</p>
<div class=cards>{cards}</div><img src="data:image/jpeg;base64,{img64}">
<h2>Corrective actions (priority order)</h2>
<table><tr><th>Severity<th>Shelf<th>Pos<th>Issue<th>Action<th>Value at risk (Rs)</tr>{rows}</table>
<h2>Availability</h2><table><tr><th>Shelf<th>Product<th>Facings filled<th>Status</tr>{av}</table>
<h2>Assumptions</h2><ul>{''.join(f'<li>{e(a)}</li>' for a in ASSUMPTIONS)}</ul>"""
