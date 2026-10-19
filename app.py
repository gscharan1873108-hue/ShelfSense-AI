"""ShelfSense AI dashboard:  streamlit run app.py"""
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from shelfsense import load_defaults
from shelfsense.pipeline import ShelfSenseAnalyzer
from shelfsense.report import ASSUMPTIONS, to_dict, to_html
from shelfsense.synthetic import SCENARIOS, render_scene
from shelfsense.video import analyze_video
from shelfsense.visualize import annotate

st.set_page_config(page_title="ShelfSense AI", layout="wide")


@st.cache_resource
def get_analyzer():
    catalog, plan = load_defaults()
    return ShelfSenseAnalyzer(catalog, plan)


analyzer = get_analyzer()
catalog = analyzer.catalog

st.title("ShelfSense AI - Autonomous Retail Compliance Agent")
st.caption(f"Store: {analyzer.plan.store}  |  runs fully on CPU, no cloud calls")

with st.sidebar:
    src = st.radio("Input", ["Demo scenario", "Upload image", "Upload video"])
    img, report, frames = None, None, None
    if src == "Demo scenario":
        name = st.selectbox("Scenario", list(SCENARIOS), index=list(SCENARIOS).index("messy"))
        seed = st.number_input("Camera noise seed", 0, 99, 1)
        img = render_scene(catalog, analyzer.plan, name, seed=int(seed))
    elif src == "Upload image":
        up = st.file_uploader("Shelf image", type=["jpg", "jpeg", "png"])
        if up:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    else:
        up = st.file_uploader("Shelf video", type=["mp4", "avi", "mov", "mkv"])
        every = st.slider("Sample every N seconds", 0.5, 5.0, 1.0, 0.5)
        if up:
            with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as f:
                f.write(up.read())
            try:
                report, img, frames = analyze_video(analyzer, f.name, every_s=every)
            finally:
                Path(f.name).unlink(missing_ok=True)
    st.markdown("**Catalogue**")
    st.dataframe(pd.DataFrame([{"SKU": s.sku, "Name": s.name, "Price": s.price} for s in catalog.values()]),
                 hide_index=True, height=250)

if img is None:
    st.info("Choose a demo scenario or upload a shelf image/video.")
    st.stop()

if report is None:
    with st.spinner("Analysing shelf..."):
        report = analyzer.analyze(img)
if frames:
    st.success(f"Video: {frames} frames analysed; showing issues present in at least half of them.")
annotated = annotate(img, report)

for w in report.warnings:
    st.warning(w)
st.info(report.summary)

s = report.scores
cols = st.columns(4)
cols[0].metric("Planogram compliance", f"{s['planogram_compliance']:.0f}%")
cols[1].metric("Availability", f"{s['availability']:.0f}%")
cols[2].metric("Label accuracy", f"{s['label_accuracy']:.0f}%" if "label_accuracy" in s else "n/a")
cols[3].metric("Overall shelf health", f"{s['overall']:.0f}/100")

left, right = st.columns([3, 2])
left.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")
with right:
    st.subheader("Corrective actions")
    if not report.issues:
        st.success("No issues detected.")
    icon = {"critical": "[CRITICAL]", "high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}
    for i in report.issues:
        with st.expander(f"{icon[i.severity]} Shelf {i.shelf} - {i.type.replace('_', ' ')}"):
            st.write(i.message)
            st.write(f"**Action:** {i.action}")
            if i.revenue_at_risk:
                st.caption(f"Estimated value at risk: Rs {i.revenue_at_risk:.0f}")

t1, t2, t3, t4, t5 = st.tabs(["Planogram", "Availability", "Labels & prices", "Export", "Assumptions"])
with t1:
    sym = {"ok": "OK", "misplaced": "WRONG", "empty": "EMPTY"}
    for shelf in analyzer.plan.shelves:
        rs = [r for r in report.slots if r.slot.shelf == shelf.index]
        st.markdown(f"**Shelf {shelf.index} - {shelf.name}**  ({s[f'shelf_{shelf.index}_planogram']:.0f}% compliant)")
        st.dataframe(pd.DataFrame([{
            "Pos": r.slot.position, "Expected": catalog[r.slot.sku].short_name,
            "Found": catalog[r.found_sku].short_name if r.found_sku else "-", "Status": sym[r.status]} for r in rs]).astype(str).T,
            width="stretch")
with t2:
    st.dataframe(pd.DataFrame(report.availability), hide_index=True, width="stretch")
with t3:
    st.dataframe(pd.DataFrame([{
        "Shelf": c.shelf, "Expected product": catalog[c.expected_sku].name,
        "Product under label": catalog[c.product_under].short_name if c.product_under else "-",
        "Label reads": c.tag.raw_text if c.tag else "(no label)",
        "Read price": c.read_price, "Catalogue price": c.catalog_price, "Status": c.status} for c in report.labels]),
        hide_index=True, width="stretch")
with t4:
    st.download_button("Download JSON report", json.dumps(to_dict(report), indent=2), "shelfsense_report.json")
    st.download_button("Download HTML report", to_html(report, annotated), "shelfsense_report.html")
    st.caption(f"Latency (ms): {report.timings_ms}")
with t5:
    st.markdown("\n".join(f"- {a}" for a in ASSUMPTIONS))
