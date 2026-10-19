# ShelfSense AI - Autonomous Retail Compliance Agent

Analyses a shelf image or video and reports planogram compliance, label/price errors and availability gaps, with prioritised corrective actions. Runs on CPU only (OpenCV + RapidOCR on ONNX Runtime), no cloud calls.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py          # dashboard
.\.venv\Scripts\python.exe -m shelfsense generate             # demo images + video -> data/samples
.\.venv\Scripts\python.exe -m shelfsense analyze data\samples\messy.jpg --out output   # JSON + HTML + annotated image
.\.venv\Scripts\python.exe -m shelfsense benchmark --threads 1                         # edge latency
.\.venv\Scripts\python.exe -m pytest
```

Use Python 3.12 or older (`rapidocr-onnxruntime` has no wheels for 3.13+).

## Pipeline

| Stage | Module | Method |
|---|---|---|
| Shelf geometry | `layout.py` | Locate shelf-edge boards by colour, derive px/cm from the planogram shelf width |
| Product detection | `detector.py` | Segment packs against the back-panel colour, split merged facings by catalogue width |
| SKU recognition | `detector.py` | Nearest catalogue reference on banded Lab colour features; "unknown" above a distance threshold |
| Placement | `planogram.py` | Match detections to planned slots by x-overlap; ok / misplaced / empty; recommends the product's planogram home |
| Availability | `availability.py` | Per SKU group: out of stock / low stock (>= 50% facings empty) / gap, with replenishment units and value at risk |
| Labels | `labels.py`, `ocr.py` | Detect tags, OCR name + price, fuzzy-match name to catalogue, compare price; flags missing, unreadable, wrong-product and wrong-price labels |
| Scores | `pipeline.py` | Planogram compliance, availability, label accuracy, overall (0.5 / 0.2 / 0.3) and per shelf |
| Output | `report.py`, `visualize.py`, `app.py` | Annotated image, JSON, HTML, Streamlit dashboard |
| Video | `video.py` | Sample frames, analyse each, keep issues present in >= 50% of frames |
| Summary | `narrator.py` | Template by default; set `SHELFSENSE_SLM=qwen2.5:3b` to have a local Ollama small model (<20B) write the manager briefing from the structured facts |

## Demo data

`data/catalog.json` (12 SKUs, prices, dimensions) and `data/planogram.json` (3 shelves, 37 facings) drive everything. `shelfsense/synthetic.py` renders shelf scenes (`perfect`, `low_stock`, `messy`) with camera noise. The `messy` scene contains 3 misplaced products, 1 out-of-stock and 2 low-stock SKUs, 2 wrong prices, 1 wrong label and 1 missing label; all 10 are detected (see `tests/`).

## Assumptions

- One front-facing, roughly fronto-parallel image per planogram bay; fixed camera or shelf-mounted device.
- Shelf-edge board colour is calibrated in `config.py`; the planogram supplies shelf width in cm and slot order.
- Only front facings are visible, so stock depth is assumed (4 units/facing) for replenishment quantities.
- The catalogue is the source of truth for price; a label is correct only if SKU and price both match.
- Price tags are bright rectangles on the shelf-edge board, showing a product name and a price.

## Limitations and next steps

- Results are validated on synthetic scenes only. For real stores, replace `SkuClassifier` and the segmentation in `detect_products` with a trained detector (e.g. YOLO exported to ONNX) and add reference pack photos in a folder passed as `ref_dir`; the rest of the pipeline consumes `Detection` objects unchanged.
- OCR is RapidOCR; small or angled tags will need a perspective correction step.
- The Dockerfile (`--cpus` / `--memory` limits) is provided for edge simulation but has not been built or run here. A single-thread CPU run of the demo frame takes roughly 0.6 s.
