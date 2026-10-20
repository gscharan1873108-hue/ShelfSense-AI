from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels


@dataclass(frozen=True)
class Sku:
    sku: str
    name: str
    short_name: str
    category: str
    price: float
    width_cm: float
    height_cm: float
    body: tuple[int, int, int]  # RGB
    accent: tuple[int, int, int]  # RGB
    mark: str


@dataclass(frozen=True)
class Slot:
    shelf: int
    position: int  # 1-based, left to right
    sku: str
    x0_cm: float
    x1_cm: float
    group: int  # index of the planogram item (contiguous run of one SKU)


@dataclass(frozen=True)
class Shelf:
    index: int
    name: str
    items: tuple[tuple[str, int], ...]  # (sku, facings)

    def slots(self, catalog: dict[str, Sku]) -> list[Slot]:
        out: list[Slot] = []
        x = 0.0
        for g, (sku, facings) in enumerate(self.items):
            w = catalog[sku].width_cm
            for _ in range(facings):
                out.append(Slot(self.index, len(out) + 1, sku, x, x + w, g))
                x += w
        return out


@dataclass(frozen=True)
class Planogram:
    store: str
    shelf_width_cm: float
    shelves: tuple[Shelf, ...]

    def home(self, sku: str, catalog: dict[str, Sku]) -> list[Slot]:
        return [s for sh in self.shelves for s in sh.slots(catalog) if s.sku == sku]


@dataclass
class Detection:
    box: Box
    sku: str | None  # None = unrecognised product
    distance: float
    confidence: float


@dataclass
class SlotResult:
    slot: Slot
    status: str  # ok | misplaced | empty
    found_sku: str | None
    box: Box  # expected slot area in the image
    confidence: float = 0.0


@dataclass
class TagRead:
    box: Box
    raw_text: str
    price: float | None
    matched_sku: str | None
    name_score: float
    ocr_conf: float


@dataclass
class LabelCheck:
    shelf: int
    group: int
    expected_sku: str
    product_under: str | None
    tag: TagRead | None
    status: str  # ok | missing | unreadable | wrong_label | wrong_price
    catalog_price: float
    read_price: float | None


@dataclass
class Issue:
    type: str
    severity: str  # critical | high | medium | low
    shelf: int
    position: int | None
    sku: str | None
    message: str
    action: str
    revenue_at_risk: float = 0.0
    box: Box | None = None


@dataclass
class Report:
    store: str
    image_size: tuple[int, int]
    scores: dict[str, float]
    slots: list[SlotResult] = field(default_factory=list)
    labels: list[LabelCheck] = field(default_factory=list)
    availability: list[dict] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    detections: dict[int, list[Detection]] = field(default_factory=dict)
    shelf_geometry: list = field(default_factory=list)
    summary: str = ""


def load_catalog(path: str | Path) -> dict[str, Sku]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        r["sku"]: Sku(**{**r, "body": tuple(r["body"]), "accent": tuple(r["accent"])})
        for r in rows
    }


def load_planogram(path: str | Path, catalog: dict[str, Sku]) -> Planogram:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    shelves = tuple(
        Shelf(s["index"], s["name"], tuple((sku, int(n)) for sku, n in s["items"]))
        for s in d["shelves"]
    )
    unknown = {sku for s in shelves for sku, _ in s.items} - catalog.keys()
    if unknown:
        raise ValueError(f"Planogram references SKUs missing from catalogue: {sorted(unknown)}")
    return Planogram(d["store"], float(d["shelf_width_cm"]), shelves)
