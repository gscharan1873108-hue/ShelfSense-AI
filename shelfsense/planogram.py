"""Planogram compliance: match detections to planned slots and raise placement issues."""
from __future__ import annotations

from .config import Config
from .layout import ShelfGeometry
from .models import Detection, Issue, Planogram, Shelf, Sku, Slot, SlotResult


def slot_box(slot: Slot, g: ShelfGeometry) -> tuple[int, int, int, int]:
    return (int(g.x0 + slot.x0_cm * g.px_per_cm), g.y0, int(g.x0 + slot.x1_cm * g.px_per_cm), g.y1)


def match_slots(slots: list[Slot], dets: list[Detection], g: ShelfGeometry,
                cfg: Config) -> tuple[list[SlotResult], list[Detection]]:
    pairs = []
    for si, s in enumerate(slots):
        sb = slot_box(s, g)
        for di, d in enumerate(dets):
            ov = min(sb[2], d.box[2]) - max(sb[0], d.box[0])
            frac = ov / min(sb[2] - sb[0], d.box[2] - d.box[0])
            if frac >= cfg.slot_overlap:
                pairs.append((frac, si, di))
    taken_s: dict[int, int] = {}
    taken_d: set[int] = set()
    for frac, si, di in sorted(pairs, reverse=True):
        if si not in taken_s and di not in taken_d:
            taken_s[si] = di
            taken_d.add(di)

    results = []
    for si, s in enumerate(slots):
        sb = slot_box(s, g)
        if si in taken_s:
            d = dets[taken_s[si]]
            results.append(SlotResult(s, "ok" if d.sku == s.sku else "misplaced", d.sku, sb, d.confidence))
        else:
            results.append(SlotResult(s, "empty", None, sb))
    return results, [d for i, d in enumerate(dets) if i not in taken_d]


def _span(slots: list[Slot]) -> str:
    a, b = min(s.position for s in slots), max(s.position for s in slots)
    return f"position {a}" if a == b else f"positions {a}-{b}"


def placement_issues(results: list[SlotResult], extras: list[Detection], shelf: Shelf,
                     plan: Planogram, catalog: dict[str, Sku], cfg: Config) -> list[Issue]:
    issues = []
    for r in results:
        if r.status != "misplaced":
            continue
        exp = catalog[r.slot.sku]
        if r.found_sku is None:
            msg = f"Unrecognised item at position {r.slot.position}; planned: {exp.name}."
            act = f"Check the item and place {exp.name} here."
        else:
            found = catalog[r.found_sku]
            homes = plan.home(r.found_sku, catalog)
            by_shelf = {s.shelf: [h for h in homes if h.shelf == s.shelf] for s in homes}
            where = "; ".join(f"shelf {k}, {_span(v)}" for k, v in sorted(by_shelf.items()))
            msg = f"{found.name} found at position {r.slot.position}; planned: {exp.name}."
            act = f"Move {found.name} to {where} and restock {exp.name} here."
        issues.append(Issue("misplaced", "medium", shelf.index, r.slot.position, r.found_sku or r.slot.sku,
                            msg, act, exp.price * cfg.facing_depth, r.box))
    for d in extras:
        issues.append(Issue("unplanned_item", "low", shelf.index, None, d.sku,
                            "Product detected outside any planned slot.", "Check the item and re-face the shelf.",
                            box=d.box))
    return issues
