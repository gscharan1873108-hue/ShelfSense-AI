"""Shelf availability: empty facings, low stock and replenishment quantities."""
from __future__ import annotations

from .config import Config
from .models import Issue, Shelf, Sku, SlotResult

_SEVERITY = {"out_of_stock": "critical", "low_stock": "high", "gap": "medium"}


def availability(results: list[SlotResult], shelf: Shelf, catalog: dict[str, Sku],
                 cfg: Config) -> tuple[list[dict], list[Issue]]:
    rows, issues = [], []
    for g in sorted({r.slot.group for r in results}):
        grp = [r for r in results if r.slot.group == g]
        sku = catalog[grp[0].slot.sku]
        empty = [r for r in grp if r.status == "empty"]
        n, e = len(grp), len(empty)
        if e == n:
            status = "out_of_stock"
        elif e / n >= cfg.low_stock_ratio:
            status = "low_stock"
        elif e:
            status = "gap"
        else:
            status = "ok"
        gap_cm = sum(r.slot.x1_cm - r.slot.x0_cm for r in empty)
        rows.append({"shelf": shelf.index, "sku": sku.sku, "name": sku.name, "planned_facings": n,
                     "empty_facings": e, "status": status, "gap_cm": round(gap_cm, 1)})
        if status == "ok":
            continue
        units = e * cfg.facing_depth
        label = {"out_of_stock": "Out of stock", "low_stock": "Low stock", "gap": "Empty space"}[status]
        issues.append(Issue(
            status, _SEVERITY[status], shelf.index, empty[0].slot.position, sku.sku,
            f"{label}: {sku.name} has {e} of {n} facings empty ({gap_cm:.0f} cm gap).",
            f"Replenish about {units} units of {sku.name} (shelf {shelf.index}, "
            f"position{'s' if e > 1 else ''} {', '.join(str(r.slot.position) for r in empty)}).",
            sku.price * units, (empty[0].box[0], empty[0].box[1], empty[-1].box[2], empty[-1].box[3])))
    return rows, issues
