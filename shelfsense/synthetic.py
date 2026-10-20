"""Procedural shelf renderer: draws packs and, for the demo, whole shelf scenes."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import Planogram, Sku

PX_PER_CM = 10
MARGIN = 40
BOARD_H = 64
SHELF_H = 300
TOP = 30
BACK_PANEL_BGR = (235, 230, 225)
FRAME_BGR = (45, 45, 50)
BOARD_BGR = (70, 110, 150)


def render_pack(sku: Sku, w: int, h: int) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = sku.body[::-1]
    img[: int(h * 0.22)] = sku.accent[::-1]
    img[int(h * 0.85):] = sku.accent[::-1]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (25, 25, 25), 2)
    scale = w / 90
    (tw, th), _ = cv2.getTextSize(sku.mark, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(img, sku.mark, ((w - tw) // 2, int(h * 0.55) + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


@dataclass
class Scenario:
    """Deviations from the planogram. Shelf/position are 1-based."""
    replace: dict[tuple[int, int], str] = field(default_factory=dict)
    empty: set[tuple[int, int]] = field(default_factory=set)
    tag_price: dict[str, float] = field(default_factory=dict)
    tag_sku: dict[str, str] = field(default_factory=dict)  # tag under SKU shows another SKU
    no_tag: set[str] = field(default_factory=set)


SCENARIOS: dict[str, Scenario] = {
    "perfect": Scenario(),
    "low_stock": Scenario(empty={(1, 13), (1, 14), (2, 10), (3, 12), (3, 13)}),
    "messy": Scenario(
        replace={(1, 6): "SS-3003", (2, 9): "SS-2003", (3, 8): "SS-3004"},
        empty={(1, 12), (1, 13), (1, 14), (2, 4), (2, 5), (3, 12), (3, 13)},
        tag_price={"SS-2003": 25.0, "SS-2001": 38.0},
        tag_sku={"SS-3002": "SS-3001"},
        no_tag={"SS-1003"},
    ),
}


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_scene(catalog: dict[str, Sku], plan: Planogram, scenario: Scenario | str = "messy",
                 seed: int = 0, augment: bool = True) -> np.ndarray:
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    n = len(plan.shelves)
    width = int(plan.shelf_width_cm * PX_PER_CM) + 2 * MARGIN
    height = TOP + n * (SHELF_H + BOARD_H) + 20
    img = np.empty((height, width, 3), np.uint8)
    img[:] = FRAME_BGR
    tags: list[tuple[int, int, int, str, str]] = []  # x, y, w, name, price

    for si, shelf in enumerate(plan.shelves):
        top = TOP + si * (SHELF_H + BOARD_H)
        board_y = top + SHELF_H
        img[top:board_y, MARGIN:width - MARGIN] = BACK_PANEL_BGR
        img[board_y:board_y + BOARD_H, MARGIN:width - MARGIN] = BOARD_BGR
        slots = shelf.slots(catalog)
        for s in slots:
            sku_id = sc.replace.get((s.shelf, s.position), s.sku)
            if (s.shelf, s.position) in sc.empty:
                continue
            sku = catalog[sku_id]
            x0 = MARGIN + int(s.x0_cm * PX_PER_CM) + 3
            w = int(sku.width_cm * PX_PER_CM) - 6
            h = int(sku.height_cm * PX_PER_CM)
            img[board_y - h:board_y, x0:x0 + w] = render_pack(sku, w, h)
        seen: set[int] = set()
        for s in slots:
            if s.group in seen:
                continue
            seen.add(s.group)
            if s.sku in sc.no_tag:
                continue
            gw = sum(t.x1_cm - t.x0_cm for t in slots if t.group == s.group) * PX_PER_CM
            shown = catalog[sc.tag_sku.get(s.sku, s.sku)]
            price = sc.tag_price.get(s.sku, shown.price)
            tags.append((MARGIN + int(s.x0_cm * PX_PER_CM) + 4, board_y + 6,
                         int(min(170, gw - 8)), shown.short_name, f"Rs {price:.2f}"))

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    name_font, price_font = _font(15), _font(24)
    for x, y, w, name, price in tags:
        d.rectangle([x, y, x + w, y + BOARD_H - 12], fill=(255, 255, 255), outline=(120, 120, 120))
        d.text((x + 6, y + 3), name, fill=(30, 30, 30), font=name_font)
        d.text((x + 6, y + 22), price, fill=(0, 0, 0), font=price_font)
    img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return camera_effects(img, seed) if augment else img


def camera_effects(img: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) * rng.uniform(0.9, 1.08)
    out += rng.normal(0, 4, out.shape)
    return cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (3, 3), 0)
