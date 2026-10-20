from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Shelf-edge board colour (BGR) used to locate shelf rows; calibrate per store.
    board_bgr: tuple = (70, 110, 150)
    board_tol: int = 60
    min_board_row_frac: float = 0.3  # tags can hide over half of a board row
    # Product segmentation against the back-panel colour.
    fg_tol: int = 45
    min_product_w_cm: float = 3.0
    min_product_h_cm: float = 6.0
    # Classifier: Lab distance above which a product is "unknown".
    unknown_dist: float = 35.0
    # Slot matching: min x-overlap (fraction of the smaller box) to claim a slot.
    slot_overlap: float = 0.5
    # Availability.
    low_stock_ratio: float = 0.5
    facing_depth: int = 4  # assumed units behind each facing
    # Labels.
    price_tol: float = 0.01
    ocr_min_conf: float = 0.6
    name_match_min: float = 0.6
    # Overall score weights.
    w_planogram: float = 0.5
    w_availability: float = 0.2
    w_labels: float = 0.3


DEFAULT = Config()
