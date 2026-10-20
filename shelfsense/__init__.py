from pathlib import Path

from .models import load_catalog, load_planogram

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_defaults():
    catalog = load_catalog(DATA_DIR / "catalog.json")
    return catalog, load_planogram(DATA_DIR / "planogram.json", catalog)
