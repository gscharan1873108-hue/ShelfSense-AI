import pytest

from shelfsense import load_defaults
from shelfsense.labels import match_name, parse_price
from shelfsense.ocr import default_engine
from shelfsense.pipeline import ShelfSenseAnalyzer
from shelfsense.synthetic import render_scene

CATALOG, PLAN = load_defaults()
OCR = default_engine()
needs_ocr = pytest.mark.skipif(OCR is None, reason="rapidocr-onnxruntime not installed")


@pytest.fixture(scope="module")
def analyzer():
    return ShelfSenseAnalyzer(CATALOG, PLAN, ocr=OCR)


def kinds(report):
    return {(i.type, i.shelf, i.position) for i in report.issues}


@pytest.mark.parametrize("text,price", [("Rs 40.00", 40.0), ("RS 125,50", 125.5), ("Rs.5O.00", 50.0),
                                         ("$12", 12.0), ("Zest Cola 1L", None), ("", None)])
def test_parse_price(text, price):
    assert parse_price(text) == price


def test_match_name_prefers_exact():
    assert match_name("ZestCola", CATALOG)[0] == "SS-1001"
    assert match_name("Zest Cola Zero", CATALOG)[0] == "SS-1002"


@needs_ocr
def test_perfect_shelf_is_fully_compliant(analyzer):
    r = analyzer.analyze(render_scene(CATALOG, PLAN, "perfect", seed=3))
    assert r.scores["planogram_compliance"] == 100
    assert r.scores["label_accuracy"] == 100
    assert not r.issues


@needs_ocr
def test_messy_shelf_issues_detected(analyzer):
    r = analyzer.analyze(render_scene(CATALOG, PLAN, "messy", seed=5))
    got = kinds(r)
    assert {("misplaced", 1, 6), ("misplaced", 2, 9), ("misplaced", 3, 8)} <= got
    assert ("out_of_stock", 2, 4) in got
    assert ("low_stock", 1, 12) in got and ("low_stock", 3, 12) in got
    assert ("price_mismatch", 2, 1) in got and ("price_mismatch", 2, 6) in got
    assert ("label_mismatch", 3, 5) in got
    assert ("missing_label", 1, 8) in got
    assert len(r.issues) == 10
    assert r.scores["planogram_compliance"] < 100


def test_placement_without_ocr(analyzer):
    a = ShelfSenseAnalyzer(CATALOG, PLAN, ocr=None)
    r = a.analyze(render_scene(CATALOG, PLAN, "low_stock", seed=2))
    assert r.scores["planogram_compliance"] == pytest.approx(100 * (37 - 5) / 37, abs=0.2)
    assert any("OCR" in w for w in r.warnings)
    assert {i.type for i in r.issues} == {"low_stock"}


def test_misplacement_recommends_home_location(analyzer):
    r = ShelfSenseAnalyzer(CATALOG, PLAN, ocr=None).analyze(render_scene(CATALOG, PLAN, "messy", seed=1))
    mis = next(i for i in r.issues if i.type == "misplaced" and i.shelf == 1)
    assert "shelf 3, positions 7-10" in mis.action
