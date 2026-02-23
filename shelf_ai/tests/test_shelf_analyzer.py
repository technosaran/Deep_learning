"""
tests/test_shelf_analyzer.py
-----------------------------
Unit tests for ShelfAnalyzer and PlanogramChecker.
These tests do NOT require model weights or a GPU.
"""

import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detector import Detection, DetectionResult
from src.shelf_analyzer import ShelfAnalyzer, StockStatus
from src.planogram import PlanogramChecker

PLANOGRAM = str(REPO_ROOT / "config" / "planogram.yaml")
THRESHOLDS = str(REPO_ROOT / "config" / "thresholds.yaml")


@pytest.fixture
def analyzer():
    return ShelfAnalyzer(PLANOGRAM, THRESHOLDS)


@pytest.fixture
def checker():
    return PlanogramChecker(PLANOGRAM)


def make_result(detections):
    return DetectionResult(
        detections=detections, image_width=640, image_height=480
    )


# ─── Stock Status Tests ────────────────────────────────────────────────────────

class TestStockStatus:
    def test_out_of_stock_when_zero(self, analyzer):
        # No maggi detected on shelf_a
        result = make_result([])
        report = analyzer.analyse(result)
        maggi_stock = next(
            s for s in report.shelf_stocks["shelf_a"] if s.product == "maggi"
        )
        assert maggi_stock.status == StockStatus.OUT_OF_STOCK
        assert maggi_stock.detected_count == 0

    def test_low_stock_below_threshold(self, analyzer):
        # Expected 8 maggi; 2 detected → 25% ≤ 40% threshold → Low Stock
        dets = [
            Detection("maggi", 0.9, 0.2, 0.12, 0.05, 0.04),
            Detection("maggi", 0.9, 0.4, 0.12, 0.05, 0.04),
        ]
        report = analyzer.analyse(make_result(dets))
        maggi_stock = next(
            s for s in report.shelf_stocks["shelf_a"] if s.product == "maggi"
        )
        assert maggi_stock.status == StockStatus.LOW_STOCK
        assert maggi_stock.detected_count == 2

    def test_ok_stock_above_threshold(self, analyzer):
        # Expected 8; 6 detected → 75% > 40% → OK
        dets = [
            Detection("maggi", 0.9, x, 0.12, 0.05, 0.04)
            for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        ]
        report = analyzer.analyse(make_result(dets))
        maggi_stock = next(
            s for s in report.shelf_stocks["shelf_a"] if s.product == "maggi"
        )
        assert maggi_stock.status == StockStatus.OK
        assert maggi_stock.detected_count == 6


# ─── Shelf Zone Assignment ─────────────────────────────────────────────────────

class TestShelfZone:
    def test_product_assigned_to_correct_shelf(self, analyzer):
        # colgate at y=0.37 → shelf_b
        dets = [Detection("colgate", 0.9, 0.3, 0.37, 0.05, 0.04)]
        report = analyzer.analyse(make_result(dets))
        colgate_stock = next(
            s for s in report.shelf_stocks["shelf_b"] if s.product == "colgate"
        )
        assert colgate_stock.detected_count == 1

    def test_product_at_bottom_of_image(self, analyzer):
        # atta at y=0.95 → shelf_d
        dets = [Detection("atta", 0.9, 0.5, 0.95, 0.05, 0.04)]
        report = analyzer.analyse(make_result(dets))
        atta_stock = next(
            s for s in report.shelf_stocks["shelf_d"] if s.product == "atta"
        )
        assert atta_stock.detected_count == 1


# ─── Misplacement Detection ────────────────────────────────────────────────────

class TestMisplacement:
    def test_misplaced_product_detected(self, analyzer):
        # dove (expected shelf_b) detected at y=0.12 → shelf_a
        dets = [Detection("dove", 0.9, 0.4, 0.12, 0.05, 0.04)]
        report = analyzer.analyse(make_result(dets))
        # At least one misplaced entry for dove
        misplaced_products = [m[0] for m in report.misplaced]
        assert "dove" in misplaced_products

    def test_no_misplacement_when_correct(self, analyzer):
        # dove at y=0.37 → shelf_b (correct)
        dets = [Detection("dove", 0.9, 0.3, 0.37, 0.05, 0.04)]
        report = analyzer.analyse(make_result(dets))
        misplaced_products = [m[0] for m in report.misplaced]
        assert "dove" not in misplaced_products

    def test_compliance_report_has_issues(self, checker):
        misplaced = [("dove", "shelf_a", "shelf_b")]
        compliance = checker.check(misplaced)
        assert not compliance.is_compliant
        assert len(compliance.issues) == 1
        issue = compliance.issues[0]
        assert issue.product == "dove"
        assert issue.detected_shelf_id == "shelf_a"
        assert issue.expected_shelf_id == "shelf_b"

    def test_compliance_report_clean(self, checker):
        compliance = checker.check([])
        assert compliance.is_compliant


# ─── Report Aggregation ────────────────────────────────────────────────────────

class TestShelfReport:
    def test_action_required_when_issues(self, analyzer):
        # No detections → everything out of stock
        report = analyzer.analyse(make_result([]))
        assert report.action_required

    def test_out_of_stock_list(self, analyzer):
        report = analyzer.analyse(make_result([]))
        oos = report.out_of_stock_items
        # All products across all shelves should be out of stock
        assert len(oos) > 0
        for item in oos:
            assert item.status == StockStatus.OUT_OF_STOCK

    def test_low_stock_items_collected(self, analyzer):
        dets = [
            Detection("maggi", 0.9, 0.2, 0.12, 0.05, 0.04),
            Detection("maggi", 0.9, 0.4, 0.12, 0.05, 0.04),
        ]
        report = analyzer.analyse(make_result(dets))
        low = report.low_stock_items
        maggi_in_low = any(i.product == "maggi" for i in low)
        assert maggi_in_low
