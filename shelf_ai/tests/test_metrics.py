"""
tests/test_metrics.py
---------------------
Unit tests for MetricsCalculator and the new ShelfReport / ProductStock fields.
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detector import Detection, DetectionResult
from src.shelf_analyzer import ShelfAnalyzer, StockStatus
from src.metrics import MetricsCalculator, ShelfMetrics

PLANOGRAM = str(REPO_ROOT / "config" / "planogram.yaml")
THRESHOLDS = str(REPO_ROOT / "config" / "thresholds.yaml")


@pytest.fixture
def analyzer():
    return ShelfAnalyzer(PLANOGRAM, THRESHOLDS)


@pytest.fixture
def calc():
    return MetricsCalculator()


def make_result(detections):
    return DetectionResult(
        detections=detections, image_width=640, image_height=480
    )


# ─── ProductStock.fill_rate ────────────────────────────────────────────────────

class TestProductStockFillRate:
    def test_fill_rate_zero_when_oos(self, analyzer):
        report = analyzer.analyse(make_result([]))
        maggi = next(
            s for s in report.shelf_stocks["shelf_a"] if s.product == "maggi"
        )
        assert maggi.fill_rate == 0.0

    def test_fill_rate_partial(self, analyzer):
        # 2 out of 8 expected → 0.25
        dets = [
            Detection("maggi", 0.9, 0.2, 0.12, 0.05, 0.04),
            Detection("maggi", 0.9, 0.4, 0.12, 0.05, 0.04),
        ]
        report = analyzer.analyse(make_result(dets))
        maggi = next(
            s for s in report.shelf_stocks["shelf_a"] if s.product == "maggi"
        )
        assert abs(maggi.fill_rate - 0.25) < 1e-9

    def test_fill_rate_full(self, analyzer):
        # All 8 expected detected
        dets = [
            Detection("maggi", 0.9, x, 0.12, 0.05, 0.04)
            for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ]
        report = analyzer.analyse(make_result(dets))
        maggi = next(
            s for s in report.shelf_stocks["shelf_a"] if s.product == "maggi"
        )
        assert abs(maggi.fill_rate - 1.0) < 1e-9


# ─── ShelfReport new fields ────────────────────────────────────────────────────

class TestShelfReportFields:
    def test_total_detections_zero(self, analyzer):
        report = analyzer.analyse(make_result([]))
        assert report.total_detections == 0

    def test_total_detections_counted(self, analyzer):
        dets = [
            Detection("maggi", 0.9, 0.2, 0.12, 0.05, 0.04),
            Detection("maggi", 0.9, 0.4, 0.12, 0.05, 0.04),
            Detection("colgate", 0.9, 0.3, 0.37, 0.05, 0.04),
        ]
        report = analyzer.analyse(make_result(dets))
        assert report.total_detections == 3

    def test_overall_fill_rate_zero(self, analyzer):
        report = analyzer.analyse(make_result([]))
        assert report.overall_fill_rate == 0.0

    def test_overall_fill_rate_bounded(self, analyzer):
        # Fill rate should never exceed 1.0
        dets = [
            Detection("maggi", 0.9, x, 0.12, 0.05, 0.04)
            for x in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ]
        report = analyzer.analyse(make_result(dets))
        assert report.overall_fill_rate <= 1.0

    def test_ok_items_property(self, analyzer):
        # Provide full stock for maggi (8/8)
        dets = [
            Detection("maggi", 0.9, x, 0.12, 0.05, 0.04)
            for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ]
        report = analyzer.analyse(make_result(dets))
        ok_products = [s.product for s in report.ok_items]
        assert "maggi" in ok_products


# ─── MetricsCalculator ─────────────────────────────────────────────────────────

class TestMetricsCalculator:
    def test_returns_shelf_metrics(self, analyzer, calc):
        report = analyzer.analyse(make_result([]))
        metrics = calc.compute(report)
        assert isinstance(metrics, ShelfMetrics)

    def test_oos_count_all_empty(self, analyzer, calc):
        report = analyzer.analyse(make_result([]))
        metrics = calc.compute(report)
        # All products across 4 shelves should be OOS
        assert metrics.oos_count > 0

    def test_health_score_range(self, analyzer, calc):
        report = analyzer.analyse(make_result([]))
        metrics = calc.compute(report)
        assert 0.0 <= metrics.health_score <= 100.0

    def test_health_score_improves_with_stock(self, analyzer, calc):
        empty_report = analyzer.analyse(make_result([]))
        empty_metrics = calc.compute(empty_report)

        dets = [
            Detection("maggi", 0.9, x, 0.12, 0.05, 0.04)
            for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ]
        stocked_report = analyzer.analyse(make_result(dets))
        stocked_metrics = calc.compute(stocked_report)

        assert stocked_metrics.health_score > empty_metrics.health_score

    def test_shelf_fill_rates_keys(self, analyzer, calc):
        report = analyzer.analyse(make_result([]))
        metrics = calc.compute(report)
        assert set(metrics.shelf_fill_rates.keys()) == {
            "shelf_a", "shelf_b", "shelf_c", "shelf_d"
        }

    def test_compliance_rate_no_misplacement(self, analyzer, calc):
        # Products on correct shelves → compliance_rate == 1.0
        dets = [Detection("maggi", 0.9, 0.2, 0.12, 0.05, 0.04)]
        report = analyzer.analyse(make_result(dets))
        metrics = calc.compute(report)
        assert metrics.compliance_rate == 1.0

    def test_metrics_summary_string(self, analyzer, calc):
        report = analyzer.analyse(make_result([]))
        metrics = calc.compute(report)
        summary = metrics.summary()
        assert "Fill Rate" in summary
        assert "Health Score" in summary

    def test_as_dict_returns_all_keys(self, analyzer, calc):
        report = analyzer.analyse(make_result([]))
        metrics = calc.compute(report)
        d = metrics.as_dict()
        expected_keys = {
            "overall_fill_rate", "compliance_rate", "health_score",
            "oos_count", "low_stock_count", "ok_count",
            "misplaced_count", "shelf_fill_rates",
        }
        assert expected_keys == set(d.keys())

    def test_as_dict_values_match_attributes(self, analyzer, calc):
        report = analyzer.analyse(make_result([]))
        metrics = calc.compute(report)
        d = metrics.as_dict()
        assert d["overall_fill_rate"] == metrics.overall_fill_rate
        assert d["health_score"] == metrics.health_score
        assert d["oos_count"] == metrics.oos_count
        assert d["shelf_fill_rates"] == dict(metrics.shelf_fill_rates)
