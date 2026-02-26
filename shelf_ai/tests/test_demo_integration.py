"""
tests/test_demo_integration.py
-------------------------------
Tests for the integrated demo pipeline:
  - print_metrics helper
  - DetectionSmoother integration in webcam count extraction
  - MetricsCalculator / StockHistory integration in demo mode
  - parse_args --smoother-window flag
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detector import Detection, DetectionResult
from src.shelf_analyzer import ShelfAnalyzer
from src.planogram import PlanogramChecker
from src.metrics import MetricsCalculator, ShelfMetrics
from src.history import StockHistory
from src.smoother import DetectionSmoother

PLANOGRAM = str(REPO_ROOT / "config" / "planogram.yaml")
THRESHOLDS = str(REPO_ROOT / "config" / "thresholds.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(detections):
    return DetectionResult(detections=detections, image_width=640, image_height=480)


@pytest.fixture
def analyzer():
    return ShelfAnalyzer(PLANOGRAM, THRESHOLDS)


# ---------------------------------------------------------------------------
# print_metrics
# ---------------------------------------------------------------------------

class TestPrintMetrics:
    def test_print_metrics_outputs_summary(self, capsys):
        from demo import print_metrics

        metrics = MetricsCalculator().compute(
            ShelfAnalyzer(PLANOGRAM, THRESHOLDS).analyse(make_result([]))
        )
        print_metrics(metrics)
        captured = capsys.readouterr()
        assert "Fill Rate" in captured.out
        assert "Health Score" in captured.out
        assert "Out of Stock" in captured.out

    def test_print_metrics_shows_health_score_value(self, capsys):
        from demo import print_metrics

        report = ShelfAnalyzer(PLANOGRAM, THRESHOLDS).analyse(make_result([
            Detection("maggi", 0.9, x_pos, 0.12, 0.05, 0.04)
            for x_pos in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        ]))
        metrics = MetricsCalculator().compute(report)
        print_metrics(metrics)
        captured = capsys.readouterr()
        assert f"{metrics.health_score:.1f}" in captured.out


# ---------------------------------------------------------------------------
# parse_args --smoother-window
# ---------------------------------------------------------------------------

class TestDemoParseArgs:
    def _parse(self, *args):
        from demo import parse_args
        return parse_args(list(args))

    def test_smoother_window_default(self):
        args = self._parse("--demo")
        assert args.smoother_window == 5

    def test_smoother_window_custom(self):
        args = self._parse("--demo", "--smoother-window", "10")
        assert args.smoother_window == 10

    def test_smoother_window_one(self):
        args = self._parse("--demo", "--smoother-window", "1")
        assert args.smoother_window == 1


# ---------------------------------------------------------------------------
# Webcam count-extraction logic (unit test without CV2)
# ---------------------------------------------------------------------------

class TestWebcamCountExtraction:
    """Test the count-extraction pattern used in run_webcam."""

    def test_count_extraction_from_detection_result(self):
        dets = [
            Detection("maggi", 0.9, 0.2, 0.12, 0.05, 0.04),
            Detection("maggi", 0.9, 0.4, 0.12, 0.05, 0.04),
            Detection("colgate", 0.9, 0.3, 0.37, 0.05, 0.04),
        ]
        result = make_result(dets)
        raw_counts: dict[str, int] = {}
        for det in result.detections:
            raw_counts[det.label] = raw_counts.get(det.label, 0) + 1
        assert raw_counts == {"maggi": 2, "colgate": 1}

    def test_smoother_receives_extracted_counts(self):
        smoother = DetectionSmoother(window=3)
        dets = [
            Detection("maggi", 0.9, 0.2, 0.12, 0.05, 0.04),
            Detection("maggi", 0.9, 0.4, 0.12, 0.05, 0.04),
        ]
        result = make_result(dets)
        raw_counts: dict[str, int] = {}
        for det in result.detections:
            raw_counts[det.label] = raw_counts.get(det.label, 0) + 1
        smoothed = smoother.update(raw_counts)
        assert "maggi" in smoothed
        assert smoothed["maggi"] == 2

    def test_smoother_smooths_across_multiple_frames(self):
        smoother = DetectionSmoother(window=3)
        # Frame 1: 6 maggi
        r1 = {"maggi": 6}
        # Frame 2: 0 maggi (missed detection)
        r2 = {}
        # Frame 3: 6 maggi
        r3 = {"maggi": 6}
        smoother.update(r1)
        smoother.update(r2)
        result = smoother.update(r3)
        # mean([6, 0, 6]) = 4
        assert result["maggi"] == 4

    def test_empty_frame_does_not_crash_smoother(self):
        smoother = DetectionSmoother(window=5)
        result = make_result([])
        raw_counts: dict[str, int] = {}
        for det in result.detections:
            raw_counts[det.label] = raw_counts.get(det.label, 0) + 1
        smoothed = smoother.update(raw_counts)
        assert smoothed == {}


# ---------------------------------------------------------------------------
# MetricsCalculator + StockHistory integration (as used in run_webcam)
# ---------------------------------------------------------------------------

class TestMetricsHistoryIntegration:
    def test_metrics_recorded_in_history(self, analyzer):
        history = StockHistory(max_entries=10)
        calc = MetricsCalculator()
        report = analyzer.analyse(make_result([]))
        metrics = calc.compute(report)
        history.record(metrics)
        assert len(history) == 1
        assert history.latest.overall_fill_rate == metrics.overall_fill_rate

    def test_health_trend_grows_over_multiple_analyses(self, analyzer):
        history = StockHistory(max_entries=10)
        calc = MetricsCalculator()
        for _ in range(3):
            report = analyzer.analyse(make_result([]))
            metrics = calc.compute(report)
            history.record(metrics)
        trend = history.health_score_trend()
        assert len(trend) == 3

    def test_trend_display_shows_last_five(self, analyzer):
        history = StockHistory(max_entries=10)
        calc = MetricsCalculator()
        for _ in range(7):
            metrics = calc.compute(analyzer.analyse(make_result([])))
            history.record(metrics)
        trend = history.health_score_trend()
        display = trend[-5:]
        assert len(display) == 5
