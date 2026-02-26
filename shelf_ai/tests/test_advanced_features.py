"""
tests/test_advanced_features.py
--------------------------------
Tests for DetectionSmoother and RestockPlanner.
No model weights or GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.smoother import DetectionSmoother
from src.restock import RestockPlanner, RestockTask
from src.detector import Detection, DetectionResult
from src.shelf_analyzer import ShelfAnalyzer, StockStatus

PLANOGRAM = str(REPO_ROOT / "config" / "planogram.yaml")
THRESHOLDS = str(REPO_ROOT / "config" / "thresholds.yaml")


def make_result(detections):
    return DetectionResult(
        detections=detections, image_width=640, image_height=480
    )


@pytest.fixture
def analyzer():
    return ShelfAnalyzer(PLANOGRAM, THRESHOLDS)


@pytest.fixture
def planner():
    return RestockPlanner()


# ===========================================================================
# DetectionSmoother
# ===========================================================================

class TestDetectionSmoother:

    def test_window_stored(self):
        smoother = DetectionSmoother(window=3)
        assert smoother.window == 3

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            DetectionSmoother(window=0)

    def test_single_frame_passthrough(self):
        smoother = DetectionSmoother(window=5)
        result = smoother.update({"maggi": 4, "lays": 2})
        assert result["maggi"] == 4
        assert result["lays"] == 2

    def test_smoothing_averages_frames(self):
        smoother = DetectionSmoother(window=3)
        smoother.update({"maggi": 0})
        smoother.update({"maggi": 3})
        result = smoother.update({"maggi": 3})
        # mean([0, 3, 3]) = 2.0 → rounded to 2
        assert result["maggi"] == 2

    def test_missing_product_treated_as_zero(self):
        smoother = DetectionSmoother(window=4)
        smoother.update({"maggi": 4})
        smoother.update({"maggi": 4})
        # maggi missing this frame → treated as 0
        result = smoother.update({})
        # mean([4, 4, 0]) = 2.67 → rounded to 3
        assert result["maggi"] == 3

    def test_new_product_appears_mid_stream(self):
        smoother = DetectionSmoother(window=3)
        smoother.update({"maggi": 2})
        result = smoother.update({"maggi": 2, "lays": 6})
        # lays first appears in frame 2: buffer = [6], mean = 6
        assert "lays" in result
        assert result["lays"] == 6

    def test_window_clips_old_frames(self):
        smoother = DetectionSmoother(window=2)
        smoother.update({"maggi": 10})  # frame 1 – drops off after window=2
        smoother.update({"maggi": 2})
        result = smoother.update({"maggi": 2})
        # Only last 2 frames: mean([2, 2]) = 2
        assert result["maggi"] == 2

    def test_reset_clears_history(self):
        smoother = DetectionSmoother(window=5)
        smoother.update({"maggi": 8})
        smoother.reset()
        result = smoother.update({"maggi": 1})
        # After reset, only one frame in buffer
        assert result["maggi"] == 1

    def test_empty_update_returns_empty_on_fresh_smoother(self):
        smoother = DetectionSmoother(window=3)
        result = smoother.update({})
        assert result == {}

    def test_output_types_are_int(self):
        smoother = DetectionSmoother(window=3)
        smoother.update({"maggi": 3})
        smoother.update({"maggi": 4})
        result = smoother.update({"maggi": 5})
        assert isinstance(result["maggi"], int)


# ===========================================================================
# RestockPlanner
# ===========================================================================

class TestRestockPlanner:

    def test_empty_report_no_tasks(self, analyzer, planner):
        # All OOS – but also confirm planner returns RestockTask objects
        report = analyzer.analyse(make_result([]))
        tasks = planner.plan(report)
        assert isinstance(tasks, list)
        assert len(tasks) > 0  # everything is OOS

    def test_oos_has_urgency_1(self, analyzer, planner):
        report = analyzer.analyse(make_result([]))
        tasks = planner.plan(report)
        oos_tasks = [t for t in tasks if t.status == StockStatus.OUT_OF_STOCK]
        for task in oos_tasks:
            assert task.urgency_score == 1.0

    def test_low_stock_urgency_less_than_1(self, analyzer, planner):
        dets = [
            Detection("maggi", 0.9, 0.2, 0.12, 0.05, 0.04),
            Detection("maggi", 0.9, 0.4, 0.12, 0.05, 0.04),
        ]
        report = analyzer.analyse(make_result(dets))
        tasks = planner.plan(report)
        low_tasks = [t for t in tasks if t.status == StockStatus.LOW_STOCK]
        for task in low_tasks:
            assert 0.0 < task.urgency_score < 1.0

    def test_tasks_ranked_sequentially(self, analyzer, planner):
        report = analyzer.analyse(make_result([]))
        tasks = planner.plan(report)
        for i, task in enumerate(tasks, start=1):
            assert task.rank == i

    def test_oos_tasks_come_before_low_stock(self, analyzer, planner):
        # Mix of OOS and Low Stock
        dets = [
            # maggi: 2/8 → low stock
            Detection("maggi", 0.9, 0.2, 0.12, 0.05, 0.04),
            Detection("maggi", 0.9, 0.4, 0.12, 0.05, 0.04),
        ]
        report = analyzer.analyse(make_result(dets))
        tasks = planner.plan(report)
        statuses = [t.status for t in tasks]
        # All OOS items must appear before any LOW_STOCK item
        last_oos = max(
            (i for i, s in enumerate(statuses) if s == StockStatus.OUT_OF_STOCK),
            default=-1,
        )
        first_low = min(
            (i for i, s in enumerate(statuses) if s == StockStatus.LOW_STOCK),
            default=len(tasks),
        )
        assert last_oos < first_low

    def test_units_needed_correct(self, analyzer, planner):
        # 2 maggi detected, 8 expected → need 6
        dets = [
            Detection("maggi", 0.9, 0.2, 0.12, 0.05, 0.04),
            Detection("maggi", 0.9, 0.4, 0.12, 0.05, 0.04),
        ]
        report = analyzer.analyse(make_result(dets))
        tasks = planner.plan(report)
        maggi_task = next((t for t in tasks if t.product == "maggi"), None)
        assert maggi_task is not None
        assert maggi_task.units_needed == 6

    def test_no_ok_items_in_tasks(self, analyzer, planner):
        # Give maggi full stock – it should not appear in restock list
        dets = [
            Detection("maggi", 0.9, x, 0.12, 0.05, 0.04)
            for x in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        ]
        report = analyzer.analyse(make_result(dets))
        tasks = planner.plan(report)
        maggi_tasks = [t for t in tasks if t.product == "maggi"]
        assert not maggi_tasks

    def test_str_representation(self, analyzer, planner):
        report = analyzer.analyse(make_result([]))
        tasks = planner.plan(report)
        task_str = str(tasks[0])
        assert "#" in task_str
        assert tasks[0].product in task_str

    def test_task_attributes(self, analyzer, planner):
        report = analyzer.analyse(make_result([]))
        tasks = planner.plan(report)
        task = tasks[0]
        assert hasattr(task, "rank")
        assert hasattr(task, "product")
        assert hasattr(task, "shelf_id")
        assert hasattr(task, "shelf_name")
        assert hasattr(task, "status")
        assert hasattr(task, "detected_count")
        assert hasattr(task, "expected_count")
        assert hasattr(task, "urgency_score")
        assert hasattr(task, "units_needed")
