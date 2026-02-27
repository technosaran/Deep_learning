"""
tests/test_history.py
---------------------
Unit tests for StockHistory and HistoryEntry.
"""

import sys
import time
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.history import HistoryEntry, StockHistory
from src.metrics import ShelfMetrics


def _make_metrics(fill_rate=0.8, compliance_rate=1.0, oos=0, low=1, ok=19, misplaced=0):
    return ShelfMetrics(
        overall_fill_rate=fill_rate,
        compliance_rate=compliance_rate,
        oos_count=oos,
        low_stock_count=low,
        ok_count=ok,
        misplaced_count=misplaced,
        shelf_fill_rates={"shelf_a": fill_rate},
    )


class TestStockHistory:
    def test_empty_on_creation(self):
        h = StockHistory()
        assert len(h) == 0
        assert h.latest is None

    def test_record_adds_entry(self):
        h = StockHistory()
        m = _make_metrics()
        entry = h.record(m)
        assert len(h) == 1
        assert isinstance(entry, HistoryEntry)
        assert entry.overall_fill_rate == m.overall_fill_rate

    def test_latest_returns_newest(self):
        h = StockHistory()
        h.record(_make_metrics(fill_rate=0.5))
        h.record(_make_metrics(fill_rate=0.9))
        assert abs(h.latest.overall_fill_rate - 0.9) < 1e-9

    def test_max_entries_enforced(self):
        h = StockHistory(max_entries=5)
        for i in range(10):
            h.record(_make_metrics(fill_rate=i / 10))
        assert len(h) == 5
        # Newest entries should be kept
        assert abs(h.latest.overall_fill_rate - 0.9) < 1e-9

    def test_fill_rate_trend(self):
        h = StockHistory()
        for rate in [0.2, 0.5, 0.8]:
            h.record(_make_metrics(fill_rate=rate))
        trend = h.fill_rate_trend()
        assert trend == [0.2, 0.5, 0.8]

    def test_health_score_trend(self):
        h = StockHistory()
        m = _make_metrics(fill_rate=1.0, compliance_rate=1.0)
        h.record(m)
        trend = h.health_score_trend()
        assert len(trend) == 1
        assert trend[0] == m.health_score

    def test_compliance_rate_trend(self):
        h = StockHistory()
        for rate in [0.6, 0.8, 1.0]:
            h.record(_make_metrics(compliance_rate=rate))
        trend = h.compliance_rate_trend()
        assert trend == [0.6, 0.8, 1.0]

    def test_compliance_rate_trend_empty(self):
        h = StockHistory()
        assert h.compliance_rate_trend() == []

    def test_clear_removes_entries(self):
        h = StockHistory()
        h.record(_make_metrics())
        h.clear()
        assert len(h) == 0
        assert h.latest is None

    def test_iso_timestamp_format(self):
        h = StockHistory()
        entry = h.record(_make_metrics())
        iso = entry.iso_timestamp
        # Should match YYYY-MM-DDTHH:MM:SSZ
        assert "T" in iso
        assert iso.endswith("Z")

    def test_json_persistence(self, tmp_path):
        path = tmp_path / "history.json"
        h1 = StockHistory(persistence_path=path)
        h1.record(_make_metrics(fill_rate=0.7))
        h1.record(_make_metrics(fill_rate=0.85))
        assert path.exists()

        # Load in a new instance
        h2 = StockHistory(persistence_path=path)
        assert len(h2) == 2
        assert abs(h2.latest.overall_fill_rate - 0.85) < 1e-9

    def test_clear_deletes_persistence_file(self, tmp_path):
        path = tmp_path / "history.json"
        h = StockHistory(persistence_path=path)
        h.record(_make_metrics())
        assert path.exists()
        h.clear()
        assert not path.exists()
