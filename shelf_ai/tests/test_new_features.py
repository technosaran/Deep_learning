"""
tests/test_new_features.py
--------------------------
Tests for newly added features:
  - ConfigValidationError + config validation helpers (src/config.py)
  - StockHistory.anomaly_detected()               (src/history.py)
  - AlertManager.prune_cooldown()                 (src/alerts.py)

No model weights or GPU required.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    ConfigValidationError,
    load_yaml,
    validate_planogram,
    validate_thresholds,
    load_and_validate_configs,
)
from src.history import StockHistory
from src.metrics import ShelfMetrics
from src.alerts import AlertManager

PLANOGRAM = str(REPO_ROOT / "config" / "planogram.yaml")
THRESHOLDS = str(REPO_ROOT / "config" / "thresholds.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(health: float, fill_rate: float = 0.8) -> ShelfMetrics:
    """Return a minimal ShelfMetrics with the given health score."""
    # Back-calculate fill_rate and compliance_rate so that health_score
    # == fill_rate*60 + compliance_rate*40 ≈ desired health
    # For simplicity we use fill_rate=0.8, compliance_rate=(health-48)/40
    compliance = max(0.0, min(1.0, (health - fill_rate * 60) / 40))
    return ShelfMetrics(
        overall_fill_rate=fill_rate,
        compliance_rate=compliance,
        oos_count=0,
        low_stock_count=0,
        ok_count=5,
        misplaced_count=0,
        shelf_fill_rates={},
    )


# ===========================================================================
# Config validation – load_yaml
# ===========================================================================

class TestLoadYaml:

    def test_loads_valid_planogram(self):
        cfg = load_yaml(PLANOGRAM)
        assert "shelves" in cfg

    def test_loads_valid_thresholds(self):
        cfg = load_yaml(THRESHOLDS)
        assert "stock" in cfg

    def test_missing_file_raises(self):
        with pytest.raises(ConfigValidationError, match="not found"):
            load_yaml("/nonexistent/path/config.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : : invalid yaml :::")
        with pytest.raises(ConfigValidationError, match="Invalid YAML"):
            load_yaml(str(bad))

    def test_non_mapping_top_level_raises(self, tmp_path):
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigValidationError, match="mapping"):
            load_yaml(str(bad))

    def test_empty_file_returns_empty_dict(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        result = load_yaml(str(empty))
        assert result == {}


# ===========================================================================
# Config validation – validate_planogram
# ===========================================================================

class TestValidatePlanogram:

    def test_valid_planogram_passes(self):
        cfg = load_yaml(PLANOGRAM)
        validate_planogram(cfg)  # must not raise

    def test_missing_shelves_key_raises(self):
        with pytest.raises(ConfigValidationError, match="'shelves'"):
            validate_planogram({})

    def test_empty_shelves_raises(self):
        with pytest.raises(ConfigValidationError, match="non-empty"):
            validate_planogram({"shelves": {}})

    def test_shelf_missing_required_key_raises(self):
        cfg = {
            "shelves": {
                "shelf_a": {
                    "name": "A",
                    "zone_y_range": [0.0, 0.5],
                    "allowed_products": ["x"],
                    # 'expected_counts' deliberately missing
                }
            }
        }
        with pytest.raises(ConfigValidationError, match="expected_counts"):
            validate_planogram(cfg)

    def test_invalid_y_range_values_raise(self):
        cfg = {
            "shelves": {
                "shelf_a": {
                    "name": "A",
                    "zone_y_range": [0.8, 0.2],  # min > max
                    "allowed_products": ["x"],
                    "expected_counts": {"x": 5},
                }
            }
        }
        with pytest.raises(ConfigValidationError, match="zone_y_range"):
            validate_planogram(cfg)

    def test_y_range_out_of_unit_interval_raises(self):
        cfg = {
            "shelves": {
                "shelf_a": {
                    "name": "A",
                    "zone_y_range": [0.0, 1.5],  # max > 1
                    "allowed_products": [],
                    "expected_counts": {},
                }
            }
        }
        with pytest.raises(ConfigValidationError, match="zone_y_range"):
            validate_planogram(cfg)

    def test_shelf_cfg_not_a_dict_raises(self):
        cfg = {"shelves": {"shelf_a": "not_a_dict"}}
        with pytest.raises(ConfigValidationError, match="mapping"):
            validate_planogram(cfg)


# ===========================================================================
# Config validation – validate_thresholds
# ===========================================================================

class TestValidateThresholds:

    def test_valid_thresholds_passes(self):
        cfg = load_yaml(THRESHOLDS)
        validate_thresholds(cfg)  # must not raise

    def test_missing_stock_section_raises(self):
        with pytest.raises(ConfigValidationError, match="'stock'"):
            validate_thresholds({})

    def test_missing_low_stock_ratio_raises(self):
        cfg = {"stock": {"out_of_stock_count": 0}}
        with pytest.raises(ConfigValidationError, match="low_stock_ratio"):
            validate_thresholds(cfg)

    def test_missing_out_of_stock_count_raises(self):
        cfg = {"stock": {"low_stock_ratio": 0.4}}
        with pytest.raises(ConfigValidationError, match="out_of_stock_count"):
            validate_thresholds(cfg)

    def test_low_stock_ratio_zero_raises(self):
        cfg = {"stock": {"low_stock_ratio": 0.0, "out_of_stock_count": 0}}
        with pytest.raises(ConfigValidationError, match="low_stock_ratio"):
            validate_thresholds(cfg)

    def test_low_stock_ratio_above_one_raises(self):
        cfg = {"stock": {"low_stock_ratio": 1.5, "out_of_stock_count": 0}}
        with pytest.raises(ConfigValidationError, match="low_stock_ratio"):
            validate_thresholds(cfg)

    def test_negative_out_of_stock_count_raises(self):
        cfg = {"stock": {"low_stock_ratio": 0.4, "out_of_stock_count": -1}}
        with pytest.raises(ConfigValidationError, match="out_of_stock_count"):
            validate_thresholds(cfg)

    def test_out_of_stock_count_not_int_raises(self):
        cfg = {"stock": {"low_stock_ratio": 0.4, "out_of_stock_count": 0.5}}
        with pytest.raises(ConfigValidationError, match="out_of_stock_count"):
            validate_thresholds(cfg)


# ===========================================================================
# Config validation – load_and_validate_configs
# ===========================================================================

class TestLoadAndValidateConfigs:

    def test_returns_two_dicts(self):
        p, t = load_and_validate_configs(PLANOGRAM, THRESHOLDS)
        assert isinstance(p, dict)
        assert isinstance(t, dict)

    def test_planogram_has_shelves(self):
        p, _ = load_and_validate_configs(PLANOGRAM, THRESHOLDS)
        assert "shelves" in p

    def test_thresholds_has_stock(self):
        _, t = load_and_validate_configs(PLANOGRAM, THRESHOLDS)
        assert "stock" in t

    def test_missing_planogram_raises(self):
        with pytest.raises(ConfigValidationError):
            load_and_validate_configs("/no/such/planogram.yaml", THRESHOLDS)

    def test_missing_thresholds_raises(self):
        with pytest.raises(ConfigValidationError):
            load_and_validate_configs(PLANOGRAM, "/no/such/thresholds.yaml")

    def test_config_validation_error_is_value_error(self):
        """ConfigValidationError must be a subclass of ValueError."""
        with pytest.raises(ValueError):
            load_and_validate_configs("/missing.yaml", THRESHOLDS)


# ===========================================================================
# StockHistory.anomaly_detected()
# ===========================================================================

class TestAnomalyDetected:

    def test_no_anomaly_when_history_empty(self):
        h = StockHistory()
        detected, reason = h.anomaly_detected()
        assert not detected
        assert reason == ""

    def test_no_anomaly_with_single_entry(self):
        h = StockHistory()
        h.record(_make_metrics(80.0))
        detected, reason = h.anomaly_detected()
        assert not detected

    def test_detects_large_drop(self):
        h = StockHistory()
        # Build a stable baseline
        for _ in range(5):
            h.record(_make_metrics(80.0))
        # Sudden drop
        h.record(_make_metrics(60.0))
        detected, reason = h.anomaly_detected(threshold=10.0)
        assert detected
        assert "dropped" in reason

    def test_no_anomaly_for_small_drop(self):
        h = StockHistory()
        for _ in range(5):
            h.record(_make_metrics(80.0))
        h.record(_make_metrics(75.0))  # only 5 points below threshold=10
        detected, _ = h.anomaly_detected(threshold=10.0)
        assert not detected

    def test_no_anomaly_for_improvement(self):
        h = StockHistory()
        for _ in range(5):
            h.record(_make_metrics(60.0))
        h.record(_make_metrics(80.0))  # improvement, not a drop
        detected, _ = h.anomaly_detected(threshold=10.0)
        assert not detected

    def test_custom_threshold_honoured(self):
        h = StockHistory()
        for _ in range(3):
            h.record(_make_metrics(70.0))
        h.record(_make_metrics(65.0))  # 5-point drop
        # With threshold=3 it should be an anomaly
        detected_low, _ = h.anomaly_detected(threshold=3.0)
        assert detected_low
        # With threshold=10 it should NOT be an anomaly
        detected_high, _ = h.anomaly_detected(threshold=10.0)
        assert not detected_high

    def test_reason_contains_baseline_and_current(self):
        h = StockHistory()
        for _ in range(5):
            h.record(_make_metrics(80.0))
        h.record(_make_metrics(55.0))
        _, reason = h.anomaly_detected(threshold=10.0)
        assert "baseline=" in reason
        assert "current=" in reason

    def test_baseline_window_limits_lookback(self):
        h = StockHistory()
        # Very old entry with low health
        h.record(_make_metrics(20.0))
        # Recent stable entries
        for _ in range(6):
            h.record(_make_metrics(80.0))
        # Drop
        h.record(_make_metrics(65.0))
        # With baseline_window=5 the baseline ignores the old entry
        detected, _ = h.anomaly_detected(threshold=10.0, baseline_window=5)
        # 80 - 65 = 15 ≥ 10 → anomaly
        assert detected

    def test_exactly_at_threshold_is_anomaly(self):
        h = StockHistory()
        for _ in range(3):
            h.record(_make_metrics(80.0))
        h.record(_make_metrics(70.0))  # exactly 10 points below
        detected, _ = h.anomaly_detected(threshold=10.0)
        assert detected


# ===========================================================================
# AlertManager.prune_cooldown()
# ===========================================================================

class TestPruneCooldown:

    def test_prune_removes_old_entries(self, tmp_path):
        t_path = tmp_path / "t.yaml"
        t_path.write_text(
            "stock:\n  low_stock_ratio: 0.4\n  out_of_stock_count: 0\n"
        )
        mgr = AlertManager(str(t_path))

        # Manually inject stale entries
        old_time = time.time() - 700  # older than default 2×300=600 s
        mgr._last_sent["key_a"] = old_time
        mgr._last_sent["key_b"] = old_time
        mgr._last_sent["key_c"] = time.time()  # fresh

        removed = mgr.prune_cooldown()
        assert removed == 2
        assert "key_a" not in mgr._last_sent
        assert "key_b" not in mgr._last_sent
        assert "key_c" in mgr._last_sent

    def test_prune_returns_zero_when_nothing_to_prune(self, tmp_path):
        t_path = tmp_path / "t.yaml"
        t_path.write_text(
            "stock:\n  low_stock_ratio: 0.4\n  out_of_stock_count: 0\n"
        )
        mgr = AlertManager(str(t_path))
        mgr._last_sent["key_fresh"] = time.time()
        assert mgr.prune_cooldown() == 0

    def test_prune_respects_custom_max_age(self, tmp_path):
        t_path = tmp_path / "t.yaml"
        t_path.write_text(
            "stock:\n  low_stock_ratio: 0.4\n  out_of_stock_count: 0\n"
        )
        mgr = AlertManager(str(t_path))
        # Entry that is 200 s old
        mgr._last_sent["key_200s"] = time.time() - 200
        # With max_age_seconds=100, it should be removed
        removed = mgr.prune_cooldown(max_age_seconds=100)
        assert removed == 1
        assert "key_200s" not in mgr._last_sent

    def test_prune_empty_dict_is_safe(self, tmp_path):
        t_path = tmp_path / "t.yaml"
        t_path.write_text(
            "stock:\n  low_stock_ratio: 0.4\n  out_of_stock_count: 0\n"
        )
        mgr = AlertManager(str(t_path))
        assert mgr.prune_cooldown() == 0

    def test_last_prune_updated_on_auto_prune(self, tmp_path):
        """Auto-prune inside send() updates _last_prune."""
        t_path = tmp_path / "t.yaml"
        t_path.write_text(
            "stock:\n  low_stock_ratio: 0.4\n  out_of_stock_count: 0\n"
        )
        mgr = AlertManager(str(t_path))
        # Force auto-prune on next send by backdating _last_prune
        mgr._last_prune = 0.0
        mgr.send("subj", "body", alert_key="unique_key_xyz")
        assert mgr._last_prune > 0
