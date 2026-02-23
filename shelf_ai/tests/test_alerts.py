"""
tests/test_alerts.py
--------------------
Unit tests for the AlertManager.
"""

import sys
from pathlib import Path
import time
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.alerts import AlertManager

THRESHOLDS = str(REPO_ROOT / "config" / "thresholds.yaml")


@pytest.fixture
def alert_mgr():
    return AlertManager(THRESHOLDS)


class TestAlertCooldown:
    def test_alert_sent_once_within_cooldown(self, alert_mgr, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            alert_mgr.send("Test", "body", alert_key="k1")
            alert_mgr.send("Test", "body", alert_key="k1")
        # Only one WARNING log line expected (second is suppressed)
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 1

    def test_different_keys_both_sent(self, alert_mgr, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            alert_mgr.send("Test A", "body", alert_key="ka")
            alert_mgr.send("Test B", "body", alert_key="kb")
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 2

    def test_no_key_always_sent(self, alert_mgr, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            alert_mgr.send("No key", "body")
            alert_mgr.send("No key", "body")
        warning_msgs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_msgs) == 2
