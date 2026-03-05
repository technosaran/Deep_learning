"""
history.py
----------
In-memory (+ optional JSON) stock history tracker for Shelf AI.

Records a timestamped :class:`~metrics.ShelfMetrics` snapshot every time the
shelf is analysed, enabling trend visualisation in the dashboard and reports.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """One timestamped KPI snapshot."""

    timestamp: float  # Unix epoch (seconds)
    overall_fill_rate: float
    compliance_rate: float
    health_score: float
    oos_count: int
    low_stock_count: int
    ok_count: int
    misplaced_count: int
    shelf_fill_rates: Dict[str, float] = field(default_factory=dict)

    @property
    def iso_timestamp(self) -> str:
        """ISO-8601 string for the entry timestamp."""
        import datetime

        return datetime.datetime.fromtimestamp(
            self.timestamp, tz=datetime.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")


class StockHistory:
    """
    Maintains a ring-buffer of :class:`HistoryEntry` objects.

    Parameters
    ----------
    max_entries : int
        Maximum number of entries to keep in memory (older entries are dropped).
    persistence_path : str | Path | None
        If provided, the history is automatically loaded from and saved to this
        JSON file so that restarts don't lose trend data.
    """

    def __init__(
        self,
        max_entries: int = 500,
        persistence_path: Optional[str | Path] = None,
    ) -> None:
        self._max_entries = max_entries
        self._entries: List[HistoryEntry] = []
        self._path: Optional[Path] = (
            Path(persistence_path) if persistence_path else None
        )
        if self._path and self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, metrics) -> HistoryEntry:
        """
        Append a new snapshot derived from a :class:`~metrics.ShelfMetrics`.

        Parameters
        ----------
        metrics : ShelfMetrics
            KPI snapshot to record.

        Returns
        -------
        HistoryEntry
            The entry that was just appended.
        """
        entry = HistoryEntry(
            timestamp=time.time(),
            overall_fill_rate=metrics.overall_fill_rate,
            compliance_rate=metrics.compliance_rate,
            health_score=metrics.health_score,
            oos_count=metrics.oos_count,
            low_stock_count=metrics.low_stock_count,
            ok_count=metrics.ok_count,
            misplaced_count=metrics.misplaced_count,
            shelf_fill_rates=dict(metrics.shelf_fill_rates),
        )
        self._entries.append(entry)
        # Trim to max size
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        if self._path:
            self._save()
        logger.debug(
            "History entry recorded: health=%.1f, fill=%.1f%%, entries=%d",
            entry.health_score,
            entry.overall_fill_rate * 100,
            len(self._entries),
        )
        return entry

    @property
    def entries(self) -> List[HistoryEntry]:
        """All stored entries, oldest first."""
        return list(self._entries)

    @property
    def latest(self) -> Optional[HistoryEntry]:
        """Most recent entry, or ``None`` if history is empty."""
        return self._entries[-1] if self._entries else None

    def fill_rate_trend(self) -> List[float]:
        """Return the time-ordered list of overall fill rates."""
        return [e.overall_fill_rate for e in self._entries]

    def health_score_trend(self) -> List[float]:
        """Return the time-ordered list of health scores."""
        return [e.health_score for e in self._entries]

    def compliance_rate_trend(self) -> List[float]:
        """Return the time-ordered list of compliance rates."""
        return [e.compliance_rate for e in self._entries]

    def anomaly_detected(
        self,
        threshold: float = 10.0,
        baseline_window: int = 5,
    ) -> tuple[bool, str]:
        """
        Detect a sudden drop in health score.

        Compares the latest health score against the rolling average of the
        previous *baseline_window* entries.  Returns ``(True, reason)`` when
        the drop equals or exceeds *threshold* points.

        Parameters
        ----------
        threshold : float
            Minimum point drop (absolute) to classify as an anomaly.
            Default is 10.0.
        baseline_window : int
            Number of historical entries (before the latest) used to compute
            the baseline average.  Default is 5.

        Returns
        -------
        tuple[bool, str]
            ``(detected, reason)`` – *reason* is an empty string when no
            anomaly is found.

        Examples
        --------
        >>> history = StockHistory()
        >>> history.record(metrics_good)   # health ≈ 80
        >>> history.record(metrics_bad)    # health ≈ 55
        >>> detected, reason = history.anomaly_detected(threshold=10)
        >>> detected
        True
        """
        if len(self._entries) < 2:
            return False, ""

        # Baseline: up to *baseline_window* entries before the latest
        historical = self._entries[-(baseline_window + 1):-1]
        if not historical:
            return False, ""

        baseline = sum(e.health_score for e in historical) / len(historical)
        latest = self._entries[-1].health_score
        drop = baseline - latest

        if drop >= threshold:
            reason = (
                f"Health score dropped {drop:.1f} points "
                f"(baseline={baseline:.1f}, current={latest:.1f})"
            )
            logger.warning("Anomaly detected: %s", reason)
            return True, reason

        return False, ""

    def clear(self) -> None:
        """Remove all entries and delete the persistence file if present."""
        self._entries.clear()
        if self._path and self._path.exists():
            self._path.unlink()

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(e) for e in self._entries]
        self._path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        assert self._path is not None
        try:
            raw = json.loads(self._path.read_text())
            self._entries = [HistoryEntry(**e) for e in raw]
            # Enforce max_entries limit after loading
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]
        except (json.JSONDecodeError, TypeError, KeyError):
            self._entries = []
