"""
metrics.py
----------
KPI metrics calculator for the Shelf AI system.

Derives high-level retail KPIs from a :class:`~shelf_analyzer.ShelfReport`:
  - Overall fill rate
  - Planogram compliance rate
  - Out-of-stock and low-stock product counts
  - Per-shelf fill rates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .shelf_analyzer import ShelfReport, StockStatus


@dataclass
class ShelfMetrics:
    """
    High-level KPI snapshot computed from a single :class:`ShelfReport`.

    Attributes
    ----------
    overall_fill_rate : float
        Total detected / total expected across all shelves, capped at 1.0.
    compliance_rate : float
        Fraction of products that are correctly placed (not misplaced).
    oos_count : int
        Number of products currently out of stock.
    low_stock_count : int
        Number of products with low stock.
    ok_count : int
        Number of products with healthy stock.
    misplaced_count : int
        Number of misplaced product detections.
    shelf_fill_rates : dict
        Mapping of shelf_id -> fill_rate for each shelf.
    """

    overall_fill_rate: float
    compliance_rate: float
    oos_count: int
    low_stock_count: int
    ok_count: int
    misplaced_count: int
    shelf_fill_rates: Dict[str, float] = field(default_factory=dict)

    @property
    def health_score(self) -> float:
        """
        Composite 0–100 health score.

        Weighted average of fill rate (60 %) and compliance rate (40 %).
        """
        return round((self.overall_fill_rate * 60 + self.compliance_rate * 40), 1)

    def as_dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON export."""
        return {
            "overall_fill_rate": self.overall_fill_rate,
            "compliance_rate": self.compliance_rate,
            "health_score": self.health_score,
            "oos_count": self.oos_count,
            "low_stock_count": self.low_stock_count,
            "ok_count": self.ok_count,
            "misplaced_count": self.misplaced_count,
            "shelf_fill_rates": dict(self.shelf_fill_rates),
        }

    def summary(self) -> str:
        lines = [
            "── Shelf KPI Summary ──────────────────────",
            f"  Fill Rate        : {self.overall_fill_rate:.1%}",
            f"  Compliance Rate  : {self.compliance_rate:.1%}",
            f"  Health Score     : {self.health_score:.1f} / 100",
            f"  Out of Stock     : {self.oos_count}",
            f"  Low Stock        : {self.low_stock_count}",
            f"  OK               : {self.ok_count}",
            f"  Misplaced        : {self.misplaced_count}",
        ]
        if self.shelf_fill_rates:
            lines.append("  Per-shelf fill:")
            for sid, rate in self.shelf_fill_rates.items():
                lines.append(f"    {sid}: {rate:.1%}")
        lines.append("────────────────────────────────────────────")
        return "\n".join(lines)


class MetricsCalculator:
    """
    Computes :class:`ShelfMetrics` from a :class:`ShelfReport`.

    Example
    -------
    >>> calc = MetricsCalculator()
    >>> metrics = calc.compute(report)
    >>> print(metrics.health_score)
    """

    def compute(self, report: ShelfReport) -> ShelfMetrics:
        """Return a :class:`ShelfMetrics` snapshot for *report*."""
        all_items = report.all_items
        total_products = len(all_items)

        oos_count = len(report.out_of_stock_items)
        low_stock_count = len(report.low_stock_items)
        ok_count = len(report.ok_items)
        misplaced_count = len(report.misplaced)

        # Compliance rate: fraction of unique product slots not misplaced.
        # We penalise at the product-slot level (one slot per item in report).
        if total_products > 0:
            compliance_rate = 1.0 - (misplaced_count / total_products)
            compliance_rate = max(0.0, compliance_rate)
        else:
            compliance_rate = 1.0

        # Per-shelf fill rates
        shelf_fill_rates: Dict[str, float] = {}
        for shelf_id, stocks in report.shelf_stocks.items():
            if not stocks:
                shelf_fill_rates[shelf_id] = 1.0
                continue
            total_exp = sum(s.expected_count for s in stocks)
            total_det = sum(s.detected_count for s in stocks)
            shelf_fill_rates[shelf_id] = (
                min(total_det / total_exp, 1.0) if total_exp else 1.0
            )

        return ShelfMetrics(
            overall_fill_rate=report.overall_fill_rate,
            compliance_rate=compliance_rate,
            oos_count=oos_count,
            low_stock_count=low_stock_count,
            ok_count=ok_count,
            misplaced_count=misplaced_count,
            shelf_fill_rates=shelf_fill_rates,
        )
