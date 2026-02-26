"""
restock.py
----------
Restock priority planner for the Shelf AI system.

Converts a :class:`~shelf_analyzer.ShelfReport` into an ordered task list
so that store staff know *exactly* which products to restock first and how
many units to carry.

Priority rules
~~~~~~~~~~~~~~
- Out-of-stock products receive an urgency score of **1.0** (critical).
- Low-stock products receive a score of ``(1 - fill_rate) × 0.7``
  (the emptier the shelf the more urgent).
- Tasks are sorted by urgency score descending, breaking ties by units
  needed (descending) then product name (alphabetical).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .shelf_analyzer import ShelfReport, StockStatus


@dataclass
class RestockTask:
    """One prioritised restock task."""

    rank: int
    product: str
    shelf_id: str
    shelf_name: str
    status: StockStatus
    detected_count: int
    expected_count: int
    urgency_score: float  # 0.0–1.0  (higher = more urgent)

    @property
    def units_needed(self) -> int:
        """Units required to bring the shelf back to expected capacity."""
        return max(0, self.expected_count - self.detected_count)

    def __str__(self) -> str:
        return (
            f"#{self.rank:>2}  [{self.urgency_score:.2f}]  "
            f"{self.product:<20}  {self.shelf_name:<28}  "
            f"need {self.units_needed:>3} unit(s)  ({self.status.value})"
        )


class RestockPlanner:
    """
    Produces a prioritised :class:`RestockTask` list from a
    :class:`~shelf_analyzer.ShelfReport`.

    Only out-of-stock and low-stock items appear in the list.  Products
    with OK status require no action and are omitted.

    Example
    -------
    >>> planner = RestockPlanner()
    >>> tasks = planner.plan(report)
    >>> for task in tasks:
    ...     print(task)
    """

    def plan(self, report: ShelfReport) -> List[RestockTask]:
        """
        Return an urgency-ordered list of :class:`RestockTask` objects.

        Parameters
        ----------
        report : ShelfReport
            Analysis result from :class:`~shelf_analyzer.ShelfAnalyzer`.

        Returns
        -------
        list[RestockTask]
            Tasks sorted by urgency (highest first).
        """
        tasks: List[RestockTask] = []

        for item in report.out_of_stock_items:
            tasks.append(
                RestockTask(
                    rank=0,  # assigned after sorting
                    product=item.product,
                    shelf_id=item.shelf_id,
                    shelf_name=item.shelf_name,
                    status=item.status,
                    detected_count=item.detected_count,
                    expected_count=item.expected_count,
                    urgency_score=1.0,
                )
            )

        for item in report.low_stock_items:
            urgency = round((1.0 - item.fill_rate) * 0.7, 4)
            tasks.append(
                RestockTask(
                    rank=0,
                    product=item.product,
                    shelf_id=item.shelf_id,
                    shelf_name=item.shelf_name,
                    status=item.status,
                    detected_count=item.detected_count,
                    expected_count=item.expected_count,
                    urgency_score=urgency,
                )
            )

        # Sort: urgency descending → units_needed descending → product name asc
        tasks.sort(
            key=lambda t: (-t.urgency_score, -t.units_needed, t.product)
        )

        for i, task in enumerate(tasks, start=1):
            task.rank = i

        return tasks
