"""
planogram.py
------------
Planogram compliance checker.

Provides a human-readable compliance report from the misplacement data
produced by :class:`~shelf_analyzer.ShelfAnalyzer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import yaml


@dataclass
class ComplianceIssue:
    """One planogram violation."""

    product: str
    detected_shelf_id: str
    detected_shelf_name: str
    expected_shelf_id: str
    expected_shelf_name: str

    def __str__(self) -> str:
        return (
            f"MISPLACED: '{self.product}' found on {self.detected_shelf_name!r} "
            f"but should be on {self.expected_shelf_name!r}"
        )


@dataclass
class ComplianceReport:
    """Full planogram compliance report."""

    issues: List[ComplianceIssue] = field(default_factory=list)

    @property
    def is_compliant(self) -> bool:
        return len(self.issues) == 0

    def summary(self) -> str:
        if self.is_compliant:
            return "✅ Planogram compliant – no misplacements detected."
        lines = [f"⚠️  {len(self.issues)} planogram violation(s) detected:"]
        for issue in self.issues:
            lines.append(f"  • {issue}")
        return "\n".join(lines)


class PlanogramChecker:
    """
    Converts raw misplacement tuples from :class:`~shelf_analyzer.ShelfAnalyzer`
    into structured :class:`ComplianceReport` objects.

    Parameters
    ----------
    planogram_path : str
        Path to ``planogram.yaml``.
    """

    def __init__(self, planogram_path: str) -> None:
        with open(planogram_path) as f:
            cfg = yaml.safe_load(f)
        self._shelf_names: Dict[str, str] = {
            sid: sc["name"] for sid, sc in cfg["shelves"].items()
        }

    def check(
        self, misplaced: List[Tuple[str, str, str]]
    ) -> ComplianceReport:
        """
        Build a :class:`ComplianceReport` from a list of
        ``(product, detected_shelf_id, expected_shelf_id)`` tuples.
        """
        issues = []
        for product, detected_sid, expected_sid in misplaced:
            issues.append(
                ComplianceIssue(
                    product=product,
                    detected_shelf_id=detected_sid,
                    detected_shelf_name=self._shelf_names.get(detected_sid, detected_sid),
                    expected_shelf_id=expected_sid,
                    expected_shelf_name=self._shelf_names.get(expected_sid, expected_sid),
                )
            )
        return ComplianceReport(issues=issues)
