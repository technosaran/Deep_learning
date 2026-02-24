# shelf_ai/src/__init__.py
from .detector import Detection, DetectionResult, ShelfDetector
from .shelf_analyzer import ProductStock, ShelfAnalyzer, ShelfReport, StockStatus
from .planogram import ComplianceIssue, ComplianceReport, PlanogramChecker
from .alerts import AlertManager
from .metrics import MetricsCalculator, ShelfMetrics
from .history import HistoryEntry, StockHistory

__all__ = [
    "Detection",
    "DetectionResult",
    "ShelfDetector",
    "ProductStock",
    "ShelfAnalyzer",
    "ShelfReport",
    "StockStatus",
    "ComplianceIssue",
    "ComplianceReport",
    "PlanogramChecker",
    "AlertManager",
    "MetricsCalculator",
    "ShelfMetrics",
    "HistoryEntry",
    "StockHistory",
]
