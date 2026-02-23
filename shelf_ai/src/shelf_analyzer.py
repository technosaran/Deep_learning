"""
shelf_analyzer.py
-----------------
Maps detections to shelf zones, counts stock, and determines stock status
(OK / Low Stock / Out of Stock) for each product on each shelf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import yaml

from .detector import Detection, DetectionResult


class StockStatus(str, Enum):
    OK = "OK"
    LOW_STOCK = "Low Stock"
    OUT_OF_STOCK = "Out of Stock"


@dataclass
class ProductStock:
    """Stock state for a single product on a single shelf."""

    product: str
    shelf_id: str
    shelf_name: str
    detected_count: int
    expected_count: int
    status: StockStatus


@dataclass
class ShelfReport:
    """Aggregated analysis report for one image / frame."""

    # shelf_id -> list of ProductStock entries
    shelf_stocks: Dict[str, List[ProductStock]] = field(default_factory=dict)
    # list of (product, detected_shelf, expected_shelf) tuples
    misplaced: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def low_stock_items(self) -> List[ProductStock]:
        items = []
        for stocks in self.shelf_stocks.values():
            items.extend(s for s in stocks if s.status == StockStatus.LOW_STOCK)
        return items

    @property
    def out_of_stock_items(self) -> List[ProductStock]:
        items = []
        for stocks in self.shelf_stocks.values():
            items.extend(s for s in stocks if s.status == StockStatus.OUT_OF_STOCK)
        return items

    @property
    def action_required(self) -> bool:
        return bool(self.low_stock_items or self.out_of_stock_items or self.misplaced)


class ShelfAnalyzer:
    """
    Uses planogram + threshold configs to analyse a :class:`DetectionResult`.

    Parameters
    ----------
    planogram_path : str
        Path to ``planogram.yaml``.
    thresholds_path : str
        Path to ``thresholds.yaml``.
    """

    def __init__(self, planogram_path: str, thresholds_path: str) -> None:
        with open(planogram_path) as f:
            self._planogram = yaml.safe_load(f)
        with open(thresholds_path) as f:
            cfg = yaml.safe_load(f)

        self._low_stock_ratio: float = cfg["stock"]["low_stock_ratio"]
        self._out_of_stock_count: int = cfg["stock"]["out_of_stock_count"]

        # Build quick look-up: product_name -> shelf_id
        self._product_to_shelf: Dict[str, str] = {}
        for shelf_id, shelf_cfg in self._planogram["shelves"].items():
            for product in shelf_cfg["allowed_products"]:
                self._product_to_shelf[product] = shelf_id

    # ------------------------------------------------------------------
    # Zone helpers
    # ------------------------------------------------------------------

    def _detect_shelf_for(self, detection: Detection) -> str | None:
        """
        Return the shelf_id whose vertical zone contains this detection,
        or ``None`` if no zone matches.
        """
        y = detection.y_center
        for shelf_id, shelf_cfg in self._planogram["shelves"].items():
            y_min, y_max = shelf_cfg["zone_y_range"]
            if y_min <= y < y_max:
                return shelf_id
        # Clamp the last shelf to catch detections at y==1.0
        last_shelf_id = list(self._planogram["shelves"].keys())[-1]
        last_cfg = self._planogram["shelves"][last_shelf_id]
        if y >= last_cfg["zone_y_range"][0]:
            return last_shelf_id
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, result: DetectionResult) -> ShelfReport:
        """
        Analyse a :class:`DetectionResult` and return a :class:`ShelfReport`.
        """
        # Count detections per shelf per product
        counts: Dict[str, Dict[str, int]] = {
            sid: {} for sid in self._planogram["shelves"]
        }
        misplaced: List[Tuple[str, str, str]] = []

        for det in result.detections:
            detected_shelf = self._detect_shelf_for(det)
            if detected_shelf is None:
                continue

            product = det.label
            expected_shelf = self._product_to_shelf.get(product)

            # Misplacement check
            if expected_shelf and expected_shelf != detected_shelf:
                misplaced.append((product, detected_shelf, expected_shelf))

            # Still count the product on the shelf it was found
            counts[detected_shelf][product] = (
                counts[detected_shelf].get(product, 0) + 1
            )

        # Build shelf stocks
        shelf_stocks: Dict[str, List[ProductStock]] = {}
        for shelf_id, shelf_cfg in self._planogram["shelves"].items():
            stocks: List[ProductStock] = []
            for product, expected in shelf_cfg["expected_counts"].items():
                detected = counts[shelf_id].get(product, 0)
                status = self._stock_status(detected, expected)
                stocks.append(
                    ProductStock(
                        product=product,
                        shelf_id=shelf_id,
                        shelf_name=shelf_cfg["name"],
                        detected_count=detected,
                        expected_count=expected,
                        status=status,
                    )
                )
            shelf_stocks[shelf_id] = stocks

        return ShelfReport(shelf_stocks=shelf_stocks, misplaced=misplaced)

    def _stock_status(self, detected: int, expected: int) -> StockStatus:
        if detected <= self._out_of_stock_count:
            return StockStatus.OUT_OF_STOCK
        if expected > 0 and detected / expected <= self._low_stock_ratio:
            return StockStatus.LOW_STOCK
        return StockStatus.OK
