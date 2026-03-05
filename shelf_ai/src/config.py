"""
config.py
---------
Configuration loading and validation utilities for the Shelf AI system.

Centralises YAML loading with robust error handling and sanity checks so
that misconfigured planogram or threshold files are detected early and
reported with clear, actionable messages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Required keys inside each shelf entry in planogram.yaml
_PLANOGRAM_SHELF_KEYS = frozenset(
    {"name", "zone_y_range", "allowed_products", "expected_counts"}
)

# Required keys inside the ``stock`` section of thresholds.yaml
_THRESHOLD_STOCK_KEYS = frozenset({"low_stock_ratio", "out_of_stock_count"})


class ConfigValidationError(ValueError):
    """Raised when a configuration file fails schema or value validation."""


# ---------------------------------------------------------------------------
# Low-level YAML loader
# ---------------------------------------------------------------------------


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Parse a YAML file and return its top-level mapping.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration mapping.

    Raises
    ------
    ConfigValidationError
        If the file does not exist, cannot be parsed, or the top-level
        value is not a mapping.
    """
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError as exc:
        raise ConfigValidationError(
            f"Configuration file not found: {path}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigValidationError(
            f"Invalid YAML in {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigValidationError(
            f"Expected a YAML mapping at top level in {path}, "
            f"got {type(data).__name__}."
        )
    return data


# ---------------------------------------------------------------------------
# Per-file validators
# ---------------------------------------------------------------------------


def validate_planogram(cfg: dict[str, Any]) -> None:
    """
    Validate a parsed planogram configuration dict.

    Checks that all required shelf keys are present and that every
    ``zone_y_range`` is a valid ``[y_min, y_max]`` pair with
    ``0 <= y_min < y_max <= 1``.

    Raises
    ------
    ConfigValidationError
        If the configuration is missing required keys or contains
        invalid values.
    """
    if "shelves" not in cfg:
        raise ConfigValidationError(
            "planogram.yaml is missing the required top-level 'shelves' key."
        )
    if not isinstance(cfg["shelves"], dict) or not cfg["shelves"]:
        raise ConfigValidationError(
            "'shelves' in planogram.yaml must be a non-empty mapping."
        )

    for shelf_id, shelf_cfg in cfg["shelves"].items():
        if not isinstance(shelf_cfg, dict):
            raise ConfigValidationError(
                f"Configuration for shelf '{shelf_id}' must be a mapping."
            )
        missing = _PLANOGRAM_SHELF_KEYS - set(shelf_cfg)
        if missing:
            raise ConfigValidationError(
                f"Shelf '{shelf_id}' in planogram.yaml is missing keys: "
                f"{sorted(missing)}."
            )
        y_range = shelf_cfg.get("zone_y_range", [])
        if (
            not isinstance(y_range, (list, tuple))
            or len(y_range) != 2
            or not (0.0 <= y_range[0] < y_range[1] <= 1.0)
        ):
            raise ConfigValidationError(
                f"Shelf '{shelf_id}' has an invalid 'zone_y_range': {y_range!r}. "
                "Expected [y_min, y_max] with 0 <= y_min < y_max <= 1."
            )


def validate_thresholds(cfg: dict[str, Any]) -> None:
    """
    Validate a parsed thresholds configuration dict.

    Raises
    ------
    ConfigValidationError
        If required keys are absent or values are out of the accepted range.
    """
    stock = cfg.get("stock")
    if not isinstance(stock, dict):
        raise ConfigValidationError(
            "thresholds.yaml is missing the required 'stock' section."
        )
    missing = _THRESHOLD_STOCK_KEYS - set(stock)
    if missing:
        raise ConfigValidationError(
            f"thresholds.yaml 'stock' section is missing keys: {sorted(missing)}."
        )

    ratio = stock.get("low_stock_ratio")
    if not isinstance(ratio, (int, float)) or not (0.0 < float(ratio) <= 1.0):
        raise ConfigValidationError(
            f"'stock.low_stock_ratio' must be a float in (0, 1], got {ratio!r}."
        )

    oos = stock.get("out_of_stock_count")
    if not isinstance(oos, int) or oos < 0:
        raise ConfigValidationError(
            f"'stock.out_of_stock_count' must be a non-negative integer, "
            f"got {oos!r}."
        )


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------


def load_and_validate_configs(
    planogram_path: str | Path,
    thresholds_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load *and* validate both configuration files.

    Parameters
    ----------
    planogram_path : str | Path
        Path to ``planogram.yaml``.
    thresholds_path : str | Path
        Path to ``thresholds.yaml``.

    Returns
    -------
    tuple[dict, dict]
        ``(planogram_cfg, thresholds_cfg)`` – both already validated.

    Raises
    ------
    ConfigValidationError
        If either file is missing, malformed, or fails validation.
    """
    planogram_cfg = load_yaml(planogram_path)
    validate_planogram(planogram_cfg)

    thresholds_cfg = load_yaml(thresholds_path)
    validate_thresholds(thresholds_cfg)

    logger.debug(
        "Configurations loaded and validated: %s, %s",
        planogram_path,
        thresholds_path,
    )
    return planogram_cfg, thresholds_cfg
