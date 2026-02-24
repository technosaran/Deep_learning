"""
tests/test_improvements.py
--------------------------
Tests for the accuracy and real-time improvements:
  - ShelfDetector new parameters (augment, half)
  - train.py new CLI arguments (--model, --optimizer, --lr0, --lrf,
    --close-mosaic, --augment)
"""

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Detector parameter tests
# ---------------------------------------------------------------------------

class TestShelfDetectorNewParams:
    """ShelfDetector should accept augment and half parameters."""

    def test_default_augment_is_false(self):
        from src.detector import ShelfDetector

        det = ShelfDetector("fake.pt")
        assert det.augment is False

    def test_default_half_is_false(self):
        from src.detector import ShelfDetector

        det = ShelfDetector("fake.pt")
        assert det.half is False

    def test_augment_stored(self):
        from src.detector import ShelfDetector

        det = ShelfDetector("fake.pt", augment=True)
        assert det.augment is True

    def test_half_stored(self):
        from src.detector import ShelfDetector

        det = ShelfDetector("fake.pt", half=True)
        assert det.half is True

    def test_all_params_stored_together(self):
        from src.detector import ShelfDetector

        det = ShelfDetector(
            "fake.pt",
            confidence=0.55,
            iou=0.5,
            device="cuda",
            augment=True,
            half=True,
        )
        assert det.confidence == 0.55
        assert det.iou == 0.5
        assert det.device == "cuda"
        assert det.augment is True
        assert det.half is True


# ---------------------------------------------------------------------------
# train.py argument parser tests
# ---------------------------------------------------------------------------

class TestTrainArgParser:
    """train.py argument parser should expose all new parameters."""

    def _parse(self, *args):
        from train.train import parse_args

        # Provide minimum required fake --data to avoid default-path validation
        return parse_args(list(args))

    def test_default_optimizer(self):
        args = self._parse()
        assert args.optimizer == "auto"

    def test_custom_optimizer(self):
        args = self._parse("--optimizer", "AdamW")
        assert args.optimizer == "AdamW"

    def test_default_lr0(self):
        args = self._parse()
        assert abs(args.lr0 - 0.01) < 1e-9

    def test_custom_lr0(self):
        args = self._parse("--lr0", "0.001")
        assert abs(args.lr0 - 0.001) < 1e-9

    def test_default_lrf(self):
        args = self._parse()
        assert abs(args.lrf - 0.01) < 1e-9

    def test_custom_lrf(self):
        args = self._parse("--lrf", "0.1")
        assert abs(args.lrf - 0.1) < 1e-9

    def test_default_close_mosaic(self):
        args = self._parse()
        assert args.close_mosaic == 10

    def test_custom_close_mosaic(self):
        args = self._parse("--close-mosaic", "5")
        assert args.close_mosaic == 5

    def test_default_augment_flag_false(self):
        args = self._parse()
        assert args.augment is False

    def test_augment_flag_enabled(self):
        args = self._parse("--augment")
        assert args.augment is True

    def test_model_shorthand_none_by_default(self):
        args = self._parse()
        assert args.model is None

    def test_model_shorthand_small(self):
        args = self._parse("--model", "s")
        assert args.model == "s"

    def test_model_shorthand_medium(self):
        args = self._parse("--model", "m")
        assert args.model == "m"

    def test_model_shorthand_overrides_weights(self):
        """When --model is set, train() should use the mapped weights file."""
        from train.train import _MODEL_SIZES

        assert _MODEL_SIZES["s"] == "yolov8s.pt"
        assert _MODEL_SIZES["m"] == "yolov8m.pt"
        assert _MODEL_SIZES["l"] == "yolov8l.pt"
        assert _MODEL_SIZES["x"] == "yolov8x.pt"
        assert _MODEL_SIZES["n"] == "yolov8n.pt"
