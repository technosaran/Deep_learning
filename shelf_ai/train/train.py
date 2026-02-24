"""
train.py
--------
Training script for the Shelf AI product detector.

Uses Ultralytics YOLOv8 to fine-tune on a custom labelled shelf dataset.

Usage
-----
1. Prepare your dataset with Roboflow or LabelImg (YOLO format).
2. Place it at  data/shelf_dataset/  (or update DATA_YAML below).
3. Run:

    python train/train.py

The best weights will be saved to:
    runs/detect/shelf_ai/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths (override via CLI args)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = str(REPO_ROOT / "data" / "shelf_dataset" / "data.yaml")
DEFAULT_WEIGHTS = "yolov8n.pt"        # nano model – fast to train, small footprint
DEFAULT_EPOCHS = 50
DEFAULT_IMG_SIZE = 640
DEFAULT_BATCH = 16
DEFAULT_PROJECT = str(REPO_ROOT / "runs" / "detect")
DEFAULT_RUN_NAME = "shelf_ai"

# Model size map: shorthand letter → YOLOv8 weights filename
_MODEL_SIZES = {
    "n": "yolov8n.pt",
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
    "l": "yolov8l.pt",
    "x": "yolov8x.pt",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for shelf product detection."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_YAML,
        help="Path to dataset data.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Base YOLOv8 weights or model size letter (n/s/m/l/x) (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        choices=list(_MODEL_SIZES.keys()),
        default=None,
        help=(
            "YOLOv8 model size shorthand: n (nano), s (small), m (medium), "
            "l (large), x (extra-large).  Overrides --weights when specified."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="Input image size (default: %(default)s)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="",
        help='Training device: "" = auto, "cpu", "0", "0,1" (default: auto)',
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="Directory for saving runs (default: %(default)s)",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_RUN_NAME,
        help="Run subdirectory name (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data-loader worker processes (default: 4)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint in the run directory",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache dataset images in RAM for faster training",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Number of backbone layers to freeze (0 = none, default: 0)",
    )
    parser.add_argument(
        "--optimizer",
        default="auto",
        choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
        help="Optimizer to use during training (default: auto)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate as a fraction of lr0 (default: 0.01)",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        dest="close_mosaic",
        help=(
            "Disable mosaic augmentation for the last N epochs to stabilise "
            "training (default: 10)"
        ),
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable test-time augmentation (TTA) during final validation",
    )
    return parser.parse_args(argv)


def train(args) -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print(
            "ERROR: ultralytics not installed.\n"
            "Install with:  pip install ultralytics",
            file=sys.stderr,
        )
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(
            f"ERROR: Dataset YAML not found: {data_path}\n\n"
            "Dataset preparation steps:\n"
            "  1. Collect 300–800 shelf photos.\n"
            "  2. Label them in Roboflow (or LabelImg) in YOLO format.\n"
            "  3. Export and place under  data/shelf_dataset/.\n"
            "     The folder should contain:\n"
            "       data.yaml\n"
            "       train/images/  train/labels/\n"
            "       valid/images/  valid/labels/\n"
            "       test/images/   test/labels/\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve weights: --model shorthand takes priority over --weights
    weights = _MODEL_SIZES[args.model] if args.model else args.weights

    print(f"Loading base model: {weights}")
    model = YOLO(weights)

    print(f"Starting training – {args.epochs} epochs, img_size={args.imgsz}")
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=15,           # early stopping patience
        save_period=10,        # save checkpoint every 10 epochs
        plots=True,
        workers=args.workers,
        resume=args.resume,
        cache=args.cache,
        freeze=args.freeze if args.freeze > 0 else None,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        close_mosaic=args.close_mosaic,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n✅ Training complete.\nBest weights saved to: {best_weights}")

    if args.augment:
        print("\n🔍 Running validation with test-time augmentation (TTA)…")
        model.val(data=str(data_path), augment=True)


if __name__ == "__main__":
    train(parse_args())
