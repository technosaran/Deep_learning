"""
demo.py
-------
Command-line demo for the Shelf AI system.

Runs detection + analysis on an image or a webcam feed and prints a
formatted shelf report to the terminal.

Usage
-----
# Analyse a saved image
python demo.py --source path/to/shelf.jpg

# Live webcam demo (press q to quit)
python demo.py --webcam

# Synthetic demo (no weights needed)
python demo.py --demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.shelf_analyzer import ShelfAnalyzer, StockStatus  # noqa: E402
from src.planogram import PlanogramChecker  # noqa: E402
from src.alerts import AlertManager  # noqa: E402

CONFIG_DIR = REPO_ROOT / "config"
PLANOGRAM_PATH = str(CONFIG_DIR / "planogram.yaml")
THRESHOLDS_PATH = str(CONFIG_DIR / "thresholds.yaml")
DEFAULT_WEIGHTS = str(
    REPO_ROOT / "runs" / "detect" / "shelf_ai" / "weights" / "best.pt"
)

# ANSI colour codes for terminal output
_RED = "\033[91m"
_YLW = "\033[93m"
_GRN = "\033[92m"
_BLU = "\033[94m"
_RST = "\033[0m"
_BLD = "\033[1m"


def _status_str(status: StockStatus) -> str:
    if status == StockStatus.OUT_OF_STOCK:
        return f"{_RED}OUT OF STOCK{_RST}"
    if status == StockStatus.LOW_STOCK:
        return f"{_YLW}LOW STOCK{_RST}"
    return f"{_GRN}OK{_RST}"


def print_report(report, compliance):
    print(f"\n{_BLD}{'─'*60}{_RST}")
    print(f"{_BLD}  SHELF AI – INVENTORY REPORT{_RST}")
    print(f"{'─'*60}")

    for shelf_id, stocks in report.shelf_stocks.items():
        shelf_name = stocks[0].shelf_name if stocks else shelf_id
        print(f"\n{_BLU}{_BLD}{shelf_name}{_RST}")
        for s in stocks:
            line = (
                f"  {s.product:<20}"
                f"  detected: {s.detected_count:>3} / {s.expected_count:<3}"
                f"  {_status_str(s.status)}"
            )
            print(line)

    print(f"\n{_BLD}{'─'*60}{_RST}")
    if compliance.is_compliant:
        print(f"  {_GRN}✅ Planogram Compliant{_RST}")
    else:
        print(f"  {_RED}⚠  Planogram Violations:{_RST}")
        for issue in compliance.issues:
            print(f"     • {issue}")

    if report.action_required:
        print(f"\n  {_RED}{_BLD}ACTION REQUIRED:{_RST}")
        for item in report.out_of_stock_items:
            print(f"  🔴 OUT OF STOCK – {item.product} ({item.shelf_name})")
        for item in report.low_stock_items:
            print(
                f"  🟡 LOW STOCK    – {item.product} "
                f"({item.detected_count}/{item.expected_count}) on {item.shelf_name}"
            )
        for product, det, exp in report.misplaced:
            print(f"  🔵 MISPLACED    – {product} on '{det}' (expected '{exp}')")
    print(f"{'─'*60}\n")


def _build_demo_result():
    """Synthetic result that covers all status types."""
    from src.detector import Detection, DetectionResult

    dets = [
        Detection("maggi", 0.92, 0.20, 0.12, 0.05, 0.04),
        Detection("maggi", 0.90, 0.30, 0.12, 0.05, 0.04),
        Detection("lays", 0.88, 0.50, 0.12, 0.05, 0.04),
        Detection("colgate", 0.85, 0.20, 0.37, 0.05, 0.04),
        Detection("colgate", 0.86, 0.35, 0.37, 0.05, 0.04),
        Detection("colgate", 0.84, 0.50, 0.37, 0.05, 0.04),
        Detection("dove", 0.80, 0.20, 0.37, 0.05, 0.04),
        Detection("dove", 0.78, 0.40, 0.12, 0.05, 0.04),   # misplaced
        Detection("coke", 0.91, 0.20, 0.62, 0.05, 0.04),
        Detection("coke", 0.89, 0.35, 0.62, 0.05, 0.04),
        Detection("coke", 0.88, 0.50, 0.62, 0.05, 0.04),
        Detection("pepsi", 0.87, 0.65, 0.62, 0.05, 0.04),
    ]
    return DetectionResult(detections=dets, image_width=640, image_height=480)


def run_on_image(source: str, weights: str, analyzer, checker, alert_mgr):
    from src.detector import ShelfDetector

    detector = ShelfDetector(weights)
    result = detector.predict(source, draw=True)

    print(f"Detected {len(result.detections)} objects in '{source}'")
    report = analyzer.analyse(result)
    compliance = checker.check(report.misplaced)
    print_report(report, compliance)

    try:
        import cv2

        if result.annotated_frame is not None:
            cv2.imshow("Shelf AI – Detections", result.annotated_frame)
            print("Press any key to close the image window…")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except ImportError:
        pass

    _fire_alerts(report, alert_mgr)


def run_webcam(weights: str, analyzer, checker, alert_mgr):
    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python required for webcam mode.", file=sys.stderr)
        sys.exit(1)

    from src.detector import ShelfDetector

    detector = ShelfDetector(weights)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.", file=sys.stderr)
        sys.exit(1)

    print("Webcam started – press 'q' to quit, 's' to analyse current frame.")
    last_analysis = 0.0
    ANALYSIS_INTERVAL = 3.0  # seconds between auto-analyses
    fps_start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()
        elapsed = now - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0.0

        if now - last_analysis >= ANALYSIS_INTERVAL:
            result = detector.predict(frame, draw=True)
            report = analyzer.analyse(result)
            compliance = checker.check(report.misplaced)
            print_report(report, compliance)
            print(f"  📷 Real-time FPS: {fps:.1f}")
            _fire_alerts(report, alert_mgr)
            last_analysis = now
            if result.annotated_frame is not None:
                frame = result.annotated_frame

        # Overlay FPS on the displayed frame
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Shelf AI – Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            last_analysis = 0  # force re-analysis on next frame

    cap.release()
    cv2.destroyAllWindows()


def _fire_alerts(report, alert_mgr):
    for item in report.out_of_stock_items:
        alert_mgr.send(
            subject=f"Out of Stock: {item.product.title()}",
            message=f"{item.product.title()} is OUT OF STOCK on {item.shelf_name}.",
            alert_key=f"{item.shelf_id}:{item.product}:oos",
        )
    for item in report.low_stock_items:
        alert_mgr.send(
            subject=f"Low Stock: {item.product.title()}",
            message=(
                f"{item.product.title()} LOW ({item.detected_count}/"
                f"{item.expected_count}) on {item.shelf_name}."
            ),
            alert_key=f"{item.shelf_id}:{item.product}:low",
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Shelf AI – demo script")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", help="Path to image file")
    group.add_argument("--webcam", action="store_true", help="Use live webcam feed")
    group.add_argument(
        "--demo", action="store_true", help="Synthetic demo (no weights needed)"
    )
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to best.pt weights")
    return p.parse_args(argv)


def main(argv=None):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args(argv)
    analyzer = ShelfAnalyzer(PLANOGRAM_PATH, THRESHOLDS_PATH)
    checker = PlanogramChecker(PLANOGRAM_PATH)
    alert_mgr = AlertManager(THRESHOLDS_PATH)

    if args.demo:
        print("Running synthetic demo (no model weights needed)…")
        result = _build_demo_result()
        report = analyzer.analyse(result)
        compliance = checker.check(report.misplaced)
        print_report(report, compliance)
        _fire_alerts(report, alert_mgr)
    elif args.source:
        run_on_image(args.source, args.weights, analyzer, checker, alert_mgr)
    else:
        run_webcam(args.weights, analyzer, checker, alert_mgr)


if __name__ == "__main__":
    main()
