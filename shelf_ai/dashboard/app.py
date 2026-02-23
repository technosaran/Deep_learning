"""
app.py  –  Shelf AI Dashboard
------------------------------
Streamlit dashboard for real-time retail shelf monitoring.

Run with:
    streamlit run dashboard/app.py

The dashboard supports three modes:
  1. Upload an image  – analyse a single shelf photo.
  2. Webcam capture   – grab a frame from the local webcam.
  3. Demo mode        – run on a built-in synthetic demo (no weights needed).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# Allow imports from the parent package when running directly
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.shelf_analyzer import ShelfAnalyzer, StockStatus  # noqa: E402
from src.planogram import PlanogramChecker  # noqa: E402
from src.alerts import AlertManager  # noqa: E402
from src.metrics import MetricsCalculator  # noqa: E402
from src.history import StockHistory  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIG_DIR = REPO_ROOT / "config"
PLANOGRAM_PATH = str(CONFIG_DIR / "planogram.yaml")
THRESHOLDS_PATH = str(CONFIG_DIR / "thresholds.yaml")
DEFAULT_WEIGHTS = str(REPO_ROOT / "runs" / "detect" / "shelf_ai" / "weights" / "best.pt")

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def load_analyzer():
    return ShelfAnalyzer(PLANOGRAM_PATH, THRESHOLDS_PATH)


@st.cache_resource
def load_checker():
    return PlanogramChecker(PLANOGRAM_PATH)


@st.cache_resource
def load_alert_manager():
    return AlertManager(THRESHOLDS_PATH)


@st.cache_resource
def load_detector(weights_path: str):
    from src.detector import ShelfDetector  # noqa: PLC0415
    import yaml  # noqa: PLC0415
    with open(THRESHOLDS_PATH) as f:
        cfg = yaml.safe_load(f)
    m = cfg.get("model", {})
    return ShelfDetector(
        weights_path=weights_path,
        confidence=m.get("confidence_threshold", 0.45),
        iou=m.get("iou_threshold", 0.45),
        device=m.get("device", "cpu"),
    )

# ---------------------------------------------------------------------------
# Status badge helpers
# ---------------------------------------------------------------------------

STATUS_COLOR = {
    StockStatus.OK: "🟢",
    StockStatus.LOW_STOCK: "🟡",
    StockStatus.OUT_OF_STOCK: "🔴",
}


def _badge(status: StockStatus) -> str:
    return f"{STATUS_COLOR[status]} {status.value}"

# ---------------------------------------------------------------------------
# Demo mode – synthetic detection result (no model weights needed)
# ---------------------------------------------------------------------------

def _build_demo_result():
    """Return a fake DetectionResult that exercises all status types."""
    from src.detector import Detection, DetectionResult  # noqa: PLC0415

    dets = [
        # Shelf A (y ≈ 0.12)
        Detection("maggi", 0.92, 0.20, 0.12, 0.05, 0.04),
        Detection("maggi", 0.90, 0.30, 0.12, 0.05, 0.04),
        # 2 out of 8 → Low Stock
        Detection("lays", 0.88, 0.50, 0.12, 0.05, 0.04),
        # Shelf B (y ≈ 0.37)
        Detection("colgate", 0.85, 0.20, 0.37, 0.05, 0.04),
        Detection("colgate", 0.86, 0.35, 0.37, 0.05, 0.04),
        Detection("colgate", 0.84, 0.50, 0.37, 0.05, 0.04),
        Detection("dove", 0.80, 0.20, 0.37, 0.05, 0.04),
        # Misplacement: soap on shelf A
        Detection("dove", 0.78, 0.40, 0.12, 0.05, 0.04),
        # Shelf C (y ≈ 0.62)
        Detection("coke", 0.91, 0.20, 0.62, 0.05, 0.04),
        Detection("coke", 0.89, 0.35, 0.62, 0.05, 0.04),
        Detection("coke", 0.88, 0.50, 0.62, 0.05, 0.04),
        Detection("pepsi", 0.87, 0.65, 0.62, 0.05, 0.04),
    ]
    return DetectionResult(detections=dets, image_width=640, image_height=480)


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------

def render_shelf_report(report, checker, history: StockHistory | None = None):
    compliance = checker.check(report.misplaced)

    # ── KPI metrics ───────────────────────────────────────────────────
    calc = MetricsCalculator()
    metrics = calc.compute(report)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("🩺 Health Score", f"{metrics.health_score:.1f} / 100")
    kpi2.metric("📦 Fill Rate", f"{metrics.overall_fill_rate:.1%}")
    kpi3.metric("🔴 Out of Stock", metrics.oos_count)
    kpi4.metric("🟡 Low Stock", metrics.low_stock_count)

    st.divider()

    # ── Summary bar ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Out of Stock", len(report.out_of_stock_items))
    col2.metric("Low Stock", len(report.low_stock_items))
    col3.metric("Misplaced", len(report.misplaced))

    st.divider()

    # ── Planogram compliance ──────────────────────────────────────────
    if compliance.is_compliant:
        st.success("✅ Planogram Compliant – no misplacements detected")
    else:
        st.error(f"⚠️ {len(compliance.issues)} planogram violation(s) detected")
        for issue in compliance.issues:
            st.warning(str(issue))

    st.divider()

    # ── Per-shelf tables ──────────────────────────────────────────────
    for shelf_id, stocks in report.shelf_stocks.items():
        shelf_name = stocks[0].shelf_name if stocks else shelf_id
        shelf_fill = metrics.shelf_fill_rates.get(shelf_id, 1.0)
        with st.expander(
            f"📦 {shelf_name}  —  fill rate: {shelf_fill:.1%}", expanded=True
        ):
            rows = []
            for s in stocks:
                rows.append(
                    {
                        "Product": s.product.title(),
                        "Detected": s.detected_count,
                        "Expected": s.expected_count,
                        "Fill Rate": f"{s.fill_rate:.1%}",
                        "Status": _badge(s.status),
                    }
                )
            st.table(rows)

    # ── Action required ───────────────────────────────────────────────
    if report.action_required:
        st.divider()
        st.subheader("🚨 Action Required")
        for item in report.out_of_stock_items:
            st.error(
                f"OUT OF STOCK – {item.product.title()} on {item.shelf_name}"
            )
        for item in report.low_stock_items:
            st.warning(
                f"LOW STOCK – {item.product.title()} on {item.shelf_name} "
                f"({item.detected_count}/{item.expected_count})"
            )
        for product, det_shelf, exp_shelf in report.misplaced:
            st.info(
                f"MISPLACED – {product.title()} found on shelf "
                f"'{det_shelf}' (expected '{exp_shelf}')"
            )

    # ── History trend charts ──────────────────────────────────────────
    if history is not None:
        history.record(metrics)
        if len(history) > 1:
            st.divider()
            st.subheader("📈 Trend")
            import pandas as pd  # noqa: PLC0415

            entries = history.entries
            df = pd.DataFrame(
                {
                    "Health Score": [e.health_score for e in entries],
                    "Fill Rate %": [round(e.overall_fill_rate * 100, 1) for e in entries],
                    "Out of Stock": [e.oos_count for e in entries],
                    "Low Stock": [e.low_stock_count for e in entries],
                }
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.line_chart(df[["Health Score", "Fill Rate %"]])
            with col_b:
                st.line_chart(df[["Out of Stock", "Low Stock"]])


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Shelf AI – Inventory Monitor",
        page_icon="🛒",
        layout="wide",
    )
    st.title("🛒 Shelf AI – Retail Inventory Monitor")
    st.caption(
        "Real-time planogram compliance & stock tracking powered by YOLOv8"
    )

    analyzer = load_analyzer()
    checker = load_checker()
    alert_mgr = load_alert_manager()

    # Session-scoped history (persists across reruns within one session)
    if "history" not in st.session_state:
        st.session_state["history"] = StockHistory(max_entries=200)
    history: StockHistory = st.session_state["history"]

    # ── Sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.radio(
            "Input Mode",
            ["📂 Upload Image", "📷 Webcam", "🎬 Demo (no model needed)"],
            index=2,
        )
        weights_path = st.text_input(
            "Model weights path", value=DEFAULT_WEIGHTS
        )
        auto_refresh = st.checkbox("Auto-refresh (webcam)", value=False)
        refresh_secs = st.number_input(
            "Refresh interval (s)", min_value=1, max_value=60, value=5
        )
        st.divider()
        if st.button("🗑️ Clear history"):
            history.clear()
            st.session_state["history"] = StockHistory(max_entries=200)
            st.rerun()

    # ── Main area ─────────────────────────────────────────────────────

    if mode == "🎬 Demo (no model needed)":
        st.info(
            "Running in **demo mode** with a synthetic detection result. "
            "No model weights are required."
        )
        demo_result = _build_demo_result()
        report = analyzer.analyse(demo_result)
        render_shelf_report(report, checker, history)

        # Send demo alerts
        for item in report.out_of_stock_items:
            alert_mgr.send(
                subject=f"[Shelf AI] Out of Stock: {item.product.title()}",
                message=f"{item.product.title()} is OUT OF STOCK on {item.shelf_name}.",
                alert_key=f"{item.shelf_id}:{item.product}:oos",
            )
        for item in report.low_stock_items:
            alert_mgr.send(
                subject=f"[Shelf AI] Low Stock: {item.product.title()}",
                message=(
                    f"{item.product.title()} is LOW ({item.detected_count}/"
                    f"{item.expected_count}) on {item.shelf_name}."
                ),
                alert_key=f"{item.shelf_id}:{item.product}:low",
            )

    elif mode == "📂 Upload Image":
        uploaded = st.file_uploader(
            "Upload a shelf image", type=["jpg", "jpeg", "png"]
        )
        if uploaded:
            import numpy as np  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
            import cv2  # noqa: PLC0415

            img = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            with st.spinner("Running detection…"):
                detector = load_detector(weights_path)
                det_result = detector.predict(frame, draw=True)

            col_img, col_info = st.columns([2, 1])
            with col_img:
                if det_result.annotated_frame is not None:
                    annotated_rgb = cv2.cvtColor(
                        det_result.annotated_frame, cv2.COLOR_BGR2RGB
                    )
                    st.image(annotated_rgb, caption="Detections", use_container_width=True)
                else:
                    st.image(img, caption="Uploaded image", use_container_width=True)

            with col_info:
                st.metric("Total detections", len(det_result.detections))

            report = analyzer.analyse(det_result)
            render_shelf_report(report, checker, history)

    elif mode == "📷 Webcam":
        import cv2  # noqa: PLC0415

        frame_placeholder = st.empty()
        report_placeholder = st.empty()

        run = st.button("▶ Capture frame")
        if run or auto_refresh:
            cap = cv2.VideoCapture(0)
            try:
                ret, frame = cap.read()
                if not ret:
                    st.error("Could not read from webcam.")
                else:
                    with st.spinner("Running detection…"):
                        detector = load_detector(weights_path)
                        det_result = detector.predict(frame, draw=True)

                    if det_result.annotated_frame is not None:
                        rgb = cv2.cvtColor(
                            det_result.annotated_frame, cv2.COLOR_BGR2RGB
                        )
                        frame_placeholder.image(rgb, use_container_width=True)

                    report = analyzer.analyse(det_result)
                    with report_placeholder.container():
                        render_shelf_report(report, checker, history)

            finally:
                cap.release()

            if auto_refresh:
                time.sleep(refresh_secs)
                st.rerun()


if __name__ == "__main__":
    main()
