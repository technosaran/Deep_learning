# 🛒 Shelf AI – Retail Inventory Monitor

A real-time **Planogram Compliance & Inventory Tracking** system powered by **YOLOv8**.

The system watches a retail shelf (image / live webcam) and answers:

| Question | Output |
|---|---|
| What products are present? | Bounding-box detections with class labels |
| How many are left? | Per-product count vs. expected quantity |
| Which shelf are they on? | Shelf-zone assignment from vertical position |
| Is something missing? | **Out of Stock** / **Low Stock** alerts |
| Is something in the wrong place? | **Planogram violation** with source → target shelf |
| What should staff restock first? | **Priority-ordered restock queue** with urgency scores |

---

## Project Structure

```
shelf_ai/
├── config/
│   ├── planogram.yaml      # Shelf zones, allowed products, expected quantities
│   └── thresholds.yaml     # Stock thresholds, alert settings, model path
├── data/
│   └── README.py           # Dataset preparation instructions
├── src/
│   ├── detector.py         # YOLOv8 inference wrapper (TTA + FP16 support)
│   ├── shelf_analyzer.py   # Shelf zone mapping + stock status logic
│   ├── planogram.py        # Planogram compliance checker
│   ├── alerts.py           # Telegram / Email / console alert system
│   ├── metrics.py          # KPI calculator (fill rate, health score, …)
│   ├── history.py          # Rolling KPI history with JSON persistence
│   ├── smoother.py         # Temporal detection smoother (real-time noise reduction)
│   └── restock.py          # Restock priority planner with urgency scoring
├── train/
│   └── train.py            # YOLOv8 fine-tuning script
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── tests/
│   ├── test_shelf_analyzer.py
│   ├── test_alerts.py
│   ├── test_metrics.py
│   ├── test_history.py
│   ├── test_improvements.py
│   └── test_advanced_features.py
├── demo.py                 # CLI demo script
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r shelf_ai/requirements.txt
```

### 2. Try the demo (no model weights needed)

```bash
# Terminal
python shelf_ai/demo.py --demo

# Dashboard
streamlit run shelf_ai/dashboard/app.py
# → Select "Demo (no model needed)" in the sidebar
```

---

## Full Setup (real shelf detection)

### Step 1 – Collect your dataset

| Requirement | Target |
|---|---|
| Photos | 300–800 shelf images |
| Lighting | Mix of bright / dim / natural |
| Angles | Front, slight side, close, far |
| Occlusion | Partially hidden products |

### Step 2 – Label with Roboflow (recommended)

1. Create a free account at [roboflow.com](https://roboflow.com).
2. Upload your photos.
3. Draw bounding boxes, one class per product SKU (e.g. `maggi`, `colgate`).
4. Export in **YOLOv8** format.
5. Download and place the dataset at `shelf_ai/data/shelf_dataset/`.

Expected layout:
```
shelf_ai/data/shelf_dataset/
├── data.yaml
├── train/images/  train/labels/
├── valid/images/  valid/labels/
└── test/images/   test/labels/
```

> **Tip:** Use the class names that match `planogram.yaml` so stock logic works out of the box.

### Step 3 – Train the model

```bash
python shelf_ai/train/train.py                # default settings
python shelf_ai/train/train.py --epochs 100   # more epochs
python shelf_ai/train/train.py --device 0     # GPU 0
```

Best weights are saved to:
```
shelf_ai/runs/detect/shelf_ai/weights/best.pt
```

### Step 4 – Run on an image

```bash
python shelf_ai/demo.py --source path/to/shelf.jpg
```

### Step 5 – Live webcam

```bash
python shelf_ai/demo.py --webcam
```
Press `q` to quit, `s` to force an immediate re-analysis.

### Step 6 – Dashboard

```bash
streamlit run shelf_ai/dashboard/app.py
```

---

## Configuration

### Planogram (`config/planogram.yaml`)

Defines shelves with:
- `zone_y_range` – vertical fraction of the image the shelf occupies.
- `allowed_products` – SKUs that belong on this shelf.
- `expected_counts` – ideal quantity of each product.

```yaml
shelves:
  shelf_a:
    name: "Shelf A - Snacks & Noodles"
    zone_y_range: [0.0, 0.25]
    allowed_products: [maggi, parleg, lays, goodday, bourbon]
    expected_counts:
      maggi: 8
      lays: 6
      # …
```

### Thresholds (`config/thresholds.yaml`)

```yaml
stock:
  low_stock_ratio: 0.4        # ≤ 40 % of expected → Low Stock
  out_of_stock_count: 0       # == 0 → Out of Stock

alerts:
  telegram:
    enabled: false
    bot_token: ""             # or set TELEGRAM_BOT_TOKEN env var
    chat_id: ""               # or set TELEGRAM_CHAT_ID env var
  email:
    enabled: false
    # …
  cooldown_seconds: 300       # min gap between repeated alerts
```

### Optional Alerts

**Telegram:**
```bash
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"
# Then set alerts.telegram.enabled: true in thresholds.yaml
```

**Email (Gmail example):**
```bash
export EMAIL_SENDER="you@gmail.com"
export EMAIL_PASSWORD="app-password"
export EMAIL_RECIPIENT="manager@store.com"
# Then set alerts.email.enabled: true in thresholds.yaml
```

---

## Advanced Features

### Temporal Detection Smoother (`src/smoother.py`)

In real-time video mode, single frames can produce noisy counts (missed
detections, false positives).  `DetectionSmoother` maintains a rolling window
of the last *N* frames and returns time-averaged counts, reducing variance
without noticeable lag.

```python
from src.smoother import DetectionSmoother

smoother = DetectionSmoother(window=5)
# call once per frame with the raw per-product counts
smoothed_counts = smoother.update({"maggi": 3, "lays": 1})
```

### Restock Priority Planner (`src/restock.py`)

Converts a `ShelfReport` into a ranked task list so staff know exactly which
products to restock first.

| Status | Urgency score |
|---|---|
| Out of Stock | 1.00 (critical) |
| Low Stock | (1 – fill_rate) × 0.70 |

```python
from src.restock import RestockPlanner

planner = RestockPlanner()
tasks = planner.plan(report)
for task in tasks:
    print(task)
# Output example:
#  #1  [1.00]  lays                  Shelf A - Snacks & Noodles   need  6 unit(s)  (Out of Stock)
#  #2  [0.53]  maggi                 Shelf A - Snacks & Noodles   need  6 unit(s)  (Low Stock)
```

### Dashboard Enhancements

- **🛒 Restock Priority Queue** – interactive sortable table in the dashboard
- **⬇️ Export shelf report as CSV** – one-click download for store management
- **📈 Trend charts** – health score and fill-rate trends across sessions

---

## Running Tests

```bash
cd shelf_ai
pytest tests/ -v
```

No GPU or model weights required – tests use synthetic detection results.

---

## Architecture

```
Image / Video Frame
        │
        ▼
┌───────────────────┐
│  ShelfDetector    │  YOLOv8 → List[Detection]
│  (detector.py)    │  (TTA + FP16 supported)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ DetectionSmoother │  Rolling-window mean (real-time noise reduction)
│  (smoother.py)    │  → smoothed per-product counts
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  ShelfAnalyzer    │  Zone mapping + stock counting
│  (shelf_analyzer) │  → ShelfReport
└───────┬───────────┘
        │
        ├──────────────────────────┬──────────────────────┐
        ▼                          ▼                      ▼
┌───────────────────┐    ┌──────────────────┐  ┌────────────────────┐
│ PlanogramChecker  │    │  AlertManager    │  │  RestockPlanner    │
│ (planogram.py)    │    │  (alerts.py)     │  │  (restock.py)      │
│ ComplianceReport  │    │ Telegram / Email │  │  Priority task list│
└───────┬───────────┘    └──────────────────┘  └────────────────────┘
        │
        ▼
┌───────────────────┐
│ Streamlit Dashboard│  KPIs · Restock Queue · CSV export · Trend charts
│ (dashboard/app.py) │
└───────────────────┘
```

---

## Product Classes (Default)

| Shelf | Products |
|---|---|
| A – Snacks & Noodles | maggi, parleg, lays, goodday, bourbon |
| B – Personal Care | colgate, dove, clinicplus, lifebuoy, pepsodent |
| C – Drinks | coke, pepsi, sprite, maaza, thumsup |
| D – Groceries | atta, sugar, salt, dalda, tata_tea |

---

## Industry Keywords

- **Planogram Compliance** – ensuring products are on the correct shelf
- **Retail Shelf Monitoring** – real-time shelf state visibility
- **SKU Detection** – brand-level product identification
- **Restock Automation** – alert-driven restocking workflow
- **Real-time Inventory Visibility** – live count vs. threshold comparison
