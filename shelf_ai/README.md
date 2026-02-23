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
│   ├── detector.py         # YOLOv8 inference wrapper
│   ├── shelf_analyzer.py   # Shelf zone mapping + stock status logic
│   ├── planogram.py        # Planogram compliance checker
│   └── alerts.py           # Telegram / Email / console alert system
├── train/
│   └── train.py            # YOLOv8 fine-tuning script
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── tests/
│   ├── test_shelf_analyzer.py
│   └── test_alerts.py
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
│  (detector.py)    │  (label, confidence, bbox)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  ShelfAnalyzer    │  Zone mapping + stock counting
│  (shelf_analyzer) │  → ShelfReport
└───────┬───────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌───────────────────┐    ┌──────────────────────┐
│ PlanogramChecker  │    │   AlertManager        │
│ (planogram.py)    │    │   (alerts.py)         │
│ ComplianceReport  │    │ Telegram / Email / log│
└───────┬───────────┘    └──────────────────────┘
        │
        ▼
┌───────────────────┐
│ Streamlit Dashboard│
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
