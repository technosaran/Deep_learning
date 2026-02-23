# Deep_learning

## Shelf AI – Retail Inventory Monitor

An end-to-end deep learning system for **real-time retail shelf monitoring** powered by YOLOv8.

### Features

| Module | Description |
|---|---|
| `src/detector.py` | YOLOv8-based product detector (lazy-loaded, CPU/GPU) |
| `src/shelf_analyzer.py` | Maps detections → shelf zones, computes stock status & fill rate |
| `src/planogram.py` | Planogram compliance checker (misplacement detection) |
| `src/alerts.py` | Multi-channel alerting (console, Telegram, Email) with cooldown |
| `src/metrics.py` | KPI calculator — fill rate, compliance rate, health score |
| `src/history.py` | In-memory + JSON-persistent trend history tracker |
| `dashboard/app.py` | Streamlit dashboard with KPI cards & trend charts |
| `train/train.py` | YOLOv8 fine-tuning script |

### Quick Start

```bash
cd shelf_ai
pip install -r requirements.txt

# Synthetic demo (no model weights needed)
python demo.py --demo

# Streamlit dashboard
streamlit run dashboard/app.py
```

### KPI Metrics

The `MetricsCalculator` computes a **health score** (0–100) from:
- **Fill rate** (60 %) – ratio of detected to expected products
- **Compliance rate** (40 %) – fraction of products placed on the correct shelf

### Training

```bash
python train/train.py \
    --data data/shelf_dataset/data.yaml \
    --epochs 100 \
    --workers 8 \
    --cache
```

New flags: `--workers`, `--resume`, `--cache`, `--freeze`

### Tests

```bash
cd shelf_ai && python -m pytest tests/ -v
```
