# Dataset Preparation Guide

## Overview

Place your labelled dataset here before training.

## Expected Layout

```
shelf_ai/data/shelf_dataset/
├── data.yaml             ← dataset configuration (see below)
├── train/
│   ├── images/           ← training images (.jpg / .png)
│   └── labels/           ← YOLO format .txt label files
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## YOLO Label Format

Each `.txt` file has one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalised to `[0, 1]` relative to image size.

## data.yaml Example

```yaml
path: ./data/shelf_dataset
train: train/images
val:   valid/images
test:  test/images

nc: 20
names:
  - maggi
  - parleg
  - lays
  - goodday
  - bourbon
  - colgate
  - dove
  - clinicplus
  - lifebuoy
  - pepsodent
  - coke
  - pepsi
  - sprite
  - maaza
  - thumsup
  - atta
  - sugar
  - salt
  - dalda
  - tata_tea
```

## Recommended Labelling Tools

- **Roboflow** (easiest) – [roboflow.com](https://roboflow.com)
- **LabelImg** – `pip install labelImg && labelImg`

## Data Collection Tips

| Factor | Recommendation |
|---|---|
| Total photos | 300–800 |
| Lighting | Mix bright / dim / natural |
| Angles | Front, slight side, close, far |
| Occlusion | Partially hidden products |
| Class split | 70% train / 20% val / 10% test |
