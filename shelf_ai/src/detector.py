"""
detector.py
-----------
YOLOv8-based product detector for shelf monitoring.

Wraps Ultralytics YOLOv8 to provide a clean interface that returns
structured detection results used by the rest of the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    """Single bounding-box detection result."""

    label: str
    confidence: float
    # Normalised [0, 1] coordinates: (x_centre, y_centre, width, height)
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def xyxy_norm(self):
        """Return (x1, y1, x2, y2) in normalised [0, 1] coordinates."""
        half_w = self.width / 2
        half_h = self.height / 2
        return (
            self.x_center - half_w,
            self.y_center - half_h,
            self.x_center + half_w,
            self.y_center + half_h,
        )


@dataclass
class DetectionResult:
    """Full detection result for one image/frame."""

    detections: List[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    # Annotated frame (numpy array, BGR) – populated if draw=True
    annotated_frame: Optional[np.ndarray] = None


class ShelfDetector:
    """
    Loads a YOLOv8 model and runs inference on images or video frames.

    Parameters
    ----------
    weights_path : str | Path
        Path to a YOLOv8 ``*.pt`` weights file.
    confidence : float
        Minimum confidence threshold for detections.
    iou : float
        IOU threshold for non-maximum suppression.
    device : str
        Inference device – ``"cpu"``, ``"cuda"``, or GPU index string.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.45,
        iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.weights_path = Path(weights_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self._model = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load the YOLO model on first use."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install it with: pip install ultralytics"
            ) from exc

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}\n"
                "Run the training script first:  python train/train.py"
            )
        self._model = YOLO(str(self.weights_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        source: "str | Path | np.ndarray",
        draw: bool = False,
    ) -> DetectionResult:
        """
        Run detection on *source* (file path, URL, or numpy BGR array).

        Parameters
        ----------
        source :
            Image source accepted by ``YOLO.predict()``.
        draw :
            If ``True`` the annotated frame is stored in the result.

        Returns
        -------
        DetectionResult
        """
        self._load_model()

        results = self._model.predict(
            source,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        # ultralytics returns a list even for single images
        yolo_result = results[0]
        h, w = yolo_result.orig_shape

        detections: List[Detection] = []
        boxes = yolo_result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                label = yolo_result.names[cls_id]
                conf = float(box.conf[0])
                # xywhn = normalised x_c, y_c, w, h
                xywhn = box.xywhn[0].tolist()
                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        x_center=xywhn[0],
                        y_center=xywhn[1],
                        width=xywhn[2],
                        height=xywhn[3],
                    )
                )

        annotated = yolo_result.plot() if draw else None

        return DetectionResult(
            detections=detections,
            image_width=w,
            image_height=h,
            annotated_frame=annotated,
        )

    @property
    def class_names(self) -> List[str]:
        """Return list of class names from the loaded model."""
        self._load_model()
        return list(self._model.names.values())
