"""
detector.py - AI Vehicle Detection Engine
Utilizes YOLOv8 with MAX/Mojo hardware acceleration fallback to detect
vehicles (cars, motorcycles, buses, trucks, bicycles) in video frames.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import numpy as np
import cv2
import torch

import config
from utils.max_accelerator import accelerator

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    YOLO-based Vehicle Detection Engine.
    Filters raw model detections exclusively to designated vehicle classes,
    extracts bounding boxes, confidences, and vehicle centroids.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        conf_threshold: float = config.DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
        target_classes: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the YOLO vehicle detector.

        Args:
            model_name_or_path: Path or filename for YOLO model weights.
            conf_threshold: Minimum confidence score [0.0 - 1.0].
            iou_threshold: Non-Max Suppression IoU threshold [0.0 - 1.0].
            target_classes: Dict mapping class IDs to class names.
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or config.TARGET_VEHICLE_CLASSES
        self.allowed_class_ids = set(self.target_classes.keys())

        # Determine model path
        if model_name_or_path:
            self.model_path = Path(model_name_or_path)
        else:
            self.model_path = config.MODEL_PATH

        self.model = None
        self.is_loaded = False
        self.device = accelerator.device

        # Apply runtime optimizations
        accelerator.optimize_torch_runtime()
        self._load_model()

    def _load_model(self):
        """Load YOLO model weights into memory, handling torch 2.6+ safe‑load restrictions."""
        try:
            from ultralytics import YOLO

            # Register safe globals required by the YOLO checkpoint (torch >=2.6 changes default)
            try:
                import torch
                import ultralytics.nn.tasks
                torch.serialization.add_safe_globals([
                    ultralytics.nn.tasks.DetectionModel,
                    torch.nn.modules.container.Sequential,
                ])
            except Exception as e:
                logger.warning(f"Unable to add torch safe globals: {e}")

            # Determine the path to the YOLO weights – download if missing
            model_file_str = str(self.model_path)
            if not self.model_path.exists():
                logger.info(
                    f"Model weights not found at {self.model_path}. Downloading default {config.DEFAULT_MODEL_NAME}..."
                )
                model_file_str = config.DEFAULT_MODEL_NAME

            self.model = YOLO(model_file_str)
            # Fuse model layers for faster CPU inference if supported
            try:
                self.model.fuse()
            except Exception:
                pass
            self.is_loaded = True
            logger.info(f"YOLO model successfully loaded on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
            self.is_loaded = False

    def set_thresholds(self, conf: float, iou: float):
        """Update detection thresholds dynamically from UI sliders."""
        self.conf_threshold = max(0.05, min(1.0, conf))
        self.iou_threshold = max(0.05, min(1.0, iou))

    def set_allowed_classes(self, allowed_names: Set[str]):
        """Filter detections to a specific subset of vehicle types."""
        self.allowed_class_ids = {
            cid for cid, name in self.target_classes.items()
            if name.lower() in {n.lower() for n in allowed_names}
        }

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run vehicle detection on a single frame.

        Args:
            frame: OpenCV BGR image (numpy array)

        Returns:
            List of detection dictionaries:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str,
                    'centroid': (cx, cy)
                }, ...
            ]
        """
        if not self.is_loaded or self.model is None:
            # Fallback: empty detections if model fails to load
            return []

        try:
            # Run inference with class filtering
            # Classes in COCO: 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
            with torch.no_grad():
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=list(self.allowed_class_ids),
                    device=self.device,
                    verbose=False,
                )

            detections: List[Dict[str, Any]] = []

            if not results or len(results) == 0:
                return detections

            result = results[0]
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                return detections

            # Extract bounding boxes, class ids, and confidences
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                cid = int(clss[i])
                conf = float(confs[i])
                box = xyxy[i].tolist()

                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)

                class_name = self.target_classes.get(cid, "vehicle")

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cid,
                    "class_name": class_name,
                    "centroid": (cx, cy),
                })

            return detections

        except Exception as e:
            logger.error(f"Inference error during detect(): {e}")
            return []
