"""
YOLO-based VLM trigger.

Runs a lightweight YOLO detector (e.g. yolo11n.pt) on sampled frames and decides
whether to fire an (expensive) VLM call based on whether the set of detected
object classes has changed since the last check.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class YoloTrigger:
    """Detects objects with YOLO and reports when the detected class set changes."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.4,
        allowed_classes: Optional[set] = None,
        cooldown: float = 2.0,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO
        import torch

        self.conf = conf
        self.allowed_classes = allowed_classes  # None = trigger on any class
        self.cooldown = cooldown
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading YOLO model '{model_path}' on device '{self.device}'...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        logger.info(f"YOLO model loaded: {model_path}")

        self.last_classes: frozenset = frozenset()
        self.last_trigger_time: float = 0.0

    @staticmethod
    def _simplify_polygon(points: np.ndarray, epsilon_frac: float = 0.01) -> list:
        """Reduce a normalized (x, y) contour to a small set of points for cheap JSON/WS transport."""
        if points.shape[0] < 3:
            return np.round(points, 4).tolist()
        contour = points.reshape(-1, 1, 2).astype(np.float32)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_frac * peri, True)
        return np.round(approx.reshape(-1, 2), 4).tolist()

    def detect(self, img_bgr: np.ndarray) -> tuple[bool, frozenset, list[dict]]:
        """
        Run detection on a single BGR frame.

        Returns:
            (triggered, current_classes, detections) where `triggered` is True
            only when the detected class set changed AND the cooldown has elapsed.
            `current_classes` always reflects what was just observed, regardless
            of cooldown, so callers can track scene state accurately.
            Each entry in `detections` includes a normalized ("box": [x1, y1, x2, y2],
            0-1 relative to frame size) and, for segmentation models, a simplified
            "mask" polygon (also normalized) for client-side overlay drawing.
        """
        results = self.model.predict(img_bgr, conf=self.conf, verbose=False, device=self.device)[0]
        masks_xyn = results.masks.xyn if results.masks is not None else None

        detected = set()
        detections = []
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            name = self.model.names.get(cls_id, str(cls_id))
            if self.allowed_classes and name not in self.allowed_classes:
                continue
            score = float(box.conf[0])
            detected.add(name)
            det = {
                "class": name,
                "conf": round(score, 3),
                "box": np.round(box.xyxyn[0].tolist(), 4).tolist(),
            }
            if masks_xyn is not None and i < len(masks_xyn):
                det["mask"] = self._simplify_polygon(masks_xyn[i])
            detections.append(det)

        current_classes = frozenset(detected)
        changed = current_classes != self.last_classes

        triggered = False
        if changed:
            now = time.time()
            if (now - self.last_trigger_time) >= self.cooldown:
                triggered = True
                self.last_trigger_time = now
            self.last_classes = current_classes

        return triggered, current_classes, detections
