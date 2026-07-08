"""
YOLO-based VLM trigger.

Runs a lightweight YOLO detector (e.g. yolo11n.pt) on sampled frames and decides
whether to fire an (expensive) VLM call based on whether the set of detected
object classes has changed since the last check.
"""

import logging
import time
from typing import Optional

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

    def detect(self, img_bgr: np.ndarray) -> tuple[bool, frozenset, list[dict]]:
        """
        Run detection on a single BGR frame.

        Returns:
            (triggered, current_classes, detections) where `triggered` is True
            only when the detected class set changed AND the cooldown has elapsed.
            `current_classes` always reflects what was just observed, regardless
            of cooldown, so callers can track scene state accurately.
        """
        results = self.model.predict(img_bgr, conf=self.conf, verbose=False, device=self.device)[0]

        detected = set()
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = self.model.names.get(cls_id, str(cls_id))
            if self.allowed_classes and name not in self.allowed_classes:
                continue
            score = float(box.conf[0])
            detected.add(name)
            detections.append({"class": name, "conf": score})

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
