from __future__ import annotations

import cv2
import numpy as np

from common.interfaces import IdSpecialist


class VehicleSignature(IdSpecialist):
    name = "vehicle.signature"
    classes = ["car", "truck", "bus"]

    def __init__(self, cfg: dict):
        self.hist_bins = int(cfg.get("hist_bins", 32))
        self.channels = cfg.get("channels", [0])
        self.ranges = cfg.get("ranges", [0, 180])

    def wants(self, det_conf, bbox, quality) -> bool:  # type: ignore[override]
        return True

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:  # type: ignore[override]
        if crop_bgr.size == 0:
            raise ValueError("Received empty crop for embedding")
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], self.channels, None, [self.hist_bins], self.ranges)
        hist = hist.flatten().astype(np.float32)
        return hist / (np.linalg.norm(hist) + 1e-9)
