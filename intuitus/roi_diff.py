from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from common.interfaces import BBox, ROIProvider


class DiffROI(ROIProvider):
    def __init__(self, cfg: dict):
        self.keyframe_sec = float(cfg.get("keyframe_sec", 2.0))
        self.min_blob_area = int(cfg.get("min_blob_area", 300))
        self.dilate = int(cfg.get("dilate", 11))
        self.max_roi = int(cfg.get("max_roi", 8))
        self.max_area_frac = float(cfg.get("max_area_frac", 0.4))
        self.prev: np.ndarray | None = None
        self.t_last_key = 0.0

    def next_rois(self, frame_bgr: np.ndarray) -> Optional[List[BBox]]:
        now = time.time()
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev is None:
            self.prev = gray
            self.t_last_key = now
            return None

        keyframe = (now - self.t_last_key) > self.keyframe_sec
        if keyframe:
            self.t_last_key = now

        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray

        blur = cv2.GaussianBlur(diff, (5, 5), 0)
        _, mask = cv2.threshold(blur, 12, 255, cv2.THRESH_BINARY)
        if self.dilate > 0:
            kernel = np.ones((self.dilate, self.dilate), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois: List[BBox] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_blob_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            rois.append((x, y, x + w, y + h))

        rois = sorted(rois, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)[
            : self.max_roi
        ]
        total_roi_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in rois)
        area_frac = total_roi_area / float(width * height + 1e-9)

        busy = (area_frac > self.max_area_frac) or (len(rois) == 0)
        if keyframe or busy:
            return None
        return rois
