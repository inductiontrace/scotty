from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]


class ROIProvider:
    """Optional Intuitus interface: return ROIs or None to request full-frame."""

    def next_rois(self, frame_bgr: np.ndarray) -> Optional[List[BBox]]:
        raise NotImplementedError


class Detector:
    """Quiddity interface: detect objects in a frame or ROI."""

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[BBox, float, str]]:
        """Return list of ``((x1, y1, x2, y2), conf, class_name)``."""

        raise NotImplementedError


class IdSpecialist:
    """Haecceity per-class ID model interface."""

    name: str = "generic"
    classes: List[str] = []

    def wants(self, det_conf: float, bbox: BBox, quality: Dict) -> bool:
        return True

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return L2-normalized embedding vector ``(D,)``."""

        raise NotImplementedError

    def compare(self, e1: np.ndarray, e2: np.ndarray) -> float:
        """Return similarity score (cosine by default)."""

        e1 = e1 / (np.linalg.norm(e1) + 1e-9)
        e2 = e2 / (np.linalg.norm(e2) + 1e-9)
        return float((e1 * e2).sum())


