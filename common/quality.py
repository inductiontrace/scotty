from __future__ import annotations

import cv2
import numpy as np


def sharpness_laplacian(gray_roi: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
