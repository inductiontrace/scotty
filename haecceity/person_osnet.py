from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

from common.interfaces import BBox, IdSpecialist


class PersonOSNet025(IdSpecialist):
    name = "person.osnet_x025"
    classes = ["person"]

    def __init__(self, cfg: dict):
        self.model_path = cfg["model_path"]
        self.crop = int(cfg.get("crop", 192))
        self.min_h_frac = float(cfg.get("min_bbox_h_frac", 0.08))
        self.min_sharp = float(cfg.get("min_sharpness", 25.0))
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def wants(self, det_conf: float, bbox: BBox, quality) -> bool:  # type: ignore[override]
        hfrac = quality.get("bbox_h_frac", 0.0) if quality else 0.0
        sharp = quality.get("sharpness", 0.0) if quality else 0.0
        return (hfrac >= self.min_h_frac) and (sharp >= self.min_sharp)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:  # type: ignore[override]
        if crop_bgr.size == 0:
            raise ValueError("Received empty crop for embedding")
        height, width = crop_bgr.shape[:2]
        side = min(height, width)
        y0 = (height - side) // 2
        x0 = (width - side) // 2
        square = crop_bgr[y0 : y0 + side, x0 : x0 + side]
        img = cv2.resize(square, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        embedding = self.session.run([self.output_name], {self.input_name: x})[0]
        embedding = np.asarray(embedding, dtype=np.float32).flatten()
        embedding /= np.linalg.norm(embedding) + 1e-9
        return embedding
