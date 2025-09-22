from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from common.interfaces import BBox, Detector


class YOLOEPT(Detector):
    """
    Wraps Ultralytics YOLOE (PyTorch .pt weights).

    Config keys (example):
      quiddity:
        impl: "quiddity.yoloe_pt:YOLOEPT"
        model_path: "models/yoloe-v8-S.pt"
        conf_th: 0.35
        classes_include: ["person"]
    """

    def __init__(self, cfg: dict):
        self.model_path = cfg["model_path"]
        self.conf_th = float(cfg.get("conf_th", 0.35))
        self.classes_include = cfg.get("classes_include")
        self.model = YOLO(self.model_path)

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[BBox, float, str]]:
        results = self.model.predict(frame_bgr, verbose=False)[0]
        out: List[Tuple[BBox, float, str]] = []
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < self.conf_th:
                continue
            cls_id = int(box.cls.item())
            cls_name = self.model.names.get(cls_id, str(cls_id))
            if self.classes_include and cls_name not in self.classes_include:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            out.append(((x1, y1, x2, y2), conf, cls_name))
        return out
