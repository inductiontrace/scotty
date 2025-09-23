from typing import List, Tuple
import numpy as np
from ultralytics import YOLO
from common.interfaces import Detector, BBox


class YOLOEPT(Detector):
    """
    Ultralytics YOLO wrapper that loads by alias or URL (auto-downloads).
    Config keys:
      quiddity:
        impl: "quiddity.yoloe_pt:YOLOEPT"
        model_name: "yolov8s"        # alias; Ultralytics downloads + caches
        conf_th: 0.35
        classes_include: ["person"]  # optional filter
    """

    def __init__(self, cfg: dict):
        self.conf_th = float(cfg.get("conf_th", 0.35))
        self.classes_include = cfg.get("classes_include")
        model_name = cfg.get("model_name", "yolov8s")  # default alias
        self.model = YOLO(model_name)  # no file path needed

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[BBox, float, str]]:
        res = self.model.predict(frame_bgr, verbose=False)[0]
        out: List[Tuple[BBox, float, str]] = []
        for b in res.boxes:
            conf = float(b.conf.item())
            if conf < self.conf_th:
                continue
            cls_id = int(b.cls.item())
            cls_name = self.model.names.get(cls_id, str(cls_id))
            if self.classes_include and cls_name not in self.classes_include:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            out.append(((x1, y1, x2, y2), conf, cls_name))
        return out
