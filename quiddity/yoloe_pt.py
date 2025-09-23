from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
from ultralytics import YOLO

from common.interfaces import BBox, Detector


def _area_frac(box: BBox, width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    denom = max(1, width * height)
    return area / float(denom)


class YOLOEPT(Detector):
    """Ultralytics YOLO wrapper with global confidence and size gates."""

    def __init__(self, cfg: dict):
        self.global_conf = float(cfg.get("conf_th", 0.35))
        classes: Optional[Iterable[str]] = cfg.get("classes_include")
        self.classes_include: Optional[Set[str]] = set(classes) if classes else None
        self.min_area = float(cfg.get("min_box_area_frac", 0.0))
        self.max_area = float(cfg.get("max_box_area_frac", 1.0))

        model_name = cfg.get("model_name")
        model_path = cfg.get("model_path")

        if model_path and model_name:
            raise ValueError("Specify only one of 'model_path' or 'model_name'.")

        if model_path:
            self.model = YOLO(model_path)
        else:
            # Default to Ultralytics alias download for backwards compatibility
            model_name = model_name or "yolov8s"
            self.model = YOLO(model_name)

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[BBox, float, str]]:
        height, width = frame_bgr.shape[:2]
        res = self.model.predict(frame_bgr, verbose=False)[0]
        out: List[Tuple[BBox, float, str]] = []
        names = getattr(self.model, "names", {})

        for box in res.boxes:
            conf = float(box.conf.item())
            if conf < self.global_conf:
                continue
            cls_id = int(box.cls.item())
            cls_name = names.get(cls_id, str(cls_id))
            if self.classes_include and cls_name not in self.classes_include:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            frac = _area_frac((x1, y1, x2, y2), width, height)
            if frac < self.min_area or frac > self.max_area:
                continue
            out.append(((x1, y1, x2, y2), conf, cls_name))
        return out
