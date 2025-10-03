from typing import Iterable, List, Optional, Set, Tuple

import logging
import os
import pathlib
import shutil
import tempfile
import urllib.request

import numpy as np
from ultralytics import YOLO

from common.interfaces import BBox, Detector


LOGGER = logging.getLogger(__name__)


def _download_weights(url: str, destination: pathlib.Path) -> None:
    """Download a model artifact to ``destination`` atomically."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading YOLOE weights from %s", url)
    tmp_name = None
    try:
        with urllib.request.urlopen(url) as response:
            with tempfile.NamedTemporaryFile(
                delete=False, dir=str(destination.parent)
            ) as tmp_fh:
                shutil.copyfileobj(response, tmp_fh)
                tmp_name = tmp_fh.name
        os.replace(tmp_name, destination)
    except Exception:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


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
        model_url = cfg.get("model_url")

        if model_path and model_name:
            raise ValueError("Specify only one of 'model_path' or 'model_name'.")
        if model_url and not model_path:
            raise ValueError("'model_url' requires 'model_path' to be set.")

        if model_path:
            weights_path = pathlib.Path(model_path)
            if not weights_path.exists():
                if not model_url:
                    raise FileNotFoundError(
                        f"Model path '{weights_path}' does not exist and no 'model_url' was provided."
                    )
                _download_weights(str(model_url), weights_path)
            self.model = YOLO(str(weights_path))
        else:
            if not model_name:
                raise ValueError("Specify either 'model_path' or 'model_name'.")
            self.model = YOLO(str(model_name))

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
