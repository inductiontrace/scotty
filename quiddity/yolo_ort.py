from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from common.interfaces import BBox, Detector


def _letterbox(im: np.ndarray, new_side: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    height, width = im.shape[:2]
    ratio = min(new_side / height, new_side / width)
    new_h, new_w = int(round(height * ratio)), int(round(width * ratio))
    resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = (new_side - new_h) // 2
    bottom = new_side - new_h - top
    left = (new_side - new_w) // 2
    right = new_side - new_w - left
    out = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return out, ratio, (left, top)


class YOLOv8ORT(Detector):
    def __init__(self, cfg: dict):
        self.model_path = cfg["model_path"]
        self.img_size = int(cfg.get("img_size", 512))
        self.conf_th = float(cfg.get("conf_th", 0.35))
        self.iou_th = float(cfg.get("iou_th", 0.5))
        self.classes_include = cfg.get("classes_include")
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # COCO index->name (trimmed list; extend as needed)
        self._coco = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[BBox, float, str]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img, ratio, (dx, dy) = _letterbox(rgb, self.img_size)
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        predictions = self.session.run([self.output_name], {self.input_name: x})[0]
        predictions = np.squeeze(predictions, axis=0)

        boxes = predictions[:, :4]  # cx, cy, w, h
        obj_conf = predictions[:, 4]
        class_scores = predictions[:, 5:]
        class_ids = class_scores.argmax(1)
        class_conf = class_scores.max(1)
        confidences = obj_conf * class_conf

        results: List[Tuple[BBox, float, str]] = []
        frame_h, frame_w = frame_bgr.shape[:2]
        for (cx, cy, w, h), score, cid in zip(boxes, confidences, class_ids):
            if score < self.conf_th:
                continue
            label = self._coco.get(int(cid), str(int(cid)))
            if self.classes_include and label not in self.classes_include:
                continue
            x1 = int((cx - w / 2 - dx) / ratio)
            y1 = int((cy - h / 2 - dy) / ratio)
            x2 = int((cx + w / 2 - dx) / ratio)
            y2 = int((cy + h / 2 - dy) / ratio)
            x1 = max(0, min(frame_w - 1, x1))
            x2 = max(0, min(frame_w - 1, x2))
            y1 = max(0, min(frame_h - 1, y1))
            y2 = max(0, min(frame_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(((x1, y1, x2, y2), float(score), label))

        if not results:
            return results

        boxes_xyxy = np.array([bbox for bbox, _, _ in results], dtype=np.float32)
        scores = np.array([score for _, score, _ in results], dtype=np.float32)
        boxes_xywh = np.column_stack(
            [
                boxes_xyxy[:, 0],
                boxes_xyxy[:, 1],
                boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
                boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
            ]
        )
        keep = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), self.conf_th, self.iou_th)
        if len(keep) == 0:
            return []
        keep_indices = [int(k[0]) if isinstance(k, (list, tuple, np.ndarray)) else int(k) for k in keep]
        return [results[i] for i in keep_indices]
