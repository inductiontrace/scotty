"""Lightweight SORT tracker for temporal durability."""

from __future__ import annotations

from typing import List, Tuple

from common import numpy_compat  # noqa: F401  Ensures NumPy compatibility shims are applied.
import numpy as np
from filterpy.kalman import KalmanFilter

BBox = Tuple[int, int, int, int]


def iou(bb_test: BBox, bb_gt: BBox) -> float:
    """Intersection over Union between two boxes."""

    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, float(xx2 - xx1))
    h = max(0.0, float(yy2 - yy1))
    inter = w * h
    area = (
        (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        - inter
    )
    return inter / (area + 1e-9)


class KalmanBoxTracker:
    """Internal Kalman state for a single track."""

    count = 0

    def __init__(self, bbox: BBox, label: str, score: float):
        # state: [cx, cy, s, r, vx, vy, vs]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.label = label
        self.score = score
        self._update_bbox(bbox)

    @staticmethod
    def xyxy_to_cxysr(bbox: BBox) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = w * h
        r = w / (h + 1e-9)
        return np.array([cx, cy, s, r], dtype=float)

    @staticmethod
    def cxysr_to_xyxy(x: np.ndarray) -> BBox:
        cx, cy, s, r = x
        w = np.sqrt(max(s * r, 0.0))
        h = s / (w + 1e-9)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return int(x1), int(y1), int(x2), int(y2)

    def _update_bbox(self, bbox: BBox) -> None:
        self.kf.x[:4] = self.xyxy_to_cxysr(bbox).reshape(-1, 1)

    def predict(self) -> BBox:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.cxysr_to_xyxy(self.kf.x[:4].reshape(-1))

    def update(self, bbox: BBox, label: str, score: float) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.label = label
        self.score = score
        measurement = self.xyxy_to_cxysr(bbox)
        self.kf.update(measurement)

    def get_state(self) -> Tuple[BBox, str, float]:
        bbox = self.cxysr_to_xyxy(self.kf.x[:4].reshape(-1))
        return bbox, self.label, self.score


class SORT:
    """Simple Online Realtime Tracking."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10, min_hits: int = 2):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, detections: List[Tuple[BBox, float, str]]) -> List[Tuple[int, BBox, str, float]]:
        """Update tracker state.

        Args:
            detections: List of tuples ``((x1, y1, x2, y2), confidence, label)``.

        Returns:
            List of tuples ``(track_id, bbox, label, score)`` for active tracks.
        """

        predictions: List[Tuple[KalmanBoxTracker, BBox]] = []
        for tracker in self.trackers:
            predictions.append((tracker, tracker.predict()))

        if detections and predictions:
            iou_matrix = np.zeros((len(detections), len(predictions)), dtype=float)
            for det_idx, (bbox, _, _) in enumerate(detections):
                for trk_idx, (_, trk_bbox) in enumerate(predictions):
                    iou_matrix[det_idx, trk_idx] = iou(bbox, trk_bbox)

            assigned_dets = set()
            assigned_trks = set()
            pairs: List[Tuple[int, int]] = []
            while True:
                det_idx, trk_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[det_idx, trk_idx] < self.iou_threshold:
                    break
                pairs.append((det_idx, trk_idx))
                assigned_dets.add(det_idx)
                assigned_trks.add(trk_idx)
                iou_matrix[det_idx, :] = -1
                iou_matrix[:, trk_idx] = -1

            for det_idx, trk_idx in pairs:
                bbox, score, label = detections[det_idx]
                tracker, _ = predictions[trk_idx]
                tracker.update(bbox, label, score)

            for det_idx, (bbox, score, label) in enumerate(detections):
                if det_idx in assigned_dets:
                    continue
                self.trackers.append(KalmanBoxTracker(bbox, label, score))

            survivors: List[KalmanBoxTracker] = []
            for trk_idx, (tracker, _) in enumerate(predictions):
                if trk_idx in assigned_trks:
                    survivors.append(tracker)
                else:
                    tracker.time_since_update += 1
                    tracker.hit_streak = max(0, tracker.hit_streak - 1)
                    if tracker.time_since_update <= self.max_age:
                        survivors.append(tracker)
            self.trackers = survivors

        else:
            for tracker in list(self.trackers):
                tracker.time_since_update += 1

            for bbox, score, label in detections:
                self.trackers.append(KalmanBoxTracker(bbox, label, score))

            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        outputs: List[Tuple[int, BBox, str, float]] = []
        survivors: List[KalmanBoxTracker] = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits or tracker.time_since_update == 0:
                bbox, label, score = tracker.get_state()
                outputs.append((tracker.id, bbox, label, score))
            if tracker.time_since_update <= self.max_age:
                survivors.append(tracker)
        self.trackers = survivors
        return outputs
