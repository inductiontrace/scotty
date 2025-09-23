"""Lightweight SORT tracker for temporal durability."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from common import numpy_compat  # noqa: F401  Ensures NumPy compatibility shims are applied.
import numpy as np
from filterpy.kalman import KalmanFilter

from common.geometry import box_center, core_center, dilate_box
from tracker.costs import class_mismatch_penalty, normalized_l2, within_asym_scale_gate

BBox = Tuple[float, float, float, float]


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
        self.last_detection: BBox = tuple(map(float, bbox))
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
        return float(x1), float(y1), float(x2), float(y2)

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
        self.last_detection = tuple(map(float, bbox))
        measurement = self.xyxy_to_cxysr(bbox)
        self.kf.update(measurement)

    def get_state(self) -> Tuple[BBox, str, float]:
        bbox = self.cxysr_to_xyxy(self.kf.x[:4].reshape(-1))
        ix1, iy1, ix2, iy2 = bbox
        return (int(ix1), int(iy1), int(ix2), int(iy2)), self.label, self.score

    def current_bbox(self) -> BBox:
        return self.cxysr_to_xyxy(self.kf.x[:4].reshape(-1))

    def is_active(self) -> bool:
        return self.time_since_update == 0

    def absorb(self, other: "KalmanBoxTracker") -> None:
        self.hits = max(self.hits, other.hits)
        self.hit_streak = max(self.hit_streak, other.hit_streak)
        self.score = max(self.score, other.score)
        if other.time_since_update == 0:
            self.last_detection = other.last_detection


def _merge_dict(defaults: Dict[str, float | bool], override: Dict[str, float | bool] | None) -> Dict[str, float | bool]:
    cfg = dict(defaults)
    if override:
        cfg.update({k: override[k] for k in override if k in cfg})
        for key, value in override.items():
            if key not in cfg:
                cfg[key] = value
    return cfg


class SORT:
    """Simple Online Realtime Tracking with prop-robust association tweaks."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        iou_th: float | None = None,
        max_age: int = 10,
        min_hits: int = 2,
        class_consistency: Dict[str, float | bool] | None = None,
        core_center: Dict[str, float | bool] | None = None,
        scale_gate: Dict[str, float | bool] | None = None,
        dilate_for_assoc: Dict[str, float | bool] | None = None,
        duplicate_merge: Dict[str, float | bool] | None = None,
    ):
        self.iou_threshold = float(iou_threshold if iou_th is None else iou_th)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.trackers: List[KalmanBoxTracker] = []
        self.class_consistency = _merge_dict(
            {"enabled": False, "penalty": 1_000_000.0}, class_consistency
        )
        self.core_center = _merge_dict({"enabled": False, "shrink": 0.3}, core_center)
        self.scale_gate = _merge_dict(
            {"enabled": False, "grow_tol": 1.6, "shrink_tol": 0.6}, scale_gate
        )
        self.dilate_for_assoc = _merge_dict(
            {"enabled": False, "dx": 0.15, "dy": 0.05}, dilate_for_assoc
        )
        self.duplicate_merge = _merge_dict(
            {"enabled": False, "iou_thresh": 0.6, "min_frames": 3, "same_class_only": True},
            duplicate_merge,
        )

    def update(
        self,
        detections: List[Tuple[BBox, float, str]],
        img_size: Tuple[int, int] | None = None,
    ) -> List[Tuple[int, BBox, str, float]]:
        """Update tracker state.

        Args:
            detections: List of tuples ``((x1, y1, x2, y2), confidence, label)``.
            img_size: Optional ``(width, height)`` tuple for normalization.

        Returns:
            List of tuples ``(track_id, bbox, label, score)`` for active tracks.
        """

        if img_size is not None:
            frame_norm: Iterable[float] | None = (max(1.0, img_size[0]), max(1.0, img_size[1]))
        else:
            frame_norm = None

        predictions: List[Tuple[KalmanBoxTracker, BBox]] = []
        for tracker in self.trackers:
            predictions.append((tracker, tracker.predict()))

        unmatched_tracks = set(range(len(predictions)))
        unmatched_dets = set(range(len(detections)))
        matches: List[Tuple[int, int]] = []

        if detections and predictions:
            track_order = sorted(range(len(predictions)), key=lambda idx: -predictions[idx][0].age)
            for trk_idx in track_order:
                tracker, pred_box = predictions[trk_idx]
                assoc_box = pred_box
                pred_w = max(1.0, pred_box[2] - pred_box[0])
                pred_h = max(1.0, pred_box[3] - pred_box[1])

                if self.dilate_for_assoc.get("enabled") and tracker.label == "person":
                    assoc_box = dilate_box(
                        *assoc_box,
                        float(self.dilate_for_assoc.get("dx", 0.0)),
                        float(self.dilate_for_assoc.get("dy", 0.0)),
                    )

                tcx, tcy = box_center(*assoc_box)
                if self.core_center.get("enabled") and tracker.label == "person":
                    tcx, tcy = core_center(
                        *assoc_box,
                        float(self.core_center.get("shrink", 0.3)),
                    )

                best_det = None
                best_score = -np.inf
                best_iou = 0.0
                for det_idx in list(unmatched_dets):
                    det_bbox, det_score, det_label = detections[det_idx]
                    det_box = tuple(map(float, det_bbox))

                    if self.class_consistency.get("enabled"):
                        penalty = class_mismatch_penalty(
                            tracker.label,
                            det_label,
                            float(self.class_consistency.get("penalty", 0.0)),
                        )
                        if penalty > 0:
                            continue

                    if self.scale_gate.get("enabled") and tracker.label == "person":
                        det_w = max(1.0, det_box[2] - det_box[0])
                        det_h = max(1.0, det_box[3] - det_box[1])
                        if not within_asym_scale_gate(
                            (pred_w, pred_h),
                            (det_w, det_h),
                            float(self.scale_gate.get("grow_tol", 1.6)),
                            float(self.scale_gate.get("shrink_tol", 0.6)),
                        ):
                            continue

                    dcx, dcy = box_center(*det_box)
                    if self.core_center.get("enabled") and det_label == "person":
                        dcx, dcy = core_center(
                            *det_box,
                            float(self.core_center.get("shrink", 0.3)),
                        )

                    iou_score = iou(det_box, assoc_box)
                    if iou_score <= 0:
                        continue

                    norm = frame_norm or (pred_w, pred_h)
                    center_cost = normalized_l2((tcx, tcy), (dcx, dcy), norm)
                    score = iou_score - center_cost
                    if score > best_score:
                        best_score = score
                        best_det = det_idx
                        best_iou = iou_score

                if best_det is not None and best_iou >= self.iou_threshold:
                    matches.append((trk_idx, best_det))
                    unmatched_tracks.discard(trk_idx)
                    unmatched_dets.discard(best_det)

        for trk_idx, det_idx in matches:
            tracker, _ = predictions[trk_idx]
            det_bbox, det_score, det_label = detections[det_idx]
            tracker.update(tuple(map(float, det_bbox)), det_label, det_score)

        for trk_idx in unmatched_tracks:
            tracker, _ = predictions[trk_idx]
            tracker.hit_streak = max(0, tracker.hit_streak - 1)

        for det_idx in sorted(unmatched_dets):
            det_bbox, det_score, det_label = detections[det_idx]
            self.trackers.append(
                KalmanBoxTracker(tuple(map(float, det_bbox)), det_label, det_score)
            )

        survivors: List[KalmanBoxTracker] = []
        for tracker in self.trackers:
            if tracker.time_since_update <= self.max_age:
                survivors.append(tracker)
        self.trackers = survivors

        if self.duplicate_merge.get("enabled"):
            removed_ids = merge_nearby_tracks(
                self.trackers,
                float(self.duplicate_merge.get("iou_thresh", 0.6)),
                int(self.duplicate_merge.get("min_frames", 3)),
                bool(self.duplicate_merge.get("same_class_only", True)),
            )
            if removed_ids:
                self.trackers = [t for t in self.trackers if t.id not in removed_ids]

        outputs: List[Tuple[int, BBox, str, float]] = []
        survivors = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits or tracker.time_since_update == 0:
                bbox, label, score = tracker.get_state()
                outputs.append((tracker.id, bbox, label, score))
            if tracker.time_since_update <= self.max_age:
                survivors.append(tracker)
        self.trackers = survivors
        return outputs


def merge_nearby_tracks(
    tracks: Iterable[KalmanBoxTracker],
    iou_thresh: float,
    min_frames: int,
    same_class_only: bool = True,
) -> set[int]:
    alive = [t for t in tracks if t.is_active() and t.hits >= min_frames]
    alive.sort(key=lambda t: -t.age)
    removed: set[int] = set()
    for i, anchor in enumerate(alive):
        if anchor.id in removed:
            continue
        anchor_box = anchor.current_bbox()
        for other in alive[i + 1 :]:
            if other.id in removed or other.id == anchor.id:
                continue
            if other.hits < min_frames:
                continue
            if same_class_only and anchor.label != other.label:
                continue
            if iou(anchor_box, other.current_bbox()) >= iou_thresh:
                anchor.absorb(other)
                removed.add(other.id)
    return removed
