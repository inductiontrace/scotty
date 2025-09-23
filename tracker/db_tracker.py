"""Distance-based tracker tuned for low frame rates."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List, Optional, Sequence, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


BBox = Tuple[int, int, int, int]
Det = Tuple[BBox, float, str]


def xyxy_to_cxysr(bbox: BBox) -> np.ndarray:
    """Convert ``(x1, y1, x2, y2)`` box to SORT state order."""

    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    s = w * h
    r = w / (h + 1e-9)
    return np.array([cx, cy, s, r], dtype=float)


def cxysr_to_xyxy(state: Sequence[float]) -> BBox:
    """Convert SORT state order back to integer box."""

    cx, cy, s, r = state
    w = sqrt(max(1e-9, s * r))
    h = max(1e-9, s / (w + 1e-9))
    x1 = int(round(cx - 0.5 * w))
    y1 = int(round(cy - 0.5 * h))
    x2 = int(round(cx + 0.5 * w))
    y2 = int(round(cy + 0.5 * h))
    return (x1, y1, x2, y2)


def chi2_inv_4d(prob: float) -> float:
    """Return the chi-square inverse value for 4 DoF at ``prob``."""

    if prob >= 0.999:
        return 23.93
    if prob >= 0.997:
        return 18.47
    if prob >= 0.99:
        return 13.28
    if prob >= 0.95:
        return 9.49
    return 7.78


class KfBox:
    """7D Kalman filter that mirrors SORT's state layout."""

    def __init__(self, z0: np.ndarray) -> None:
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        F = np.eye(7)
        F[0, 4] = 1.0
        F[1, 5] = 1.0
        F[2, 6] = 1.0
        self.kf.F = F

        H = np.zeros((4, 7))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        self.kf.H = H

        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.R = np.diag([1.0, 1.0, 10.0, 1.0])
        self.kf.Q = np.eye(7) * 0.01

        self.kf.x[:4, 0] = z0

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        self.kf.predict()
        return self.kf.x[:4, 0].copy(), self.kf.P.copy()

    def update(self, z: np.ndarray) -> None:
        self.kf.update(z)

    def measurement(self) -> np.ndarray:
        return self.kf.x[:4, 0].copy()


@dataclass
class Track:
    bbox: BBox
    label: str
    score: float

    id: int
    hits: int
    missed: int
    kf: KfBox

    embed: Optional[np.ndarray] = None
    last_embed_frame: int = -9999

    @classmethod
    def from_detection(cls, bbox: BBox, label: str, score: float, next_id: int) -> "Track":
        z0 = xyxy_to_cxysr(bbox)
        kf = KfBox(z0)
        return cls(bbox, label, score, next_id, 1, 0, kf)

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        z, cov = self.kf.predict()
        self.bbox = cxysr_to_xyxy(z)
        return z, cov

    def update(self, bbox: BBox, label: str, score: float) -> None:
        z = xyxy_to_cxysr(bbox)
        self.kf.update(z)
        self.bbox = cxysr_to_xyxy(self.kf.measurement())
        self.label = label
        self.score = 0.8 * self.score + 0.2 * score
        self.hits += 1
        self.missed = 0


class DBTracker:
    """Distance-based association tracker suitable for low FPS streams."""

    def __init__(
        self,
        iou_threshold: float = 0.0,
        max_age: int = 10,
        min_hits: int = 2,
        center_gate_frac: float = 0.12,
        maha_gate_p: float = 0.997,
        w_center: float = 1.0,
        w_scale: float = 0.2,
        w_aspect: float = 0.2,
        w_app: float = 0.0,
    ) -> None:
        del iou_threshold  # parity with SORT config but unused

        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.center_gate_frac = float(center_gate_frac)
        self.maha_gate2 = chi2_inv_4d(float(maha_gate_p))
        self.w_center = float(w_center)
        self.w_scale = float(w_scale)
        self.w_aspect = float(w_aspect)
        self.w_app = float(w_app)

        self._tracks: List[Track] = []
        self._next_id = 1
        self._img_diag = 1.0
        self._diag: List[str] = []

    # ------------------------------------------------------------------
    # Diagnostics helper
    def consume_diagnostics(self) -> List[str]:
        """Return and clear diagnostic messages from the last update."""

        diag, self._diag = self._diag, []
        return diag

    # ------------------------------------------------------------------
    def _prepare_measurements(
        self,
        detections: List[Det],
        predictions: np.ndarray,
        covariances: List[np.ndarray],
        track_ids: Sequence[int],
    ) -> np.ndarray:
        num_dets = len(detections)
        num_trks = predictions.shape[0]

        cost_matrix = np.full((num_dets, num_trks), 1e3, dtype=np.float32)
        self._diag = []

        if num_dets == 0 or num_trks == 0:
            return cost_matrix

        det_measurements = np.stack([xyxy_to_cxysr(det[0]) for det in detections], axis=0)

        for trk_idx, (pred, cov, track_id) in enumerate(zip(predictions, covariances, track_ids)):
            innovation_cov = cov[:4, :4] + np.diag([1.0, 1.0, 10.0, 1.0])
            try:
                inv_cov = np.linalg.inv(innovation_cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(innovation_cov)

            deltas = det_measurements - pred
            maha2 = np.einsum("ni,ij,nj->n", deltas, inv_cov, deltas)
            center_dist = np.sqrt(deltas[:, 0] ** 2 + deltas[:, 1] ** 2) / max(1e-6, self._img_diag)
            scale_pen = np.abs(np.log((det_measurements[:, 2] + 1e-6) / (pred[2] + 1e-6)))
            aspect_pen = np.abs(np.log((det_measurements[:, 3] + 1e-6) / (pred[3] + 1e-6)))

            base_cost = self.w_center * center_dist + self.w_scale * scale_pen + self.w_aspect * aspect_pen

            row_costs = base_cost.astype(np.float32)
            for det_idx in range(num_dets):
                reasons: List[str] = []
                if center_dist[det_idx] > self.center_gate_frac:
                    reasons.append("center")
                if maha2[det_idx] > self.maha_gate2:
                    reasons.append("maha")

                if reasons:
                    message = (
                        f"trk{track_id}↔det{det_idx}: center={center_dist[det_idx]:.3f} "
                        f"scale={scale_pen[det_idx]:.3f} aspect={aspect_pen[det_idx]:.3f} "
                        f"maha={maha2[det_idx]:.2f} REJECT({'/'.join(reasons)})"
                    )
                    self._diag.append(message)
                    row_costs[det_idx] = 1e3
                else:
                    # Record an accepted candidate with its composite cost for transparency.
                    message = (
                        f"trk{track_id}↔det{det_idx}: center={center_dist[det_idx]:.3f} "
                        f"scale={scale_pen[det_idx]:.3f} aspect={aspect_pen[det_idx]:.3f} "
                        f"maha={maha2[det_idx]:.2f} cost={base_cost[det_idx]:.3f}"
                    )
                    self._diag.append(message)

            cost_matrix[:, trk_idx] = row_costs

        return cost_matrix

    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Det],
        img_size: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, BBox, str, float]]:
        if img_size is None:
            raise ValueError("DBTracker.update requires img_size=(width, height)")

        width, height = img_size
        self._img_diag = sqrt(width * width + height * height)

        predictions: List[np.ndarray] = []
        covariances: List[np.ndarray] = []
        track_ids: List[int] = []
        for track in self._tracks:
            pred, cov = track.predict()
            predictions.append(pred)
            covariances.append(cov)
            track_ids.append(track.id)

        if predictions:
            pred_arr = np.stack(predictions, axis=0)
        else:
            pred_arr = np.zeros((0, 4), dtype=float)

        cost_matrix = self._prepare_measurements(detections, pred_arr, covariances, track_ids)

        assigned_det_indices: set[int] = set()
        assigned_trk_indices: set[int] = set()
        matches: List[Tuple[int, int]] = []

        if detections and self._tracks:
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            for det_idx, trk_idx in zip(row_idx, col_idx):
                if cost_matrix[det_idx, trk_idx] >= 1e2:
                    continue
                matches.append((det_idx, trk_idx))
                assigned_det_indices.add(det_idx)
                assigned_trk_indices.add(trk_idx)

        for det_idx, trk_idx in matches:
            bbox, score, label = detections[det_idx]
            track = self._tracks[trk_idx]
            track.update(bbox, label, score)

        alive: List[Track] = []
        for trk_idx, track in enumerate(self._tracks):
            if trk_idx not in assigned_trk_indices:
                track.missed += 1
                track.score *= 0.95
            if track.missed <= self.max_age:
                alive.append(track)
        self._tracks = alive

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_det_indices:
                continue
            bbox, score, label = det
            self._tracks.append(Track.from_detection(bbox, label, score, self._next_id))
            self._next_id += 1

        outputs: List[Tuple[int, BBox, str, float]] = []
        for track in self._tracks:
            if track.hits >= self.min_hits or track.missed == 0:
                outputs.append((track.id, track.bbox, track.label, float(track.score)))
        return outputs

