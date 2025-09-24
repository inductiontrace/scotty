"""Distance-based tracker tuned for low frame rates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import List, Optional, Sequence, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from common.dedup import iou_xyxy


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

    try:
        from scipy.stats import chi2

        value = float(chi2.ppf(prob, 4))
        if isfinite(value) and value > 0.0:
            return value
    except Exception:
        pass

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

    def __init__(
        self,
        z0: np.ndarray,
        q_pos: float,
        q_sz: float,
        r_pos: float,
        r_sz: float,
        img_diag: float,
    ) -> None:
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
        self._r_pos = float(r_pos)
        self._r_sz = float(r_sz)
        self.kf.R = np.eye(4)
        Q = np.zeros((7, 7), dtype=float)
        pos_var = q_pos ** 2
        sz_var = q_sz ** 2
        Q[0, 0] = pos_var
        Q[1, 1] = pos_var
        Q[2, 2] = sz_var
        Q[3, 3] = sz_var
        Q[4, 4] = pos_var
        Q[5, 5] = pos_var
        Q[6, 6] = sz_var
        self.kf.Q = Q

        self.kf.x[:4, 0] = z0
        self.update_measurement_noise(img_diag)

    def _current_dimensions(self) -> Tuple[float, float]:
        s = max(1e-6, float(self.kf.x[2, 0]))
        r = max(1e-6, float(self.kf.x[3, 0]))
        w = sqrt(max(1e-9, s * r))
        h = s / max(w, 1e-6)
        return w, h

    def update_measurement_noise(self, img_diag: float) -> np.ndarray:
        diag = max(1e-6, float(img_diag))
        w, h = self._current_dimensions()
        extent = max(diag, sqrt(w * w + h * h))
        pos_sigma = max(1e-6, self._r_pos * extent)
        s = max(1e-6, float(self.kf.x[2, 0]))
        r = max(1e-6, float(self.kf.x[3, 0]))
        scale_sigma = max(1e-6, self._r_sz * s)
        aspect_sigma = max(1e-6, self._r_sz * r)
        noise = np.diag(
            [pos_sigma ** 2, pos_sigma ** 2, scale_sigma ** 2, aspect_sigma ** 2]
        )
        self.kf.R = noise
        return noise

    def predict(self, img_diag: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.kf.predict()
        noise = self.update_measurement_noise(img_diag)
        return self.kf.x[:4, 0].copy(), self.kf.P.copy(), noise

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
    dup_hits: int = 0

    @classmethod
    def from_detection(
        cls,
        bbox: BBox,
        label: str,
        score: float,
        next_id: int,
        img_diag: float,
        kf_params: dict,
    ) -> "Track":
        z0 = xyxy_to_cxysr(bbox)
        kf = KfBox(z0, img_diag=img_diag, **kf_params)
        return cls(bbox, label, score, next_id, 1, 0, kf)

    def predict(self, img_diag: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z, cov, noise = self.kf.predict(img_diag)
        self.bbox = cxysr_to_xyxy(z)
        return z, cov, noise

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
        q_pos: float = 1.0,
        q_sz: float = 2.0,
        r_pos: float = 0.12,
        r_sz: float = 0.4,
        merge_dups: Optional[dict] = None,
    ) -> None:
        del iou_threshold  # parity with SORT config but unused

        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.center_gate_frac = float(center_gate_frac)
        self.maha_gate_p = float(maha_gate_p)
        self.maha_gate2 = chi2_inv_4d(self.maha_gate_p)
        self.w_center = float(w_center)
        self.w_scale = float(w_scale)
        self.w_aspect = float(w_aspect)
        self.w_app = float(w_app)

        self._kf_params = {
            "q_pos": float(q_pos),
            "q_sz": float(q_sz),
            "r_pos": float(r_pos),
            "r_sz": float(r_sz),
        }

        md_cfg = merge_dups or {}
        self._merge_dups_enabled = bool(md_cfg.get("enabled", True))
        self._merge_dups_iou = float(md_cfg.get("iou_thr", 0.8))
        self._merge_dups_hold = int(md_cfg.get("hold", 2))

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
        measurement_noises: Sequence[np.ndarray],
    ) -> np.ndarray:
        num_dets = len(detections)
        num_trks = predictions.shape[0]

        cost_matrix = np.full((num_dets, num_trks), 1e3, dtype=np.float32)
        self._diag = []

        if num_dets == 0 or num_trks == 0:
            return cost_matrix

        det_measurements = np.stack([xyxy_to_cxysr(det[0]) for det in detections], axis=0)

        for trk_idx, (pred, cov, track_id, meas_noise) in enumerate(
            zip(predictions, covariances, track_ids, measurement_noises)
        ):
            innovation_cov = cov[:4, :4] + meas_noise
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
        measurement_noises: List[np.ndarray] = []
        for track in self._tracks:
            pred, cov, meas_noise = track.predict(self._img_diag)
            predictions.append(pred)
            covariances.append(cov)
            track_ids.append(track.id)
            measurement_noises.append(meas_noise)

        if predictions:
            pred_arr = np.stack(predictions, axis=0)
        else:
            pred_arr = np.zeros((0, 4), dtype=float)

        cost_matrix = self._prepare_measurements(
            detections, pred_arr, covariances, track_ids, measurement_noises
        )

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
            self._tracks.append(
                Track.from_detection(
                    bbox, label, score, self._next_id, self._img_diag, self._kf_params
                )
            )
            self._next_id += 1

        if self._merge_dups_enabled:
            self._merge_duplicates(self._merge_dups_iou, self._merge_dups_hold)

        outputs: List[Tuple[int, BBox, str, float]] = []
        for track in self._tracks:
            if track.hits >= self.min_hits or track.missed == 0:
                outputs.append((track.id, track.bbox, track.label, float(track.score)))
        return outputs

    def _merge_duplicates(self, iou_thr: float, hold: int) -> None:
        alive = [t for t in self._tracks if t.missed == 0]
        if len(alive) < 2:
            for track in alive:
                track.dup_hits = 0
            return

        overlap_flags = {track.id: False for track in alive}
        to_drop: set[int] = set()

        for i in range(len(alive)):
            ti = alive[i]
            ti.dup_hits = getattr(ti, "dup_hits", 0)
            for j in range(i + 1, len(alive)):
                tj = alive[j]
                tj.dup_hits = getattr(tj, "dup_hits", 0)
                overlap = iou_xyxy(ti.bbox, tj.bbox)
                if overlap >= iou_thr:
                    overlap_flags[ti.id] = True
                    overlap_flags[tj.id] = True
                    ti.dup_hits = min(ti.dup_hits + 1, hold)
                    tj.dup_hits = min(tj.dup_hits + 1, hold)
                    if ti.dup_hits >= hold and tj.dup_hits >= hold:
                        winner, loser = (ti, tj) if ti.id <= tj.id else (tj, ti)
                        to_drop.add(loser.id)
                else:
                    # keep counters for other potential overlaps this frame
                    continue

        for track in alive:
            if not overlap_flags.get(track.id, False):
                track.dup_hits = 0

        if to_drop:
            self._tracks = [t for t in self._tracks if t.id not in to_drop]

