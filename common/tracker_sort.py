from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter

from common.interfaces import BBox, Tracker


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    au = (ax2 - ax1) * (ay2 - ay1)
    bu = (bx2 - bx1) * (by2 - by1)
    return inter / (au + bu - inter + 1e-9)


@dataclass
class _Track:
    id: int
    box_xyxy: BBox
    conf: float
    clazz: str
    time_since_update: int = 0
    hits: int = 1
    global_id: int | None = None
    _last_sim: float = 0.0
    _kf: KalmanFilter | None = None
    embed: np.ndarray | None = None
    last_embed_frame: int = -9999


class Sort(Tracker):
    _next = 1

    def __init__(self, iou_th: float = 0.3, max_age: int = 12):
        self.tracks: List[_Track] = []
        self.iou_th = iou_th
        self.max_age = max_age

    def _kf_init(self, box: BBox) -> KalmanFilter:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        s = max(1.0, (x2 - x1) * (y2 - y1))
        r = (x2 - x1) / (y2 - y1 + 1e-6)
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array(
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
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )
        kf.R *= 1.0
        kf.P *= 10.0
        kf.Q *= 0.01
        kf.x[:4] = np.array([[cx], [cy], [s], [r]], dtype=float)
        return kf

    def step(self, dets: List[Tuple[BBox, float, str]]) -> List[_Track]:
        for track in self.tracks:
            if track._kf is not None:
                track._kf.predict()
            track.time_since_update += 1

        unmatched = list(range(len(dets)))
        matches = []
        for ti, track in enumerate(self.tracks):
            best_j = -1
            best_iou = 0.0
            for j in unmatched:
                iou = _iou(track.box_xyxy, dets[j][0])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= self.iou_th and best_j >= 0:
                matches.append((ti, best_j))
                unmatched.remove(best_j)

        for ti, j in matches:
            box, conf, clazz = dets[j]
            track = self.tracks[ti]
            track.box_xyxy = box
            track.conf = conf
            track.clazz = clazz
            track.time_since_update = 0
            track.hits += 1
            if track._kf is not None:
                x1, y1, x2, y2 = box
                z = np.array(
                    [
                        [(x1 + x2) / 2.0],
                        [(y1 + y2) / 2.0],
                        [(x2 - x1) * (y2 - y1)],
                        [(x2 - x1) / (y2 - y1 + 1e-6)],
                    ]
                )
                track._kf.update(z)

        for j in unmatched:
            box, conf, clazz = dets[j]
            kf = self._kf_init(box)
            self.tracks.append(
                _Track(id=Sort._next, box_xyxy=box, conf=conf, clazz=clazz, _kf=kf)
            )
            Sort._next += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.tracks
