from __future__ import annotations

from typing import Dict

import numpy as np


class GlobalRegistry:
    def __init__(self, new_id_threshold: float, hysteresis: float):
        self.centroids: Dict[int, np.ndarray] = {}
        self.next_gid = 1
        self.new_threshold = float(new_id_threshold)
        self.hysteresis = float(hysteresis)

    def assign(self, track, embedding: np.ndarray) -> int:
        emb = embedding / (np.linalg.norm(embedding) + 1e-9)
        if not self.centroids:
            gid = self._new(emb)
            track.global_id = gid
            track._last_sim = 1.0
            return gid

        ids = list(self.centroids.keys())
        cents = np.stack([self.centroids[i] for i in ids], axis=0)
        sims = cents @ emb
        best_idx = int(np.argmax(sims))
        best_gid = ids[best_idx]
        best_score = float(sims[best_idx])

        if getattr(track, "global_id", None) == best_gid:
            last = getattr(track, "_last_sim", 0.0)
            if best_score + self.hysteresis >= last:
                self._update(best_gid, emb)
                track._last_sim = best_score
                return best_gid

        if best_score >= self.new_threshold:
            self._update(best_gid, emb)
            track.global_id = best_gid
            track._last_sim = best_score
            return best_gid

        gid = self._new(emb)
        track.global_id = gid
        track._last_sim = 1.0
        return gid

    def _new(self, emb: np.ndarray) -> int:
        gid = self.next_gid
        self.next_gid += 1
        self.centroids[gid] = emb.copy()
        return gid

    def _update(self, gid: int, emb: np.ndarray) -> None:
        updated = 0.9 * self.centroids[gid] + 0.1 * emb
        self.centroids[gid] = updated / (np.linalg.norm(updated) + 1e-9)
