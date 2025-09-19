from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import json


@dataclass
class DetectionEvent:
    ts_ms: int
    cam_id: str
    frame: int
    track_id_local: int
    global_id: Optional[int]
    clazz: str
    conf: float
    box_xyxy: Tuple[int, int, int, int]
    img_wh: Tuple[int, int]
    have_embed: bool
    specialist: Optional[str] = None
    quality: Optional[Dict] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))
