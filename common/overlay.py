"""Utilities for drawing tracking overlays on video frames."""

from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import cv2

BBox = Tuple[int, int, int, int]
TrackTuple = Tuple[int, BBox, str, float]


def _put_boxed_text(
    img,
    text: str,
    org: Tuple[int, int],
    *,
    font_scale: float = 0.5,
    thickness: int = 1,
    fg: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.6,
):
    """Draw ``text`` at ``org`` with a filled background box."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 3
    x1, y1 = x, y - th - 2 * pad
    x2, y2 = x + tw + 2 * pad, y + th + 2 * pad
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1] - 1, x2)
    y2 = min(img.shape[0] - 1, y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x + pad, y - base), font, font_scale, fg, thickness, cv2.LINE_AA)


def draw_tracks(frame, tracks: Iterable[TrackTuple], draw_scores: bool = True):
    """Annotate ``frame`` with bounding boxes and labels for ``tracks``."""

    for tid, (x1, y1, x2, y2), clazz, conf in tracks:
        color = (37 * tid % 255, 17 * tid % 255, 89 * tid % 255)
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(frame, pt1, pt2, color, 2)

        tag = f"{clazz} {tid}"
        if draw_scores:
            tag += f"  {conf:.2f}"

        _put_boxed_text(
            frame,
            tag,
            org=(pt1[0], max(0, pt1[1] - 4)),
            font_scale=0.5,
            thickness=1,
            fg=(255, 255, 255),
            bg=color,
            alpha=0.6,
        )
    return frame


def draw_hud(
    frame,
    stats: Mapping[str, object],
    *,
    corner: str = "tl",
    scale: float = 0.6,
    opacity: float = 0.6,
):
    """Render a diagnostics heads-up display on ``frame``."""

    lines = [
        "cam: {}  {}x{}".format(
            stats.get("cam_id", "?"),
            stats.get("img_wh", (0, 0))[0],
            stats.get("img_wh", (0, 0))[1],
        ),
        f"t={stats.get('ts_str', '')}",
        f"frame: {stats.get('frame_idx', 0)}   fps~{stats.get('fps', 0.0):.2f}",
        f"dets: {stats.get('n_dets', 0)}   tracks: {stats.get('n_tracks', 0)}",
    ]

    tracker_stats = stats.get("tracker", {}) or {}
    if tracker_stats:
        name = tracker_stats.get("name", "")
        max_age = tracker_stats.get("max_age", "?")
        min_hits = tracker_stats.get("min_hits", "?")
        iou_thr = tracker_stats.get("iou_threshold")
        if iou_thr is not None:
            lines.append(
                f"tracker: {name}  iou_thr={float(iou_thr):.2f}  age={max_age}  hits={min_hits}"
            )
        else:
            cg = tracker_stats.get("center_gate_frac", "?")
            maha = tracker_stats.get("maha_gate_p", "?")
            if isinstance(cg, (int, float)):
                cg = f"{cg:.2f}"
            if isinstance(maha, (int, float)):
                maha = f"{maha:.3f}"
            lines.append(
                f"tracker: {name}  cg={cg}  maha_p={maha}  age={max_age}  hits={min_hits}"
            )

    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = frame.shape[:2]
    line_h = int(18 * scale)
    pad = int(8 * scale)
    box_w = int(max(cv2.getTextSize(text, font, scale, 1)[0][0] for text in lines) + 2 * pad)
    box_h = int(line_h * len(lines) + 2 * pad)

    if corner == "tl":
        x1, y1 = 5, 5 + box_h
    elif corner == "tr":
        x1, y1 = width - box_w - 5, 5 + box_h
    elif corner == "bl":
        x1, y1 = 5, height - 5
    else:  # "br"
        x1, y1 = width - box_w - 5, height - 5

    x2, y2 = x1 + box_w, y1 - box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    y = y1 - pad
    for text in lines:
        cv2.putText(frame, text, (x1 + pad, y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y -= line_h

    return frame

