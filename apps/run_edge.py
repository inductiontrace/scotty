from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from paho.mqtt import client as mqtt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.events import DetectionEvent
from common.loader import load_object
from common.quality import sharpness_laplacian
from common.video_utils import open_source, to_bgr

BBox = Tuple[int, int, int, int]


@dataclass
class _InstantTrack:
    """Lightweight track used when no tracker is configured."""

    id: int
    box_xyxy: BBox
    conf: float
    clazz: str


class _PassthroughTracker:
    """Fallback tracker that yields one track per detection."""

    def __init__(self) -> None:
        self._next_id = 1

    def step(self, dets: List[Tuple[BBox, float, str]]) -> List[_InstantTrack]:
        tracks: List[_InstantTrack] = []
        for bbox, conf, clazz in dets:
            tracks.append(_InstantTrack(self._next_id, bbox, conf, clazz))
            self._next_id += 1
        return tracks


class _TrackerAdapter:
    """Wraps a tracker that outputs tuples and exposes the step() API."""

    def __init__(self, tracker) -> None:
        self._tracker = tracker
        self._states: Dict[int, _InstantTrack] = {}

    def step(self, dets: List[Tuple[BBox, float, str]]) -> List[_InstantTrack]:
        tracks: List[_InstantTrack] = []
        seen: set[int] = set()
        for tid, bbox, label, score in self._tracker.update(dets):
            box_xyxy = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            track = self._states.get(tid)
            if track is None:
                track = _InstantTrack(tid, box_xyxy, float(score), label)
                self._states[tid] = track
            else:
                track.box_xyxy = box_xyxy
            track.conf = float(score)
            track.clazz = label
            tracks.append(track)
            seen.add(tid)

        stale = [tid for tid in self._states if tid not in seen]
        for tid in stale:
            del self._states[tid]
        return tracks

LOGGER = logging.getLogger("haecceity.edge")
logger = LOGGER


def _instantiate_from_config(entry):
    cls = load_object(entry["impl"])
    return cls(entry)


def _build(cfg):
    cam_id = cfg["edge"]["cam_id"]
    # Intuitus (optional)
    intu = None
    if cfg.get("intuitus", {}).get("enabled", False):
        impl = load_object(cfg["intuitus"]["impl"])
        intu = impl(cfg["intuitus"])

    # Detector
    det_impl = load_object(cfg["quiddity"]["impl"])
    detector = det_impl(cfg["quiddity"])

    # Tracker (may be null)
    tracker_cfg = cfg.get("tracker", {})
    tracker = None
    if tracker_cfg and tracker_cfg.get("impl"):
        tr_impl = load_object(tracker_cfg["impl"])
        tracker_kwargs = {k: v for k, v in tracker_cfg.items() if k != "impl"}
        tracker = _TrackerAdapter(tr_impl(**tracker_kwargs))
    else:
        logger.info("No tracker configured; using passthrough detections.")

    # Haecceity
    from haecceity.registry import GlobalRegistry

    hcfg = cfg.get("haecceity", {})
    embed_classes = set(hcfg.get("embed_classes", []))
    specialists = []
    for sc in hcfg.get("specialists", []):
        impl_path = sc.get("impl")
        try:
            # 1) skip if class not in allowlist (if class list is declared on class)
            cls_type = load_object(impl_path)
            supported = set(getattr(cls_type, "classes", []))
            if supported and not (supported & embed_classes):
                logger.debug(
                    "Skipping %s (unsupported for embed_classes=%s)",
                    impl_path,
                    embed_classes,
                )
                continue
            # 2) skip if it declares a model_path that doesn't exist
            mp = sc.get("model_path")
            if mp and not os.path.exists(mp):
                logger.warning(
                    "Skipping specialist %s: missing model_path '%s'",
                    impl_path,
                    mp,
                )
                continue
            specialists.append(_instantiate_from_config(sc))
            logger.info("Loading specialist: %s", impl_path)
        except Exception as e:
            logger.warning(
                "Failed to instantiate specialist %s: %s. Continuing without it.",
                impl_path,
                e,
            )

    fallbacks = []
    for fc in hcfg.get("fallbacks", []):
        try:
            mp = fc.get("model_path")
            if mp and not os.path.exists(mp):
                logger.warning(
                    "Skipping fallback %s: missing model_path '%s'",
                    fc["impl"],
                    mp,
                )
                continue
            fallbacks.append(_instantiate_from_config(fc))
            logger.info("Loading fallback: %s", fc["impl"])
        except Exception as e:
            logger.warning(
                "Failed to instantiate fallback %s: %s. Continuing without it.",
                fc.get("impl"),
                e,
            )

    reg = GlobalRegistry(hcfg.get("new_id_threshold", 0.55), hcfg.get("hysteresis", 0.05))
    return cam_id, intu, detector, tracker, specialists, fallbacks, reg, hcfg


def _pick_specialist(
    specialists: Iterable, fallbacks: Iterable, clazz: str
):  # pragma: no cover - trivial selector
    for spec in specialists:
        classes = getattr(spec, "classes", None) or []
        if clazz in classes:
            return spec
    for fallback in fallbacks:
        return fallback
    return None


def _connect_mqtt(url: Optional[str]):
    if not url:
        return None, None
    import urllib.parse as up

    parsed = up.urlparse(url)
    topic = parsed.path.lstrip("/") or "tracks"
    client = mqtt.Client()
    if parsed.scheme.startswith("mqtts"):
        client.tls_set()
    if parsed.username:
        client.username_pw_set(parsed.username, parsed.password or "")
    client.connect(
        parsed.hostname,
        parsed.port or (8883 if parsed.scheme.startswith("mqtts") else 1883),
        60,
    )
    client.loop_start()
    LOGGER.info("Connected to MQTT broker at %s", parsed.hostname)
    return client, topic


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args(list(argv) if argv is not None else None)

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    cam_id, intu, detector, tracker, specialists, fallbacks, registry, hcfg = _build(cfg)
    if tracker is None:
        tracker = _PassthroughTracker()

    emit_path = cfg["edge"]["emit_jsonl"]
    pathlib.Path(os.path.dirname(emit_path)).mkdir(parents=True, exist_ok=True)
    output = open(emit_path, "a", buffering=1, encoding="utf-8")
    LOGGER.info("Writing events to %s", emit_path)

    src = cfg["edge"].get("source", 0)
    try:
        read_fn, close_fn, size_fn = open_source(src, cfg["edge"].get("fps", 1.0))
    except SystemExit as exc:
        LOGGER.error(str(exc))
        return 1

    fps = float(cfg["edge"].get("fps", 1.0))
    period = 1.0 / max(0.1, fps)
    last = 0.0
    frame_idx = 0

    client, topic = _connect_mqtt(cfg["edge"].get("mqtt"))

    embed_interval = int(hcfg.get("embed_interval_frames", 1))
    embed_classes = set(hcfg.get("embed_classes", []))

    try:
        while True:
            now = time.time()
            if now - last < period:
                time.sleep(0.002)
                continue
            last = now

            ok, frame = read_fn()
            if not ok:
                LOGGER.warning("Failed to read frame from camera; stopping.")
                break

            frame_bgr = to_bgr(frame)
            if frame_bgr is None:
                LOGGER.warning("Received empty frame; skipping.")
                continue

            height, width = frame_bgr.shape[:2]
            rois = intu.next_rois(frame_bgr) if intu is not None else None

            detections = []
            if rois is None:
                detections = detector.detect(frame_bgr)
            else:
                for (x1, y1, x2, y2) in rois:
                    crop = frame_bgr[y1:y2, x1:x2]
                    local_dets = detector.detect(crop)
                    for (bx1, by1, bx2, by2), conf, cls in local_dets:
                        detections.append(((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1), conf, cls))

            tracks = tracker.step(detections)
            events = []
            for track in tracks:
                x1, y1, x2, y2 = track.box_xyxy
                crop = frame_bgr[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
                sharp = sharpness_laplacian(gray) if gray is not None else 0.0
                hfrac = (y2 - y1) / float(height + 1e-9)
                quality = {"sharpness": sharp, "bbox_h_frac": hfrac}

                specialist = None
                have_embed = False
                if track.clazz in embed_classes:
                    specialist = _pick_specialist(specialists, fallbacks, track.clazz)
                    if specialist is not None and crop.size:
                        last_embed_frame = getattr(track, "last_embed_frame", -9999)
                        if frame_idx - last_embed_frame >= embed_interval:
                            try:
                                embedding = specialist.embed(crop)
                                embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
                                prev = getattr(track, "embed", None)
                                if prev is not None:
                                    embedding = 0.75 * prev + 0.25 * embedding
                                    embedding /= np.linalg.norm(embedding) + 1e-9
                                track.embed = embedding
                                track.last_embed_frame = frame_idx
                                track.global_id = registry.assign(track, embedding)
                                have_embed = True
                            except Exception as exc:  # pragma: no cover - logging path
                                LOGGER.warning(
                                    "Specialist %s failed to embed: %s", specialist, exc
                                )

                event = DetectionEvent(
                    ts_ms=int(time.time() * 1000),
                    cam_id=cam_id,
                    frame=frame_idx,
                    track_id_local=track.id,
                    global_id=getattr(track, "global_id", None),
                    clazz=track.clazz,
                    conf=float(track.conf),
                    box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    img_wh=(int(width), int(height)),
                    have_embed=have_embed,
                    specialist=getattr(specialist, "name", None) if specialist else None,
                    quality=quality,
                )
                events.append(event)

            for event in events:
                line = event.to_json()
                LOGGER.info(
                    "Frame %d: track %d (%s) conf=%.2f bbox=%s embed=%s",
                    event.frame,
                    event.track_id_local,
                    event.clazz,
                    event.conf,
                    event.box_xyxy,
                    "yes" if event.have_embed else "no",
                )
                output.write(line + "\n")
                if client is not None and topic is not None:
                    client.publish(topic, line, qos=0, retain=False)

            frame_idx += 1

    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        LOGGER.info("Stopping edge loop (keyboard interrupt).")
    finally:
        output.close()
        if client is not None:
            client.loop_stop()
            client.disconnect()
        close_fn()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    sys.exit(main())
