from __future__ import annotations

import argparse
import atexit
import datetime
import logging
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import shlex
import subprocess

import cv2
import numpy as np
import yaml
from paho.mqtt import client as mqtt

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.dedup import simple_nms
from common.events import DetectionEvent
from common.loader import load_object
from common.overlay import draw_detections, draw_hud
from common.quality import sharpness_laplacian
from common.video_utils import open_source, to_bgr
from common.webviewer import WebViewer

BBox = Tuple[int, int, int, int]


@dataclass
class _InstantDetection:
    """Lightweight detection used for event enrichment and overlays."""

    id: int
    box_xyxy: BBox
    conf: float
    clazz: str

LOGGER = logging.getLogger("haecceity.edge")
logger = LOGGER


def _instantiate_from_config(entry):
    cls = load_object(entry["impl"])
    return cls(entry)


def _default_zmq_bind(endpoint: str) -> str:
    """Return a tcp:// bind endpoint for a tcp:// connect endpoint."""

    if endpoint.startswith("tcp://"):
        host_port = endpoint[len("tcp://") :]
        if ":" in host_port:
            _, port = host_port.rsplit(":", 1)
            return f"tcp://*:{port}"
    return endpoint


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

    # Haecceity
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

    return cam_id, intu, detector, specialists, fallbacks, embed_classes


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
    topic = parsed.path.lstrip("/") or "detections"
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

    cam_id, intu, detector, specialists, fallbacks, embed_classes = _build(cfg)

    edge_cfg = cfg["edge"]

    emit_path = cfg["edge"]["emit_jsonl"]
    pathlib.Path(os.path.dirname(emit_path)).mkdir(parents=True, exist_ok=True)
    output = open(emit_path, "a", buffering=1, encoding="utf-8")
    LOGGER.info("Writing events to %s", emit_path)

    output_cfg = cfg.get("output", {})
    video_path = output_cfg.get("video_path")
    if video_path:
        pathlib.Path(os.path.dirname(video_path)).mkdir(parents=True, exist_ok=True)
        video_writer_settings = {
            "path": video_path,
            "fps": float(output_cfg.get("fps", cfg["edge"].get("fps", 1.0))),
            "codec": output_cfg.get("codec", "mp4v"),
        }
        LOGGER.info("Writing annotated video to %s", video_path)
    else:
        video_writer_settings = None
        LOGGER.info("No video output configured")
    draw_scores = bool(output_cfg.get("draw_scores", True))
    draw_diag = bool(output_cfg.get("draw_hud", True))
    hud_corner = output_cfg.get("hud_corner", "tl")
    hud_scale = float(output_cfg.get("hud_scale", 0.6))
    hud_opacity = float(output_cfg.get("hud_opacity", 0.6))
    video_out = None

    web_cfg = edge_cfg.get("web", {}) or {}
    web_viewer: Optional[WebViewer] = None
    if web_cfg.get("enabled", False):
        host = str(web_cfg.get("host", "0.0.0.0"))
        port = int(web_cfg.get("port", 8080))
        title = str(web_cfg.get("title") or f"{cam_id} live")
        jpeg_quality = int(web_cfg.get("jpeg_quality", 80))
        try:
            web_viewer = WebViewer(
                host=host, port=port, title=title, jpeg_quality=jpeg_quality
            )
        except OSError as exc:
            LOGGER.error("Failed to start web viewer: %s", exc)
            web_viewer = None
        else:
            LOGGER.info("Web viewer enabled at http://%s:%d/", host, port)
    else:
        LOGGER.info("Web viewer disabled")

    source_cfg = edge_cfg.get("source", {})
    use_zmq = isinstance(source_cfg, dict) and source_cfg.get("kind") == "zmq"
    read_fn = None
    frame_iter = None
    close_fn = lambda: None

    hub_proc: Optional[subprocess.Popen] = None
    zmq_frames_seen = 0

    if use_zmq:
        endpoint = source_cfg.get("endpoint", "tcp://127.0.0.1:5555")
        bind_endpoint = source_cfg.get("bind", _default_zmq_bind(endpoint))
        publisher_cfg = source_cfg.get("publisher", {}) or {}
        hub_source = str(publisher_cfg.get("source", "picam"))
        hub_fps = float(publisher_cfg.get("fps", edge_cfg.get("fps", 5.0)))
        hub_cam_id = str(publisher_cfg.get("camera_id", edge_cfg.get("cam_id", "cam")))
        hub_loop = bool(publisher_cfg.get("loop", False))

        hub_cmd = [
            sys.executable,
            "-m",
            "framebus.hub",
            "--endpoint",
            str(bind_endpoint),
            "--source",
            hub_source,
            "--fps",
            str(hub_fps),
            "--camera-id",
            hub_cam_id,
        ]
        if hub_loop:
            hub_cmd.append("--loop")

        LOGGER.info("Starting FrameBus hub: %s", " ".join(shlex.quote(arg) for arg in hub_cmd))
        try:
            hub_proc = subprocess.Popen(hub_cmd)
        except OSError as exc:
            LOGGER.error("Failed to launch FrameBus hub: %s", exc)
            return 1

        def _stop_hub() -> None:
            if hub_proc is not None and hub_proc.poll() is None:
                LOGGER.debug("Stopping FrameBus hub (atexit).")
                hub_proc.terminate()
                try:
                    hub_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    hub_proc.kill()

        atexit.register(_stop_hub)

        LOGGER.info("Connecting to FrameBus at %s", endpoint)
        from framebus.reader import ZmqFrameReader

        reader = ZmqFrameReader(endpoint=endpoint)
        frame_iter = iter(reader)
        close_fn = getattr(reader, "close", lambda: None)
    else:
        if isinstance(source_cfg, dict):
            src = source_cfg.get("device", source_cfg.get("path", 0))
        else:
            src = source_cfg if source_cfg not in (None, {}) else 0
        try:
            read_fn, close_fn, _ = open_source(src, edge_cfg.get("fps", 1.0))
        except SystemExit as exc:
            LOGGER.error(str(exc))
            return 1

    fps = float(edge_cfg.get("fps", 1.0))
    period = 0.0 if use_zmq else 1.0 / max(0.1, fps)
    last = 0.0
    frame_idx = -1
    fps_ema: Optional[float] = None
    ema_alpha = 0.2
    t_prev: Optional[float] = None
    frame_ts: Optional[float] = None

    client, topic = _connect_mqtt(cfg["edge"].get("mqtt"))

    embed_classes = set(embed_classes)
    next_detection_id = 1

    try:
        while True:
            frame_ts = None
            if use_zmq and hub_proc is not None and hub_proc.poll() is not None:
                LOGGER.error(
                    "FrameBus hub exited unexpectedly with code %s", hub_proc.returncode
                )
                break
            if use_zmq:
                if frame_iter is None:
                    LOGGER.error("Frame iterator unavailable for ZMQ source.")
                    break
                try:
                    frame_bgr, meta = next(frame_iter)
                except StopIteration:
                    LOGGER.info("Frame stream ended; stopping.")
                    break
                if frame_bgr is None:
                    LOGGER.warning("Received empty frame from bus; skipping.")
                    continue
                frame_ts = meta.get("ts")
                frame_id_meta = meta.get("frame_id")
                if frame_id_meta is not None:
                    frame_idx = int(frame_id_meta)
                else:
                    frame_idx += 1
                if frame_ts is None:
                    frame_ts = time.time()
                zmq_frames_seen += 1
                if zmq_frames_seen == 1:
                    LOGGER.info(
                        "FrameBus stream connected at %s (frame_idx=%d).",
                        endpoint,
                        frame_idx,
                    )
                elif zmq_frames_seen % 10 == 0:
                    LOGGER.info(
                        "FrameBus processed %d frames (last frame_idx=%d).",
                        zmq_frames_seen,
                        frame_idx,
                    )
            else:
                now = time.time()
                if now - last < period:
                    time.sleep(0.002)
                    continue
                last = now

                if read_fn is None:
                    LOGGER.error("No frame reader configured for camera source.")
                    break
                ok, frame = read_fn()
                if not ok:
                    LOGGER.warning("Failed to read frame from camera; stopping.")
                    break

                frame_bgr = to_bgr(frame)
                if frame_bgr is None:
                    LOGGER.warning("Received empty frame; skipping.")
                    continue
                frame_ts = now
                frame_idx += 1
                meta = {}

            height, width = frame_bgr.shape[:2]
            t_now = time.time()
            if t_prev is not None:
                dt = max(1e-6, t_now - t_prev)
                inst_fps = 1.0 / dt
                fps_ema = (
                    inst_fps
                    if fps_ema is None
                    else (1 - ema_alpha) * fps_ema + ema_alpha * inst_fps
                )
            t_prev = t_now
            if video_out is None and video_writer_settings is not None:
                fourcc = cv2.VideoWriter_fourcc(*video_writer_settings["codec"])
                video_out = cv2.VideoWriter(
                    video_writer_settings["path"],
                    fourcc,
                    video_writer_settings["fps"],
                    (int(width), int(height)),
                )
                if not video_out.isOpened():
                    LOGGER.warning(
                        "Failed to open video writer at %s",
                        video_writer_settings["path"],
                    )
                    video_out.release()
                    video_out = None
                    video_writer_settings = None
            rois = intu.next_rois(frame_bgr) if intu is not None else None

            detections: List[Tuple[BBox, float, str]] = []
            if rois is None:
                detections = detector.detect(frame_bgr)
            else:
                for (x1, y1, x2, y2) in rois:
                    crop = frame_bgr[y1:y2, x1:x2]
                    local_dets = detector.detect(crop)
                    for (bx1, by1, bx2, by2), conf, cls in local_dets:
                        detections.append(((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1), conf, cls))

            det_cfg = cfg.get("detector", {}) or {}
            if not det_cfg:
                det_cfg = cfg.get("quiddity", {}) or {}
            dd_cfg = det_cfg.get("dedup", {}) or {}
            if dd_cfg.get("enabled", True):
                detections = simple_nms(
                    detections, iou_thr=float(dd_cfg.get("iou_thr", 0.6))
                )

            observations: List[_InstantDetection] = []
            for bbox, conf, clazz in detections:
                observations.append(
                    _InstantDetection(next_detection_id, bbox, float(conf), clazz)
                )
                next_detection_id += 1

            events = []
            for det in observations:
                x1, y1, x2, y2 = det.box_xyxy
                crop = frame_bgr[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
                sharp = sharpness_laplacian(gray) if gray is not None else 0.0
                hfrac = (y2 - y1) / float(height + 1e-9)
                quality = {"sharpness": sharp, "bbox_h_frac": hfrac}

                specialist = None
                have_embed = False
                embedding_vec: Optional[List[float]] = None
                if det.clazz in embed_classes:
                    specialist = _pick_specialist(specialists, fallbacks, det.clazz)
                    if specialist is not None and crop.size:
                        try:
                            if specialist.wants(det.conf, det.box_xyxy, quality):
                                embedding = specialist.embed(crop)
                                embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
                                if hasattr(embedding, "astype"):
                                    embedding_vec = embedding.astype(float).tolist()
                                else:
                                    embedding_vec = list(embedding)
                                have_embed = True
                        except Exception as exc:  # pragma: no cover - logging path
                            LOGGER.warning(
                                "Specialist %s failed to embed: %s", specialist, exc
                            )

                event = DetectionEvent(
                    ts_ms=int((frame_ts or time.time()) * 1000),
                    cam_id=cam_id,
                    frame=frame_idx,
                    detection_id=det.id,
                    clazz=det.clazz,
                    conf=float(det.conf),
                    box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    img_wh=(int(width), int(height)),
                    have_embed=have_embed,
                    specialist=getattr(specialist, "name", None) if specialist else None,
                    quality=quality,
                    embedding=embedding_vec,
                )
                events.append(event)

            for event in events:
                line = event.to_json()
                LOGGER.info(
                    "Frame %d @ %dms: detection %d (%s) conf=%.2f bbox=%s embed=%s",
                    event.frame,
                    event.ts_ms,
                    event.detection_id,
                    event.clazz,
                    event.conf,
                    event.box_xyxy,
                    "yes" if event.have_embed else "no",
                )
                output.write(line + "\n")
                if client is not None and topic is not None:
                    client.publish(topic, line, qos=0, retain=False)

            need_vis = (video_out is not None) or (web_viewer is not None)
            if need_vis:
                vis = frame_bgr.copy()
                detection_tuples = [
                    (det.id, det.box_xyxy, det.clazz, float(det.conf))
                    for det in observations
                ]
                draw_detections(vis, detection_tuples, draw_scores=draw_scores)
                if draw_diag:
                    stats = {
                        "cam_id": cam_id,
                        "img_wh": (int(width), int(height)),
                        "ts_str": datetime.datetime.fromtimestamp(
                            frame_ts or time.time()
                        ).strftime("%H:%M:%S"),
                        "frame_idx": frame_idx,
                        "fps": fps_ema or 0.0,
                        "n_dets": len(detections),
                        "n_detections": len(detection_tuples),
                    }
                    draw_hud(
                        vis,
                        stats,
                        corner=hud_corner,
                        scale=hud_scale,
                        opacity=hud_opacity,
                    )
                if video_out is not None:
                    video_out.write(vis)
                if web_viewer is not None:
                    web_viewer.publish(vis)

    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        LOGGER.info("Stopping edge loop (keyboard interrupt).")
    finally:
        output.close()
        if client is not None:
            client.loop_stop()
            client.disconnect()
        if close_fn is not None:
            try:
                close_fn()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        if hub_proc is not None and hub_proc.poll() is None:
            LOGGER.debug("Stopping FrameBus hub (finalize).")
            hub_proc.terminate()
            try:
                hub_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                hub_proc.kill()
        if video_out is not None:
            video_out.release()
        if web_viewer is not None:
            web_viewer.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    sys.exit(main())
