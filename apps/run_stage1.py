#!/usr/bin/env python3
import os
import time
import argparse
import json
import pathlib
import yaml
import cv2
from common.events import DetectionEvent
from common.loader import load_object


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    cam_id = cfg["edge"]["cam_id"]
    src = cfg["edge"]["source"]
    fps = float(cfg["edge"]["fps"])
    out_path = cfg["edge"]["emit_jsonl"]
    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    out = open(out_path, "a", buffering=1)

    det_impl = load_object(cfg["quiddity"]["impl"])
    detector = det_impl(cfg["quiddity"])

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")

    period = 1.0 / max(0.1, fps)
    last = 0.0
    frame_idx = 0
    ema_fps = None

    try:
        while True:
            now = time.time()
            if now - last < period:
                time.sleep(0.002)
                continue
            last = now

            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]

            t0 = time.time()
            dets = detector.detect(frame)
            infer_ms = (time.time() - t0) * 1000.0

            for (x1, y1, x2, y2), conf, clazz in dets:
                ev = DetectionEvent(
                    ts_ms=int(time.time() * 1000),
                    cam_id=cam_id,
                    frame=frame_idx,
                    track_id_local=-1,
                    global_id=None,
                    clazz=clazz,
                    conf=float(conf),
                    box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    img_wh=(int(W), int(H)),
                    have_embed=False,
                    specialist=None,
                    quality=None,
                )
                out.write(ev.to_json() + "\n")

            inst_fps = 1000.0 / max(1.0, infer_ms)
            ema_fps = inst_fps if ema_fps is None else 0.9 * ema_fps + 0.1 * inst_fps
            print(
                f"frame {frame_idx} | dets {len(dets)} | infer {infer_ms:.1f} ms | ~{ema_fps:.2f} FPS"
            )

            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
