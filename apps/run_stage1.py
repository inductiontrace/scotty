#!/usr/bin/env python3
import os, time, argparse, json, pathlib, yaml, cv2, sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.events import DetectionEvent
from common.loader import load_object


def open_source(src, fps):
    """
    Returns a tuple (read_fn, close_fn, size_fn)
      - read_fn() -> (ok, frame_bgr)
      - close_fn() -> None
      - size_fn() -> (W, H) after first frame
    Supports:
      - "picam"     -> Picamera2 capture_array()
      - int/index   -> OpenCV VideoCapture(index)
      - path/rtsp   -> OpenCV VideoCapture(url)
    """
    if isinstance(src, str) and src.lower() == "picam":
        try:
            from picamera2 import Picamera2
        except Exception as e:
            raise SystemExit("Picamera2 not available. Install with: sudo apt install -y python3-picamera2") from e

        picam = Picamera2()
        # 1280x720 is a good default; adjust if you want
        config = picam.create_video_configuration({"size": (1280, 720)})
        picam.configure(config)
        picam.start()
        time.sleep(0.3)
        first = picam.capture_array()
        H, W = first.shape[:2]

        def read_fn():
            frame = picam.capture_array()
            return True, frame

        def close_fn():
            try:
                picam.stop()
            except Exception:
                pass

        def size_fn():
            return (W, H)

        return read_fn, close_fn, size_fn

    # else OpenCV path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src}")

    # Warm up to get size
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit(f"Failed to read first frame from source: {src}")
    H, W = frame.shape[:2]

    # Rewind if it was a file
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    except Exception:
        pass

    def read_fn():
        ok, f = cap.read()
        return ok, f

    def close_fn():
        cap.release()

    def size_fn():
        return (W, H)

    return read_fn, close_fn, size_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
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

    # OPEN SOURCE (Picamera2 or OpenCV)
    read_fn, close_fn, size_fn = open_source(src, fps)
    W, H = size_fn()

    period = 1.0 / max(0.1, fps)
    last = 0.0
    frame_idx = 0
    ema_fps = None

    try:
        while True:
            now = time.time()
            if now - last < period:
                time.sleep(0.002); continue
            last = now

            ok, frame = read_fn()
            if not ok:
                print("WARNING: Failed to read frame; stopping.")
                break

            t0 = time.time()
            dets = detector.detect(frame)  # [((x1,y1,x2,y2), conf, class)]
            infer_ms = (time.time() - t0) * 1000.0

            for (x1,y1,x2,y2), conf, clazz in dets:
                ev = DetectionEvent(
                    ts_ms=int(time.time()*1000),
                    cam_id=cam_id,
                    frame=frame_idx,
                    track_id_local=-1,
                    global_id=None,
                    clazz=clazz,
                    conf=float(conf),
                    box_xyxy=(int(x1),int(y1),int(x2),int(y2)),
                    img_wh=(int(W),int(H)),
                    have_embed=False,
                    specialist=None,
                    quality=None
                )
                out.write(ev.to_json()+"\n")

            inst_fps = 1000.0 / max(1.0, infer_ms)
            ema_fps = inst_fps if ema_fps is None else 0.9*ema_fps + 0.1*inst_fps
            print(f"frame {frame_idx} | dets {len(dets)} | infer {infer_ms:.1f} ms | ~{ema_fps:.2f} FPS")
            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        out.close()
        close_fn()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
