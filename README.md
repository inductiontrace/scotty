# haecceity — identity that persists across space & time

A Raspberry Pi–friendly, modular pipeline:
  camera → (Intuitus: optional ROI gate) → (Quiddity: detector/classifier)
         → tracker → (Haecceity: per-class ID specialists) → events (JSONL/MQTT)

## Design goals
- **Config-first**: swap models and modules via YAML.
- **Pluggable**: dynamic Python module loading behind stable interfaces.
- **Privacy-aware**: ship embeddings/metadata, not frames, by default.
- **Pi 5 friendly**: CPU-only works; accelerators optional later.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python apps/run_edge.py -c configs/edge_pi5.yaml
```

## Streaming frames over ZeroMQ
Scotty can ingest frames published over the network via the `framebus` helper,
which uses [`imagezmq`](https://github.com/jeffbass/imagezmq) (ZeroMQ PUB/SUB)
to broadcast JPEG frames and metadata.

1. **Start a publisher** on the machine with camera access:
   ```bash
   source .venv/bin/activate
   # Publish from a Pi camera (Picamera2)
   python -m framebus.hub --source picam

   # or stream a video file / USB webcam while developing on a laptop
   python -m framebus.hub --source /path/to/video.mp4 --loop
   ```
   Useful options include `--endpoint tcp://*:5555` to change the bind
   address/port, `--camera-id my_cam` to override the camera identifier, and
   `--fps 5` to throttle the publish rate (set `0` to disable throttling).

2. **Configure the edge consumer** to subscribe to the stream by setting the
   source in your YAML config. For example:
   ```yaml
   source:
     kind: zmq
     endpoint: tcp://127.0.0.1:5555
   ```

3. **Run the pipeline** as usual:
   ```bash
   python apps/run_edge.py -c configs/edge_pi5_stage1.yaml
   ```

Multiple consumers can subscribe to the same publisher simultaneously (HUD,
recorder, analytics, etc.). Switch back to a direct camera feed by setting
`source.kind: camera` in your config.

## Repository layout
- `intuitus/`    – optional motion/ROI gating modules
- `quiddity/`    – detectors/classifiers
- `haecceity/`   – ID specialists + global registry
- `common/`      – interfaces, events schema, loader utilities
- `apps/`        – runnable entrypoints
- `configs/`     – YAML config sets

License: MIT (edit as needed).
