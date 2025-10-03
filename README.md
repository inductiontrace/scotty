# haecceity — identity that persists across space & time

A Raspberry Pi–friendly, modular pipeline:
  camera → (Intuitus: optional ROI gate) → (Quiddity: detector/classifier)
         → (Haecceity: per-class ID specialists) → events (JSONL/MQTT)

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

1. **Configure the edge consumer** to subscribe to the stream by setting the
   source in your YAML config. `apps/run_edge.py` now launches the publisher on
   your behalf:
   ```yaml
   source:
     kind: zmq
     endpoint: tcp://127.0.0.1:5555
     publisher:
       source: /path/to/video.mp4  # defaults to picam when omitted
       loop: true                  # optional
       fps: 5                      # optional, defaults to edge.fps
       camera_id: dev_cam         # optional override
   ```

2. **Run the pipeline** as usual:
   ```bash
   python apps/run_edge.py -c configs/edge_pi5_stage1.yaml
   ```

Multiple consumers can subscribe to the same publisher simultaneously (HUD,
recorder, analytics, etc.). Switch back to a direct camera feed by setting
`source.kind: camera` in your config. Override the bind address by setting
`source.bind` (defaults to `tcp://*:<port>` based on the consumer endpoint).

## Live web viewer
Enable the lightweight MJPEG viewer to see the same overlay that is written to
MP4 files. Toggle it via the `edge.web` block in your YAML config:

```yaml
edge:
  web:
    enabled: true
    host: 0.0.0.0   # bind address for the HTTP server
    port: 8080      # open http://<host>:8080/ in your browser
    title: "pi5_cam_a live"
```

Every frame published to the viewer is annotated with bounding boxes and the
HUD (if enabled) before being encoded as JPEG for streaming.

## Repository layout
- `intuitus/`    – optional motion/ROI gating modules
- `quiddity/`    – detectors/classifiers
- `haecceity/`   – ID specialists and embedding helpers
- `common/`      – interfaces, events schema, loader utilities
- `apps/`        – runnable entrypoints
- `configs/`     – YAML config sets

License: MIT (edit as needed).
