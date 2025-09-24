# FrameBus (PUB/SUB)

Single producer (camera) → many consumers via ZeroMQ.

- Transport: imagezmq over TCP (PUB/SUB)
- Payload: JPEG frames + JSON metadata header
- Metadata: frame_id, ts, camera_id, w, h, fmt

## Start an edge consumer
```
python3 apps/run_edge.py --config configs/edge_pi5_stage1.yaml
```

`apps/run_edge.py` now launches the hub automatically when `source.kind: zmq` is
configured. The default publisher uses Picamera2 (`--source picam`), but you can
override it with a `publisher` block in your YAML:

```yaml
source:
  kind: zmq
  endpoint: tcp://127.0.0.1:5555
  publisher:
    source: /path/to/video.mp4
    loop: true
    fps: 5
```

Set `source.bind` to override the bind address (defaults to `tcp://*:<port>`)
and `publisher.camera_id` to inject a specific ID.

## Notes
- Consumers skip ahead if they lag; hub stays real-time.
- Multiple consumers can connect simultaneously (HUD, recorder, analytics).
- Switch back to direct camera by setting `source.kind: camera`.
