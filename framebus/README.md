# FrameBus (PUB/SUB)

Single producer (camera) → many consumers via ZeroMQ.

- Transport: imagezmq over TCP (PUB/SUB)
- Payload: JPEG frames + JSON metadata header
- Metadata: frame_id, ts, camera_id, w, h, fmt

## Start the publisher
```
. venv/bin/activate
# Picamera2 on-device
python3 -m framebus.hub --source picam

# OR stream a video/file/webcam when developing on a laptop
python3 -m framebus.hub --source /path/to/video.mp4 --loop
```

Other useful options:

- `--endpoint tcp://*:5555` – change the bind address/port.
- `--camera-id my_cam` – override the camera identifier stamped into metadata.
- `--fps 5` – throttle publish rate (set `0` to disable throttling).

## Start an edge consumer
```
python3 apps/run_edge.py --config configs/edge_pi5_stage1.yaml
```

Ensure the config has `source.kind: zmq` and `source.endpoint: tcp://127.0.0.1:5555`.

## Notes
- Consumers skip ahead if they lag; hub stays real-time.
- Multiple consumers can connect simultaneously (HUD, recorder, analytics).
- Switch back to direct camera by setting `source.kind: camera`.
