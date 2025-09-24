# FrameBus (PUB/SUB)

Single producer (camera) → many consumers via ZeroMQ.

- Transport: imagezmq over TCP (PUB/SUB)
- Payload: JPEG frames + JSON metadata header
- Metadata: frame_id, ts, camera_id, w, h, fmt

## Start the publisher
```
. venv/bin/activate
python3 -m framebus.hub
```

## Start an edge consumer
```
python3 apps/run_edge.py --config configs/edge_pi5_stage1.yaml
```

Ensure the config has `source.kind: zmq` and `source.endpoint: tcp://127.0.0.1:5555`.

## Notes
- Consumers skip ahead if they lag; hub stays real-time.
- Multiple consumers can connect simultaneously (HUD, recorder, analytics).
- Switch back to direct camera by setting `source.kind: camera`.
