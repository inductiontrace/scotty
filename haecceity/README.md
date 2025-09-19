# Haecceity — per-class identity specialists + global ID registry

## Protocol
- **Input**: tracked detections with class labels + crops from frame.
- **Output**: (optional) embeddings per track and a stable `global_id` assigned via
  similarity to class-specific centroids with hysteresis.

## Config keys (example)
```yaml
haecceity:
  new_id_threshold: 0.55
  hysteresis: 0.05
  min_bbox_h_frac: 0.08
  min_sharpness: 25.0
  embed_interval_frames: 3
  specialists:
    - impl: "haecceity.person_osnet:PersonOSNet025"
      model_path: "models/osnet_x0_25.onnx"
      crop: 192
    - impl: "haecceity.vehicle_signature:VehicleSignature"
      hist_bins: 32
  fallbacks:
    - impl: "haecceity.generic_osnet:GenericOSNet025"
      model_path: "models/osnet_x0_25.onnx"
      crop: 192
```

Write specialists that implement `common.interfaces.IdSpecialist`.
