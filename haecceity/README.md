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

## Person re-ID via torchreid (OSNet x0.25)

**Why `osnet_x0_25`?**
- Lightweight backbone suitable for Raspberry Pi 5 CPU.
- Strong baseline for person re-identification with good generalization.
- No external model file needed: `torchreid` auto-downloads weights.

**What does it output?**
- A fixed-length, L2-normalized embedding vector (np.float32) for each qualifying person crop.
- Haecceity’s registry uses cosine similarity + hysteresis to assign a stable `global_id`.

**How to enable / disable**
- Enable by adding the specialist to `haecceity.specialists` and include `"person"` in `haecceity.embed_classes`.
- Disable embeddings (Stage-1) by setting:
  ```yaml
  haecceity:
    embed_classes: []
    specialists: []
    fallbacks: []
  ```
  The system fails soft: if a specialist references a missing model_path, it logs a warning and continues.

Example config (enable person embeddings via torchreid):
```
haecceity:
  embed_classes: ["person"]     # only embed for persons
  new_id_threshold: 0.55
  hysteresis: 0.05
  embed_interval_frames: 3

  specialists:
    - impl: "haecceity.person_torchreid:PersonTorchreid"
      model_name: "osnet_x0_25"   # auto-download
      crop: 192
      min_bbox_h_frac: 0.08
      min_sharpness: 25.0

  fallbacks: []
```
