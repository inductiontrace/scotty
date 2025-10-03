# Haecceity — per-class identity specialists

## Protocol
- **Input**: detections with class labels plus crops from the frame.
- **Output**: optional embeddings per detection for downstream analytics.

## Config keys (example)
```yaml
haecceity:
  min_bbox_h_frac: 0.08
  min_sharpness: 25.0
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
- Downstream consumers can compare embeddings directly via cosine similarity.

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

  specialists:
    - impl: "haecceity.person_torchreid:PersonTorchreid"
      model_name: "osnet_x0_25"   # auto-download
      crop: 192
      min_bbox_h_frac: 0.08
      min_sharpness: 25.0

  fallbacks: []
```
