# Intuitus — motion/ROI gate (optional)

## Protocol
- **Input**: BGR frame (`numpy.ndarray`, shape H×W×3).
- **Output**: list of ROI rectangles `[(x1, y1, x2, y2)]` *or* `None` to request full-frame.

## Config keys (example)
```yaml
intuitus:
  enabled: true
  keyframe_sec: 2.0
  min_blob_area: 300
  dilate: 11
  max_roi: 8
  max_area_frac: 0.4
```

## Implementation notes
- If total ROI area > `max_area_frac` or no blobs are detected → return `None` (full-frame).
- Always force a keyframe every `keyframe_sec` seconds.
