# SORT tracker (1 FPS friendly)

We defer hard filtering (per-class thresholds/NMS) and instead rely on
**temporal durability**: a valid object should persist across frames with
overlapping boxes. SORT assigns a stable `track_id_local` when a detection
reappears with sufficient IoU. Configure:

```yaml
tracker:
  impl: "tracker.sort:SORT"
  iou_threshold: 0.3  # association strictness (0.2..0.5 typical)
  max_age: 10         # frames to keep an unmatched track alive
  min_hits: 2         # frames before a new track is emitted
```

At 1 FPS, `max_age: 10` ≈ 10 seconds of occlusion tolerance.
