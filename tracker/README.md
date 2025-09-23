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

## Prop-Robust Association

People carrying large props (bags, boxes, umbrellas) can quickly change shape.
Enable the prop-aware association knobs to keep a single `person` track alive
through these transitions:

```yaml
tracker:
  impl: "tracker.sort:SORT"
  class_consistency:
    enabled: true          # refuse cross-class swaps when matching
    penalty: 1000000       # huge cost ⇒ practical hard constraint
  core_center:
    enabled: true          # focus on torso center instead of extremities
    shrink: 0.30           # crop 30% from each side before centering
  scale_gate:
    enabled: true          # asymmetric scale tolerance (growth > shrink)
    grow_tol: 1.60         # allow +60% jump vs. prediction
    shrink_tol: 0.60       # allow -40% drop vs. prediction
  dilate_for_assoc:
    enabled: true          # inflate predicted boxes for matching only
    dx: 0.15               # widen ±15% (total width ×1.30)
    dy: 0.05               # grow height modestly
  duplicate_merge:
    enabled: true          # collapse overlapping duplicates post-update
    iou_thresh: 0.6
    min_frames: 3
```

- **Class consistency**: prevents ID theft by penalizing mismatched classes.
- **Core center**: uses a torso-like center for `person` tracks/detections so
  swinging arms or props have less influence.
- **Scale gate**: allows a person to grow quickly (due to props) while still
  rejecting implausible shrinkage.
- **Dilate for association**: expands predicted `person` regions before scoring
  IoU/centers; rendering still uses the original boxes.
- **Duplicate merge**: after updates, combines overlapping tracks of the same
  class that survived at least `min_frames`.

Disable any feature via `enabled: false` to revert to legacy behavior.
