# Quiddity Protocol

- **Input**: `numpy.ndarray` (BGR H×W×3)
- **Output**: `List[((x1, y1, x2, y2), conf: float, class: str)]`
- **Constraints**:
  - All coordinates must be clamped to image bounds.
  - `conf` ∈ [0, 1]; `class` must be a short string label.
  - Apply NMS internally before returning results.
