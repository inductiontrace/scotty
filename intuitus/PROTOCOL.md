# Intuitus Protocol

- **Input**: `numpy.ndarray` (BGR H×W×3)
- **Output**: `Optional[List[Tuple[int, int, int, int]]]`  → either a list of `(x1, y1, x2, y2)` boxes or `None` for full-frame processing.
- **Behavior notes**:
  - May return `None` at any time to request full-frame detection.
  - Should enforce periodic keyframes independent of motion.
  - Configurable thresholds; see `intuitus/README.md`.
