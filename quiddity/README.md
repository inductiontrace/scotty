# Quiddity — detector/classifier

## Protocol
- **Input**: BGR frame (H×W×3).
- **Output**: list of detections `[((x1, y1, x2, y2), conf, class_name)]`.

## Config keys (example)
```yaml
quiddity:
  impl: "quiddity.yolo_ort:YOLOv8ORT"
  model_path: "models/yolov8n.onnx"
  img_size: 512
  conf_th: 0.35
  iou_th: 0.5
  classes_include: ["person", "car"]
```

Swap models by changing `impl` and `model_path` in the config.
