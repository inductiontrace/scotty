# haecceity — identity that persists across space & time

A Raspberry Pi–friendly, modular pipeline:
  camera → (Intuitus: optional ROI gate) → (Quiddity: detector/classifier)
         → tracker → (Haecceity: per-class ID specialists) → events (JSONL/MQTT)

## Design goals
- **Config-first**: swap models and modules via YAML.
- **Pluggable**: dynamic Python module loading behind stable interfaces.
- **Privacy-aware**: ship embeddings/metadata, not frames, by default.
- **Pi 5 friendly**: CPU-only works; accelerators optional later.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python apps/run_edge.py -c configs/edge_pi5.yaml
```

## Repository layout
- `intuitus/`    – optional motion/ROI gating modules
- `quiddity/`    – detectors/classifiers
- `haecceity/`   – ID specialists + global registry
- `common/`      – interfaces, events schema, loader utilities
- `apps/`        – runnable entrypoints
- `configs/`     – YAML config sets

License: MIT (edit as needed).
