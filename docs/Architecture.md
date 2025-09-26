# Scotty Architecture

Scotty is an edge-friendly perception stack that streams frames from embedded
hardware, detects objects, tracks them over time, and enriches the resulting
tracks with appearance signatures. The platform is composed of loosely coupled
modules that communicate through well-defined interfaces and configuration
contracts.

```mermaid
graph TD
    Source[Camera / Video Source]
    FrameBus[framebus.hub<br>ZeroMQ Publisher]
    EdgeApp[apps/run_edge.py]
    Detector[quiddity.* Detector]
    Tracker[tracker/*]
    Registry[haecceity.registry]
    Specialists[haecceity specialists]
    Intuitus[intuitus telemetry]
    WebViewer[common.webviewer]

    Source -->|frames| FrameBus
    FrameBus -->|JPEG frames| EdgeApp
    EdgeApp -->|RGB tensors| Detector
    Detector -->|detections| Tracker
    Tracker -->|tracks| Registry
    Registry -->|embedding requests| Specialists
    Specialists -->|signatures| Registry
    Registry -->|enriched events| EdgeApp
    EdgeApp -->|events| Intuitus
    EdgeApp -->|video overlays| WebViewer
```

## Component overview

### Frame ingestion (`framebus/`)
The frame bus provides a ZeroMQ hub capable of reading from PiCamera2, video
files, or OpenCV devices. It publishes JPEG-encoded frames plus metadata to
consumers such as `apps/run_edge.py`.

### Edge application (`apps/`)
`apps/run_edge.py` orchestrates the full perception pipeline. It reads frames
from FrameBus or direct sources, loads configured detectors, trackers, and
haecceity specialists, and produces annotated frames plus structured events. The
script also manages lifecycle concerns such as graceful shutdown, MQTT
publication, and interactive diagnostics.

### Detection (`quiddity/`)
Detectors implement the inference stage that turns raw frames into bounding
boxes. The loader in `common.loader` instantiates detector classes based on
configuration entries.

### Tracking (`tracker/`)
Tracking modules associate detections over time. The modern implementation lives
in `tracker/` and is wired through configuration. Legacy experimental trackers
have been removed during this cleanup to avoid confusion.

### Embedding and identity (`haecceity/`)
Haecceity manages specialists that compute embeddings for tracks, merges fallback
models, and maintains a global registry of observed identities.

### Telemetry (`intuitus/`)
Intuitus modules capture runtime metrics and emit them to observability backends.
The edge app wires the configured telemetry publisher to receive events and
health reports.

### Utilities (`common/` and `docs/`)
Shared utilities cover geometry math, overlay drawing, and video helpers. The
new `tests/` structure separates quick unit tests from media-driven scenarios,
and `docs/` hosts reference material such as this architecture guide.
