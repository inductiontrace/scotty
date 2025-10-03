# Scotty Contribution Guide

This repository powers Scotty's edge analytics stack. Use the following
conventions when extending the project:

## High-level layout

| Directory      | Purpose |
| -------------- | ------- |
| `apps/`        | Entry points and orchestration scripts that wire components together for a deployment scenario. |
| `common/`      | Cross-cutting utilities (math helpers, video I/O, overlays, events) that may be shared by multiple services. Add new helpers here only when they are clearly generic. |
| `configs/`     | YAML configuration bundles that describe how to compose detectors, classifiers, and enrichers. |
| `docs/`        | End-user and internal documentation, architecture notes, and protocol descriptions. |
| `framebus/`    | ZeroMQ-based frame distribution utilities. New transports or hub features live here. |
| `haecceity/`   | Entity-signature generation (feature embedding) logic and registries. Add new specialists or fallback implementations here. |
| `intuitus/`    | Telemetry and health monitoring integrations. |
| `quiddity/`    | Detection models and pipelines. Introduce new detectors in this package. |
| `tests/`       | Automated unit tests (`tests/unit/`) and media-driven scenario manifests (`tests/scenarios/`). |

## Implementation notes

* Keep third-party integration shims self-contained inside the closest relevant
  package (for example, new MQTT publishers belong in `intuitus/`).
* Prefer dependency injection via configuration over hard-coded imports. Entry
  points in `apps/` should remain thin wiring layers.
* When adding documentation that explains the system as a whole, place it in
  `docs/` and link it from `docs/Architecture.md` when appropriate.
* Scenario `.mp4`-based regression assets belong in `tests/scenarios/` next to a
  `test.json` manifest that records how to replay the clip and what to assert.

Following these conventions keeps the boundary between inference and analytics
layers clear for future contributors.
