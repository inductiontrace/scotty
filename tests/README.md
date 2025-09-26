# Scotty Test Layout

The `tests/` tree is split into two complementary areas:

* `unit/` – fast, deterministic tests that exercise individual functions or
  classes. These are picked up automatically by `pytest`.
* `scenarios/` – media-driven regression scenarios. Each scenario should live in
  its own subdirectory that contains:
  * `test.json` – the command to execute and the expectations to assert.
  * Supporting artifacts such as `.mp4` clips or sidecar files that the command
    consumes.

A scenario directory therefore looks like:

```
scotty/
└── tests/
    ├── unit/
    │   └── test_*.py
    └── scenarios/
        └── <scenario-name>/
            ├── test.json
            └── clip.mp4
```

The JSON manifest is intentionally flexible so that it can describe exact or
fuzzy assertions. A simple template:

```json
{
  "command": "python apps/run_edge.py --config configs/edge_pi5.yaml --source clip.mp4",
  "expects": {
    "type": "fuzzy",
    "checks": [
      {"field": "events[0].label", "contains": "person"}
    ]
  }
}
```

Future tooling can load each `test.json`, execute the command, and compare the
output according to the declared expectation strategy.
