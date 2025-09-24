# Detection Event Protocol

This document outlines the JSON Lines emitted by the edge detector.

## Frame metadata passthrough
If the source is FrameBus, `frame_id` (int) and `ts` (float seconds) MUST be included in each detection event.
When reading from a direct camera, these fields MAY be omitted.
