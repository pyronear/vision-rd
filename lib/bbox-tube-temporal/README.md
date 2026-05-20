# bbox-tube-temporal-core

Inference core for the bbox-tube temporal smoke classifier — a **mirror copy** of
the serving-path modules from
`experiments/temporal-models/bbox-tube-temporal/src/bbox_tube_temporal/`.

The production temporal-model API (`services/temporal-model-api/`) depends on this
package so it never imports the experiment. The copy keeps the same import name
and file layout as the experiment so the two can be `diff`-ed to detect drift.
Deduplication (the experiment depending on this lib) is deferred — see
`docs/specs/2026-05-20-temporal-model-serving-design.md` (Future work).
