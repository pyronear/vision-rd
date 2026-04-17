# Track C prerequisite: confidence threshold target

## Observation

Scanned all 6-column YOLO-format `.txt` label files under
`data/01_raw/datasets/{train,val}/{fp,wildfire}/**/labels/`:

| split   | category   | n_files | n_detections | min    | p01    | p05    | median | max    |
|---------|------------|--------:|-------------:|-------:|-------:|-------:|-------:|-------:|
| val     | fp         |   2339  |       2198   | 0.1001 | 0.1044 | 0.1240 | 0.4847 | 0.9041 |
| val     | wildfire   |   2037  |         72   | 0.1385 | 0.1385 | 0.1672 | 0.6228 | 0.8809 |
| train   | fp         |  22918  |      22753   | 0.1000 | 0.1040 | 0.1219 | 0.4760 | 0.9379 |
| train   | wildfire   |  22016  |        508   | 0.1008 | 0.1119 | 0.1455 | 0.6264 | 0.8813 |

Observed minimum confidence across all FP detections: **0.1000**.

## Interpretation

The training-label YOLO step was filtered at
`confidence_threshold = 0.10`. No detection below 0.10 made it into
the stored labels, and therefore none made it into the classifier's
training tubes.

Deployment runs at `package.infer.confidence_threshold = 0.01` —
**10× below** the training floor. Every YOLO detection in the
`[0.01, 0.10)` band is something the classifier has never seen
during training. At inference those detections feed into tube
construction and get scored by a classifier that wasn't calibrated
for them.

## Track C — experiment 1 target

Set `package.infer.confidence_threshold = 0.10` to align the
deployment detection floor with the training-label floor. Re-package
and re-run `evaluate_packaged` on train + val for both variants;
feed the new `predictions.json` into `analyze_aggregation_rules.py`.

Expected effect: fewer, higher-quality tubes per sequence → lower
per-tube FPR → lower sequence-level FPR.
