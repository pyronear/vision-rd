# Temporal Fire-Tube

## Objective

Reduce false positives from YOLO-based smoke/fire detection by applying a temporal
verification step inspired by the fire-tube method (Park & Ko, 2020). This experiment builds 3D "tubes" from tracked detections across frames, extracts temporal features from each tube, and classifies with a Random Forest.

## Approach

Adapted two-step pipeline from "Two-Step Real-Time Night-Time Fire Detection
Using Static ELASTIC-YOLOv3 and Temporal Fire-Tube" (Park & Ko, Sensors 2020):

1. **YOLO inference** -- YOLOv11s detects smoke candidates in each frame.
2. **Padding** -- Short sequences padded to minimum length by repeating boundary frames.
3. **Tube construction** -- Detections tracked across frames via greedy IoU matching. For each tracked chain, the corresponding image regions are cropped and resized into a fire-tube.
4. **Feature extraction** -- Tabular features (24 dims) computed per tube: area change, centroid shift, intensity change, histogram distance, confidence -- aggregated across consecutive pairs plus global features.
5. **Classification** -- Random Forest (120 trees, depth 20) with balanced class weights classifies each tube as smoke/non-smoke.
6. **Sequence decision** -- Positive if any tube is classified positive.

**Key adaptation for Pyronear's 30s cadence:** The original paper uses optical flow on dense video (30fps). At 30-second intervals optical flow is meaningless, so we replace HoF features with tabular temporal features that capture how the detection region evolves over time.

## Data

Same dataset as `tracking-fsm-baseline` (shared `01_raw` DVC data):

- Train: ~1,034 wildfire + ~1,433 false positive sequences (Pyronear only)
- Val: ~112 wildfire + ~147 FP sequences (Pyronear only)
- Ground truth: sequence-level binary labels

## Results

*TBD -- run `dvc repro` to generate.*

## Pipeline

```
infer (01_raw -> 02_intermediate)
  -> pad (02_intermediate -> 03_primary)
    -> build_tubes (03_primary + 01_raw -> 04_feature)
      -> extract_features (04_feature -> 05_model_input)
        -> train (05_model_input/train -> 06_models)
          -> predict (06_models + 05_model_input -> 07_model_output)
            -> evaluate (07_model_output -> 08_reporting)
```

## How to Reproduce

```bash
make install          # Install dependencies
dvc pull              # Download data from S3
dvc repro             # Run full pipeline
make lint             # Run linter
make test             # Run tests
```

