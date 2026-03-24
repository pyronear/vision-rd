# FLAME: Fire Detection via DNN and Motion Analysis

- **Title:** FLAME: A Framework for Fire Detection in Videos Combining a Deep Neural Network and Motion Analysis
- **Authors:** Gragnaniello, D.; Greco, A.; Sansone, C.; Vento, B.
- **Year/Venue:** 2025, Neural Computing and Applications 37, 6181-6197
- **PDF:** `pdfs/2024-FLAME-Fire-Detection-Video-DNN-Motion-Analysis.pdf`

## Key Idea

FLAME is a three-stage video-level fire detection framework: (1) YOLOv8s detects
flame and smoke candidates frame-by-frame, (2) Gaussian Mixture Model background
subtraction filters out stationary false positives (reflections, puddles, fog), and
(3) a tracking-based motion filter uses a finite state machine (New -> Confirmed /
Occluded -> Rejected) to require temporal persistence before raising an alarm.
The system achieves the best F-Score (93.66) and FDS (74.28) on the ONFIRE
contest benchmark by combining high detector recall with temporal FP suppression.

## Architecture

```
┌──────────────────┐
│ Video stream      │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Stage I: YOLOv8s │
│ 2-class (flame,  │
│ smoke) detector  │ 640x640
│ conf > 0.4       │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Stage II: BG sub │
│ GMM-based        │ Mahalanobis
│ (h=500 frames)   │ distance
│ stationary FP    │
│ removal          │ T_fg threshold
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Stage III: Motion│
│ tracking FSM     │
│ ┌─────────────┐  │
│ │New─▶Confirmed│ │ t_C = 1s
│ │ ▼    ▲       │ │
│ │Occluded──────│ │ t_O = 0.1s
│ │ ▼            │ │
│ │Rejected      │ │
│ └─────────────┘  │
└────────┬─────────┘
         ▼
   FIRE ALARM
   (when Confirmed)
```

## Results

### Ablation study (ONFIRE test set, video-level)

| Detector | BG sub | Tracking | P (%)  | R (%)  | F-Score | D_n (s) |
|----------|--------|----------|--------|--------|---------|---------|
| YOLOv8s  | --     | --       | 74.42  | 94.12  | 83.12   | 4.54    |
| YOLOv8s  | yes    | --       | 76.00  | 93.14  | 83.70   | 5.43    |
| YOLOv8s  | --     | yes      | 92.31  | 94.12  | 93.20   | 9.17    |
| YOLOv8s  | yes    | yes      | **93.20**| 94.12 | **93.66**| 9.20   |

- Motion tracking alone: **+18% Precision**, +10% F-Score
- Cost: notification delay increases from 4.5s to 9.2s (acceptable)

### ONFIRE contest comparison

| Method             | Year | P     | R     | F-Score | FDS   |
|--------------------|------|-------|-------|---------|-------|
| FLAME (ours)       | 2024 | 93.20 | 94.12 | 93.66   | 74.28 |
| Firebusters        | 2023 | 81.15 | 98.02 | 88.79   | 73.06 |
| YOLOv8s baseline   | 2023 | 74.42 | 94.12 | 83.12   | 64.74 |
| Slow-Fast ONFIRE   | 2023 | 77.48 | 87.36 | 82.12   | 54.92 |

### Generalization (Web test set)

| Method    | P     | R     | F-Score | FDS   |
|-----------|-------|-------|---------|-------|
| FLAME     | 92.75 | 87.67 | 90.14   | 75.38 |
| Firebusters| 85.33| 87.67 | 86.49   | 69.63 |

## Applicability to Pyronear

**Very high relevance -- closest architecture to Pyronear's actual system.**

- Uses YOLOv8s as first-stage detector (Pyronear uses YOLOv8)
- Three-stage pipeline with progressive FP reduction mirrors Pyronear's
  design philosophy (edge detector -> server verification)
- Background subtraction exploits fixed camera assumption -- perfectly
  suited to Pyronear's fixed 360-degree cameras
- Motion tracking FSM is simple, interpretable, and parameter-light
- Separate flame/smoke classification branches improve recall

**Key insight:** Motion tracking (stage III) provides the largest single
improvement (+18% Precision), not the DNN or background subtraction.
The temporal persistence requirement (t_C = 1s to confirm) is the most
effective FP filter.

**Limitations for Pyronear:**
- 30s between frames means standard background subtraction (designed for
  video at 10+ fps) needs significant adaptation
- t_C = 1s confirmation time assumes continuous video; at 30s cadence,
  confirmation would require multiple detection cycles (minutes)
- Motion tracking based on centroid distance assumes frame-to-frame
  continuity; with 30s gaps, smoke position can shift significantly
- Training set (LFDN, 36,554 images) is mostly urban fire + flames,
  not specifically wildfire smoke at distance

## Takeaways for Implementation

1. **Temporal state machine for FP reduction:** Implement a per-region FSM with
   states {New, Confirmed, Occluded, Rejected}. At 30s cadence, set t_C to
   2-3 consecutive detections (60-90s) before confirming alarm. This is the
   single most impactful FP reduction technique
2. **Background subtraction for static cameras:** GMM-based background model
   (3 Gaussians per pixel, updated over h=500 frames ~= 4+ hours at 30s).
   Discard detections that don't overlap with foreground (moving pixels).
   Filters reflections, static lights, permanent cloud-like objects
3. **Two-class detection (flame + smoke):** Separate classification heads for
   flame and smoke improve feature learning for each. Fire alarm raised when
   either class is confirmed. Pyronear currently detects only smoke; adding
   flame class could improve close-range fire detection
4. **SGD > Adam for generalization:** Authors found SGD optimizer yields +5%
   F-Score over Adam for YOLOv8. Worth testing for Pyronear's training
5. **FDS metric:** Fire Detection Score combines detection rate, delay, speed,
   and memory. Good holistic metric for evaluating Pyronear's full system
6. **Confidence threshold tuning:** conf=0.4 after grid search (0.1-0.9, step
   0.05) on validation set. Pyronear should tune this similarly
