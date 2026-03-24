# Two-Step Real-Time Night-Time Fire Detection Using Static ELASTIC-YOLOv3 and Temporal Fire-Tube

**Authors:** Park & Ko
**Year/Venue:** 2020, Sensors (MDPI)
**PDF:** `pdfs/2020-Night-Time-Fire-Detection-ELASTIC-YOLOv3-Fire-Tube-Park-Ko.pdf`

## Key Idea

Two-step pipeline: (1) ELASTIC-YOLOv3 detects fire candidates in each frame, (2) a temporal "fire-tube" stacks N=50 cropped fire regions backward in time with adaptive skip-frame selection, extracts Histogram of Optical Flow (HoF) features, converts to Bag-of-Features (BoF), and classifies with a Random Forest. The temporal step filters false positives (neon signs, headlights) that look like fire in single frames but lack dynamic motion.

## Architecture

```
Step 1: ELASTIC-YOLOv3 (per frame, GPU)
═══════════════════════════════════════

  Frame ──▶ ELASTIC-Darknet53 ──▶ YOLOv3 head ──▶ Fire candidate bboxes
                │
       ┌────────┴────────┐
       │  ELASTIC Block  │
       │                 │
       │  Path1: 3×conv  │──┐
       │  (full res)     │  │ concat
       │                 │  ├──────▶ output
       │  Path2: AvgPool │  │
       │  → 2×conv →     │──┘
       │  Upsample       │
       └─────────────────┘
       (multi-scale features without extra params)


Step 2: Fire-Tube + BoF + Random Forest (CPU)
══════════════════════════════════════════════

  Fire candidate regions across N=50 frames (backward in time)
  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐
  │ f_50 │ │ f_49 │ │ f_48 │ │ f_47 │ ... │ f_1  │  ← current
  │(crop)│ │(crop)│ │(crop)│ │(crop)│     │(crop)│
  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘     └──┬───┘
     │        │        │        │             │
     └────────┴────────┴────────┴─────────────┘
                    Fire-Tube (3D volume)
                         │
                    Optical Flow
                    between pairs
                         │
                    ┌────┴────┐
                    │  HoF    │  9 orientations per frame
                    │  + SPP  │  (1×1 + 2×2 = 5 blocks)
                    │         │  → 45 dims/frame
                    │         │  × 49 pairs = 2205 dims
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │  BoF    │  K-means codebook (K=80)
                    │ mapping │  → 80-dim histogram
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │ Random  │  120 trees, depth 20
                    │ Forest  │
                    └────┬────┘
                         │
                    fire / non-fire
```

## Fire-Tube Skip-Frame Selection

```
Algorithm: Build fire-tube backward from current frame

  current frame i ──▶ F-Tube[0]

  for each previous frame (i-1, i-2, ...):
      compute HoF magnitude between frame and F-Tube[k]
      if |HoF_mag difference| ≥ threshold:
          stack into F-Tube[k+1]     ← significant motion change
      else:
          skip (max 3 consecutive)   ← redundant frame

  repeat until F-Tube has N=50 frames
```

This ensures only frames with **meaningful motion change** enter the tube. Distant fires (slow motion) may require looking further back in time.

## Temporal Modeling

- N=50 frames accumulated into the fire-tube
- Adaptive selection based on HoF magnitude change (skip redundant frames, max 3 skips)
- The tube adapts size/location per frame (nonlinear, tracks actual fire region)
- HoF captures motion direction and magnitude of the fire/candidate across time

## Results

**Step 1 - ELASTIC-YOLOv3 vs baselines:**

| Method         | Precision | Recall | F1    | Time (ms) |
|----------------|:---------:|:------:|:-----:|:---------:|
| SSD            | 99.0%     | 67.7%  | 80.4  | 22.5      |
| Faster R-CNN   | 81.7%     | 94.5%  | 87.2  | 45.0      |
| YOLOv3         | 97.9%     | 91.2%  | 94.3  | 15.9      |
| ELASTIC-YOLOv3 | 98.8%     | 97.0%  | 97.9  | 16.0      |

ELASTIC block: +10.2pp recall on small fires vs YOLOv3, only +0.1ms.

**Step 2 - Temporal verification (all use ELASTIC-YOLOv3 as Step 1):**

| Verification Method      | Precision | Recall | F1    | Time (ms) | Hardware     |
|--------------------------|:---------:|:------:|:-----:|:---------:|:------------:|
| + 3D ConvNets            | 97.9%     | 89.3%  | 89.3  | 20.4      | GPU          |
| + LSTM                   | 76.7%     | 99.5%  | 86.6  | 56.0      | GPU          |
| **+ Fire-tube RF**       | **96.7%** | **99.5%** | **98.0** | **24.8** | **GPU+CPU** |

- LSTM has terrible precision (76.7%) -- too many false positives pass through
- 3D ConvNets miss fires (89.3% recall)
- Fire-tube+RF: best of both worlds -- 96.7% precision AND 99.5% recall

**Codebook size (Fig 6):** K=80 optimal. Below 80: recall and precision both increase. Above 150: recall drops sharply (outlier patterns get isolated clusters).

## Applicability to Pyronear

**The two-step pattern maps exactly to Pyronear's architecture:**
```
Pyronear:   YOLOv8 on Pi  ──▶  Temporal model on server
Fire-Tube:  ELASTIC-YOLOv3 ──▶  Fire-tube + RF
```

**What transfers:**
1. **The "tube" concept**: stack cropped YOLO detection regions across time into a 3D volume. With fixed cameras, this captures how the candidate region evolves (smoke grows, false positive stays static).
2. **BoF + Random Forest as a fast baseline**: very lightweight, runs on CPU, surprisingly effective (98.0% F1). Could be a quick first implementation.
3. **Two-step precision boost**: Step 1 alone = 97.9% F1. Adding temporal verification = 98.0% F1. Small overall gain but the critical improvement is **rejecting false positives** that the detector can't distinguish spatially.

**What breaks at 30s intervals:**
- Optical flow between frames 30s apart is meaningless (too much displacement)
- Skip-frame logic assumes dense upstream frames (max 3 skips)
- HoF features trained on dense motion patterns won't match sparse patterns

**Adaptations needed:**
- Replace HoF with **background subtraction features**: pixel diff against reference, histogram of change magnitudes, area/growth rate of changed region
- Replace optical flow with **CNN features per frame**: extract EfficientNet/ResNet features from each crop, use as the tube's per-frame representation
- The BoF+RF classifier could still work on these alternative features

## Takeaways for Implementation

1. **Start with the two-step pattern**: YOLO proposes, temporal model verifies. This paper proves it works.
2. **RF on tabular temporal features as baseline**: extract simple features from N smoke crops (area, intensity, growth rate, centroid shift) → RF classifier. Fast to implement, no GPU needed for verification.
3. **The LSTM comparison is sobering**: LSTM achieved 99.5% recall but only 76.7% precision -- it's bad at rejecting false positives. The fire-tube+RF approach is much better at discrimination.
4. **The ELASTIC block** for multi-scale detection could improve YOLOv8 on the Pi for small smoke detection, but this is orthogonal to the temporal model.
