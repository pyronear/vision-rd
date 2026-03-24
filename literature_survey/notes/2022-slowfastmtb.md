# A Video-based SlowFastMTB Model for Detection of Small Amounts of Smoke from Incipient Forest Fires

**Authors:** Choi, Kim & Oh
**Year/Venue:** 2022, Journal of Computational Design and Engineering (Oxford)
**PDF:** `pdfs/2022-SlowFastMTB-Incipient-Forest-Fire-Smoke-Choi-et-al.pdf`

## Key Idea

Combines a novel bounding box annotation algorithm (MTB) for objects with fuzzy boundaries like smoke, with a SlowFast 3D CNN for classification. MTB uses pixel-wise frame subtraction to detect moving regions and adaptively scales the bounding box to capture the full smoke plume including semitransparent areas.

## Architecture

```
 V(t)     V(t+Δt)
   │          │
   └────┬─────┘
        ▼
 ┌──────────────┐
 │ MTB Algo     │  1. |V(t+Δt) - V(t)| per pixel
 │              │  2. Threshold → binary mask
 │              │  3. Smallest bbox around white px
 │              │  4. Scale: B_r = B × (r_dt/r_ds)
 └──────┬───────┘
        │ Bounding box B_r
        ▼
 ┌──────────────┐
 │ Crop region  │
 └──────┬───────┘
        │ 1-second video clip (25 frames)
        ▼
 ┌──────────────────┐
 │ SlowFast 3D CNN  │
 │                  │
 │ Slow pathway     │  low fps, full channels
 │   ▲              │  (spatial semantics)
 │   │ lateral      │
 │ Fast pathway     │  high fps, 1/8 channels
 │                  │  (temporal dynamics)
 │                  │
 │ ROI Align +      │
 │ Classifier       │
 └────────┬─────────┘
          │
          ▼
   smoke / no smoke
```

## MTB Algorithm Detail

```
 V(t=10s)       V(t=15s)
 ┌─────────┐    ┌─────────┐
 │ mountain│    │ mountain│
 │      .  │    │    ..:: │
 │     .   │    │   .:::  │
 │ forest  │    │ forest  │
 └────┬────┘    └────┬────┘
      └──────┬───────┘
             ▼
 ┌───────────────────┐
 │ |V(t+Δt) - V(t)|  │  pixel-wise subtraction
 └─────────┬─────────┘
           ▼
 ┌───────────────────┐
 │ Threshold (T=19)  │  change > T → white, else black
 └─────────┬─────────┘
           ▼
 ┌───────────────────┐
 │ Bounding box B    │  smallest box around white px
 │                   │
 │ r_MTB,dt = 0.5    │  50% of box is moving pixels
 │ r_MTB,ds = 0.05   │  want 5% density
 │ Scale: B × 10     │  → enlarge 10x to capture
 │                   │    full fuzzy smoke region
 └───────────────────┘
```

**3 hyperparameters:**
- `Δt` (time interval): gap between subtracted frames
- `T` (threshold): pixel difference cutoff
- `r_MTB,ds` (desired density): controls box scaling factor

## Temporal Modeling

- SlowFast processes 1-second video clips (25 frames at 25fps)
- Slow pathway: temporal ratio α=4 (sees every 4th frame)
- Fast pathway: channel ratio β=0.125 (1/8 channels of slow)
- MTB time intervals tested: 5s, 10s, 15s

## Results

**ANOVA on MTB time interval (CRITICAL for Pyronear):**

| Time interval | Accuracy | P-value |
|:---:|:---:|:---:|
| 5s  | 89.4% | |
| 10s | 91.4% | 0.524 (NOT significant) |
| 15s | 92.4% | |

**Performance on Video B (smoke = 0.48% of image):**

| Method          | Accuracy | FP rate | FN rate | F1    | IoU   |
|-----------------|:---:|:---:|:---:|:---:|:---:|
| CNN (1 frame)   | 50.0% | 10.0% | 40.0% | 0.286 | N/A   |
| Faster R-CNN    | 44.7% | 41.5% | 13.9% | 0.566 | 0.440 |
| SlowFast        | 84.8% | 0.4%  | 14.8% | 0.822 | 0.573 |
| **SlowFastMTB** | **93.3%** | **0%** | **6.7%** | **0.928** | **0.865** |

- CNN at 50% = coin flip. Single frames are useless for tiny smoke.
- SlowFastMTB achieves **zero false positives** with 6.7% false negatives.
- All FN cases were when smoke occupied < 0.04% of the image.

## Applicability to Pyronear

**Very high relevance.** The MTB algorithm is directly applicable.

**MTB as preprocessing for Pyronear:**
- Fixed 360° cameras with static backgrounds make pixel subtraction very effective.
- Subtract frame at t from frame at t+30s → get change mask → find bounding box → scale it.
- The ANOVA result is encouraging: accuracy **improved** from 5s to 15s (89.4% → 92.4%), suggesting longer intervals give smoke more time to develop visible changes. 30s might be even better.
- MTB could run alongside or instead of YOLO for region proposals.
- The `r_MTB,ds` parameter would need re-tuning for 30s intervals (more pixels will have changed → higher r_MTB,ds = less scaling).

**Interaction between Δt and r_MTB,ds (Table 3, P=0.003):**
This is the key finding -- if you increase the time interval, you MUST re-tune r_MTB,ds. They're coupled. For 30s intervals, a parametric study is needed.

**What doesn't transfer:**
- The SlowFast 3D CNN requires dense 1-second clips (25fps). Not applicable at 30s intervals.
- Replace with a CNN+Transformer or TeSTra-style temporal model on the server.

**Zero FP rate** is exactly what Pyronear wants. The MTB bounding box gives the classifier the right region to focus on -- this is the key to high precision.

## Takeaways for Implementation

1. **Implement MTB as a baseline** for Pyronear: pixel subtraction between frames 30s apart, threshold, bounding box, scale. Simple, fast, no GPU needed.
2. **MTB + YOLO fusion**: use MTB to validate YOLO proposals. If YOLO detects smoke AND MTB shows change in the same region → high confidence. If YOLO detects but MTB shows no change → likely false positive (static object).
3. **Re-tune r_MTB,ds for 30s intervals**: parametric study needed.
4. **Grad-CAM analysis showed** SlowFastMTB consistently attends to actual smoke, while CNN/Faster R-CNN look at mountains/roads. This validates the MTB approach.
5. **The 0.48% smoke occupancy** in Video B is realistic for Pyronear (distant smoke plumes are tiny).
