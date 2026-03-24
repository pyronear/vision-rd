# Wildfire Smoke Detection with CCPE and Swin Transformer

- **Title:** Wildfire Smoke Detection Based on Cross Contrast Patch Embedding and Separable Negative Sampling
- **Authors:** Wang, C.; Xu, C.; Zhang, Q.; Shan, Z.; Wang, Z.
- **Year/Venue:** 2025 (preprint submitted to IJIS)
- **PDF:** `pdfs/2025-Wildfire-Smoke-Detection-CCPE-Swin-Transformer-Wang-et-al.pdf`

## Key Idea

Transformers are strong at global context but weak at capturing the low-level texture
and contrast cues that distinguish smoke from clouds/fog. This paper replaces the
standard patch embedding of Swin Transformer with a Cross Contrast Patch Embedding
(CCPE) module that captures multi-scale horizontal and vertical contrast patterns
via column/row shift-and-subtract operations. Additionally, a Separable Negative
Sampling Mechanism (SNSM) separately handles negatives from smoke-containing vs
smoke-free images to address the boundary ambiguity problem unique to smoke
detection.

## Architecture

```
┌──────────────────┐
│ Input 640x640     │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ CCPE module      │
│ ├─ H contrast   │ column shifts
│ │  (multi-stride)│ at 8 strides
│ └─ V contrast   │ row shifts
│   concat+conv    │ -> 48ch
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Swin Transformer │
│ Tiny (4 stages)  │
│ ├─ Stage1 x2    │
│ ├─ Stage2 x2    │
│ ├─ Stage3 x6    │
│ └─ Stage4 x2    │
└────────┬─────────┘
     P2  P3  P4
         ▼
┌──────────────────┐
│ PAFPN neck       │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ YOLOX head       │
│ + SNSM loss      │ separable neg
│ (cls/reg/conf)   │ sampling
└──────────────────┘
```

## Results

### CCPE ablation (SKLFS-WildFire Test set)

| Model              | Backbone         | BBox AP@0.1 | Img AUC | Vid AUC |
|--------------------|------------------|-------------|---------|---------|
| YOLOX              | CSPDarknet Large | 0.506       | 0.712   | 0.900   |
| YOLOX-Swin         | Swin Tiny        | 0.476       | 0.732   | 0.909   |
| YOLOX-ContrastSwin | CCPE + Swin Tiny | **0.537**   | **0.765**| 0.908  |

### SNSM ablation

| Sampling    | BBox AP@0.1 | Img AUC | Vid AUC |
|-------------|-------------|---------|---------|
| None (base) | 0.537       | 0.765   | 0.908   |
| OHEM        | 0.574       | 0.783   | 0.924   |
| SNSM (ours) | 0.503       | **0.900**| **0.934**|

- SNSM trades box-level AP for massive image/video-level gains (+13.5% Img AUC)

### FIgLib comparison (multi-frame, 2 frames)

| Model                | Backbone     | Acc(%) | F1(%) | Prec(%) | Recall(%) |
|----------------------|-------------- |--------|-------|---------|-----------|
| Ours (2-frame)       | ContrastSwin | 84.67  | 84.05 | 88.74   | 79.83     |
| SmokeyNet (3-frame)  | ResNet34+ViT | 83.62  | 82.83 | 90.85   | 76.11     |

### Comparison with classical detectors (SKLFS-WildFire)

| Model      | Img AUC | Vid AUC | Params (M) | GFLOPs |
|------------|---------|---------|------------|--------|
| Ours       | 0.900   | 0.934   | 35.9       | 53.3   |
| Sparse RCNN| 0.852   | 0.914   | 125.2      | 472.9  |
| YOLOX      | 0.712   | 0.900   | 54.2       | 155.7  |
| Faster RCNN| 0.598   | 0.858   | 60.3       | 566.6  |

## Applicability to Pyronear

**High relevance -- addresses the exact visual challenges of wildfire smoke.**

- CCPE is lightweight (only adds shift+subtract+conv) yet significantly
  improves smoke texture discrimination -- exactly what's needed to
  distinguish smoke from clouds/fog on fixed cameras
- SNSM addresses the boundary ambiguity problem of smoke (where does smoke
  end?), improving image-level and video-level recall at the cost of
  box-level precision. For fire alarm systems, this trade-off is excellent:
  detecting fire presence matters more than precise bbox
- The SKLFS-WildFire dataset (30,470 videos, open-scene early fire) is
  directly relevant; test set publicly available
- 2-frame temporal model achieves SOTA on FIgLib with simple channel
  concatenation -- applicable to Pyronear's 30s frame pairs

**Limitations for Pyronear:**
- Swin Transformer Tiny (35.9M params) too heavy for RasPi
- CCPE designed for Transformer patch embedding; not directly applicable
  to CNN-based edge detectors
- Still primarily a single-frame detector (multi-frame via simple concat)

## Takeaways for Implementation

1. **Cross-contrast as preprocessing:** The shift-and-subtract idea could be
   applied as a preprocessing step before any backbone: compute H/V contrast
   maps at multiple strides, concatenate with RGB. Captures smoke texture
   cues that plain convolutions miss
2. **Separable negative sampling:** In training, separate negatives from
   smoke-containing images (hard negatives near smoke boundary) vs pure
   background images. Apply different sampling ratios to each group.
   Dramatically improves recall at image/video level
3. **2-frame temporal input:** Concatenate current frame with frame from
   t-2 minutes ago in channel dimension. Simple but effective temporal
   signal for Swin/Transformer models. At 30s cadence, use t and t-1
   (60s apart) for analogous approach
4. **Video-level AUC as primary metric:** For alarm systems, image/video-level
   classification metrics (AUC, F1) are more relevant than bounding-box AP.
   A fire is either present or not -- localization precision is secondary
5. **SKLFS-WildFire dataset:** Use this public early-fire test set for
   benchmarking. Strict train/test geographic split prevents data leakage
