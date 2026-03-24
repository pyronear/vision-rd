# Smoke-DETR: Smoke Detection Transformer based on RT-DETR

- **Title:** Smoke Detection Based on an Improved RT-DETR Algorithm (Smoke-DETR)
- **Authors:** Sun, P.; Cheng, H.
- **Year/Venue:** 2024, Fire 7(12), 488
- **PDF:** `pdfs/2024-Smoke-Detection-Transformer-RT-DETR-Sun-Cheng.pdf`

## Key Idea

Smoke-DETR improves on RT-DETR for smoke detection with three modifications:
(1) Enhanced Channel-wise Partial Convolution (ECPConv) in the backbone for
efficient feature extraction by convolving only the most important channels,
(2) Efficient Multi-Scale Attention (EMA) for cross-spatial feature learning,
and (3) Multi-Scale Foreground-Focus Fusion Pyramid Network (MFFPN) with
Rectangular Self-Calibration Modules (RCM) to focus on smoke foreground and
suppress background confusion from clouds/fog.

## Architecture

```
┌──────────────────┐
│ Input 640x640     │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ ECPConvBlock x4   │ channel-select
│ + EMA attention   │ partial conv
│ + downsample      │
└────────┬─────────┘
     S3  S4  S5
         ▼
┌──────────────────┐
│ MFFPN            │
│ ├─ PyramidCtxExt │ RCM x3
│ ├─ FG enhance    │ multi-scale
│ └─ FG fusion     │ foreground focus
└────────┬─────────┘
     P3  P4  P5
         ▼
┌──────────────────┐
│ RT-DETR decoder  │
│ (IoU-aware query)│
└────────┬─────────┘
         ▼
   smoke detections
```

## Results

### Comparison with advanced detectors (custom smoke dataset, 4874 images)

| Model       | Prec.  | Recall | mAP50  | mAP95  | Params | FLOPs  |
|-------------|--------|--------|--------|--------|--------|--------|
| Smoke-DETR  | 0.868  | 0.800  | 0.862  | 0.539  | 16.4M  | 43.3G  |
| RT-DETR     | 0.832  | 0.764  | 0.824  | 0.517  | 19.9M  | 56.9G  |
| YOLOv8m     | 0.813  | 0.784  | 0.826  | 0.527  | 25.8M  | 78.7G  |
| YOLOv9m     | 0.823  | 0.790  | 0.836  | 0.516  | 16.6M  | 60.0G  |
| DINO        | 0.876  | 0.776  | 0.841  | 0.532  | 47M    | 279G   |

- Smoke-DETR: +3.8% mAP50 over RT-DETR baseline with fewer params (16.4M vs 19.9M)
- +3.6% Precision, +3.6% Recall over baseline
- Faster convergence: converges at ~100 epochs vs ~125 for RT-DETR

## Applicability to Pyronear

**Medium-high relevance -- improved single-frame smoke detector for server side.**

- ECPConv reduces computation by only convolving important channels -- could
  help make a lighter backbone for edge deployment
- MFFPN with foreground focus addresses the cloud/fog confusion problem
  critical for Pyronear's outdoor cameras
- 16.4M params and 43.3G FLOPs is still too heavy for RasPi but feasible for
  server-side second stage
- The foreground-focus approach (RCM) is specifically designed to separate
  smoke from background -- valuable for fixed cameras where background is known

**Limitations for Pyronear:**
- Single-frame detector only -- no temporal component
- Custom dataset (4874 images) not publicly released
- mAP95 still low (0.539) -- smoke bbox localization remains imprecise
- RTX 3090 training; inference speed on modest GPU not reported

## Takeaways for Implementation

1. **Partial convolution for efficiency:** ECPConv selects top-K important channels
   via learned weights, convolves only those, reinserts. Can be applied to any
   backbone to reduce FLOPs without much accuracy loss
2. **Foreground-focus FPN:** RCM uses strip convolutions (horizontal + vertical)
   for rectangular self-calibration, then DWConv fusion. This foreground-
   attention module helps distinguish thin smoke from background. Consider
   adding to Pyronear's server-side model
3. **EMA (Efficient Multi-Scale Attention):** Lightweight attention via channel
   grouping, parallel sub-network reconstruction, and cross-spatial learning.
   Adds discriminative features at negligible parameter cost
4. **Non-smoke negatives in training:** Authors include images with smoke-
   resembling objects to reduce FP rate -- consistent with Nemo findings
5. **RT-DETR as baseline:** RT-DETR already a strong real-time transformer
   detector. Smoke-DETR's improvements are incremental but meaningful,
   especially for small/thin smoke targets
