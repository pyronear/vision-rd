# RT-DETR-Smoke: Real-Time Forest Smoke Detection

- **Title:** RT-DETR-Smoke: Real-Time Forest Smoke Detection Based on Improved RT-DETR
- **Authors:** (not fully specified in first pages; Fire 2025, 8, 170)
- **Year/Venue:** 2025, Fire 8(5), 170
- **PDF:** `pdfs/2025-RT-DETR-Smoke-Real-Time-Forest-Smoke-Detection.pdf`

## Key Idea

RT-DETR-Smoke enhances the RT-DETR-R18 baseline for forest smoke detection with
three key contributions: (1) CoordAtt attention mechanism in the backbone (P4/P5
stages) for positional-aware feature encoding of elongated smoke shapes,
(2) WShapeIoU loss function combining shape-distance, focal weighting, and
monotonic scaling for better bounding-box regression on fluid, irregular smoke
shapes, and (3) uncertainty-minimization query selection and auxiliary prediction
heads in the decoder for hard-to-detect smoke regions.

## Architecture

```
┌──────────────────┐
│ Input 640x640     │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ ResNet18 backbone │
│ + CoordAtt at    │
│   P4 and P5      │ H/V pooling
└────────┬─────────┘
     P3  P4  P5
         ▼
┌──────────────────┐
│ Hybrid encoder   │
│ ├─ AIFI (attn)   │ intra-scale
│ └─ CCFF (CNN)    │ cross-scale
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Transformer      │
│ decoder +        │
│ uncertainty      │ prioritize hard
│ query selection  │ samples
│ + aux pred heads │
└────────┬─────────┘
         ▼
   WShapeIoU loss
   smoke detections
```

## Results

### Loss function comparison (RT-DETR-R18 baseline)

| Loss Function | mAP@0.5 | mAP50-95 |
|---------------|---------|----------|
| WShapeIoU     | 0.8722  | 0.5084   |
| ShapeIoU      | 0.8638  | 0.5071   |
| GIoU          | 0.8598  | 0.5118   |
| CIoU          | 0.8595  | 0.5084   |

### Attention mechanism comparison

| Mechanism     | Params  | GFLOPs | mAP@0.5 | mAP50-95 |
|---------------|---------|--------|---------|----------|
| CoordAtt      | 20.1M   | 58.3   | 0.8775  | 0.5233   |
| SeqPolarAttn  | 21.4M   | 59.5   | 0.8709  | 0.5220   |
| SKAttention   | 20.7M   | 58.3   | 0.8648  | 0.5163   |

### Full model comparison (16,316-image custom dataset)

| Model           | Params  | GFLOPs | mAP@0.5 | FPS    | GPU Time |
|-----------------|---------|--------|---------|--------|----------|
| RT-DETR-Smoke   | 19.9M   | 56.9   | 0.8775  | 445.5  | 2.24 ms  |
| RT-DETR-R18     | 19.9M   | 56.9   | 0.8598  | 506.6  | 1.97 ms  |
| YOLOv8n         | 3.0M    | 8.1    | 0.8467  | 415.4  | 2.41 ms  |
| YOLOv9t         | 2.0M    | 7.8    | 0.8155  | 234.4  | 4.27 ms  |
| YOLOv8-DETR     | 6.1M    | 11.7   | 0.8372  | 433.2  | 2.31 ms  |

- +1.77% mAP@0.5 over baseline with no parameter increase
- 445 FPS on A100 with TensorRT FP16 -- truly real-time
- CoordAtt adds best mAP gain with lowest overhead

## Applicability to Pyronear

**High relevance -- directly applicable to server-side detector.**

- RT-DETR-R18 backbone is relatively lightweight (19.9M params) and fast
  (2.24ms on A100) -- feasible for Pyronear's server GPU
- CoordAtt encodes directional (H/V) positional information -- excellent
  for elongated smoke plumes rising vertically from horizon
- WShapeIoU loss is a drop-in replacement for standard IoU loss that
  improves bbox regression for irregular smoke shapes
- 16,316 images from real outdoor surveillance footage (70% forest, 30% rural)
  is representative of Pyronear's deployment environment

**Limitations for Pyronear:**
- Still a single-frame detector; no temporal verification
- 19.9M params too heavy for RasPi edge deployment
- Custom dataset not publicly released
- FPS measured with TensorRT on A100; actual inference on modest server GPU
  will be slower

## Takeaways for Implementation

1. **CoordAtt as plug-in module:** Simple to add to P4/P5 stages of any backbone.
   Decomposes attention into H and V directions via 1D pooling, then learns
   direction-aware weights. Low overhead, good gains for elongated objects
2. **WShapeIoU loss:** Combines ShapeIoU (penalizes shape deviations) with focal
   factor (prioritizes hard boxes with low IoU) and monotonic scaling. Drop-in
   replacement for CIoU/GIoU in any detector training. Particularly beneficial
   for smoke's irregular, semi-transparent bounding boxes
3. **Uncertainty-minimization query selection:** Select decoder queries with
   highest uncertainty (divergence between localization and classification
   predictions) to force attention on hard-to-detect regions. Useful for
   thin/faint smoke
4. **RT-DETR-R18 as efficient baseline:** Good balance of speed and accuracy for
   server-side smoke detection. Consider as candidate for Pyronear's second
   stage (replacing or complementing current approach)
5. **Auxiliary prediction heads:** Adding prediction heads at multiple decoder
   layers enables iterative refinement -- helps with smoke shape distortions
