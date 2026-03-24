# Real-Time Fire & Smoke Detection via ViT + 3D-CNN Spatiotemporal Learning

- **Title:** Real-Time Fire and Smoke Detection Using Vision Transformer and 3D-CNN for Enhanced Spatiotemporal Learning
- **Authors:** (Scientific Reports, 2026, 16:8928)
- **Year/Venue:** 2026, Scientific Reports 16, 8928 (doi:10.1038/s41598-026-36687-9)
- **PDF:** `pdfs/2026-Real-Time-Fire-Smoke-ViT-Spatiotemporal-Learning.pdf`

## Key Idea

A hybrid multimodal model that combines Vision Transformers (ViT) for spatial
feature extraction from static images with 3D-CNNs for spatiotemporal feature
extraction from video sequences. Features from both branches are fused via
concatenation + cross-attention weighting, then processed by a Transformer
encoder to capture temporal dependencies. The model classifies fire/smoke/non-fire
for both images and videos. Class-weighted BCE loss handles fire/non-fire imbalance.

## Architecture

```
┌─────────┐  ┌──────────────┐
│  Image  │  │ Video frames  │
│(512x512)│  │ (10 @ 1fps)  │
└────┬────┘  └──────┬───────┘
     ▼               ▼
┌─────────┐  ┌──────────────┐
│   ViT   │  │  3D-CNN      │
│ patch   │  │  3x3x3 kern  │
│ embed + │  │  + BN + ReLU │
│ MSA     │  │  + MaxPool3D │
└────┬────┘  └──────┬───────┘
     │  spatial      │ temporal
     │  features     │ features
     ▼               ▼
┌──────────────────────┐
│ Fusion Layer          │
│ align + concat/add   │
│ cross-attention wt.  │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Transformer Encoder  │
│ (pos.enc + MSA +     │
│  FFN + residual)     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ FC + Softmax         │
│ fire / smoke /       │
│ non-fire             │
└──────────────────────┘
```

## Results

### NASA Space Apps Challenge (clear daylight, images)

| Model              | Acc (%) | Prec (%) | Recall (%) | F1 (%) | AUC-ROC (%) |
|--------------------|---------|----------|------------|--------|-------------|
| Proposed hybrid    | 99.2    | 99.3     | 99.0       | 99.1   | 99.5        |
| ResNet50           | 90.5    | 89.7     | 91.1       | 90.4   | 92.3        |
| VGG16              | 87.6    | 85.3     | 89.4       | 87.3   | 90.5        |
| LSTM               | 91.3    | 92.1     | 90.5       | 91.3   | 93.2        |
| 3D-CNNs            | 94.7    | 95.0     | 94.4       | 94.7   | 96.1        |
| ResNet50+LSTM      | 95.8    | 95.5     | 96.2       | 95.8   | 96.8        |
| VGG16+3D-CNN       | 95.2    | 94.9     | 95.5       | 95.2   | 96.3        |

- Proposed model: 99.2% accuracy, vastly outperforming single-modality baselines
- Fire Videos Dataset (heavy smoke): 98.3% accuracy, 98.4% precision

### Cross-dataset results

| Scenario             | Acc (%) | Prec (%) | Recall (%) | F1 (%) |
|----------------------|---------|----------|------------|--------|
| Clear daylight (NASA)| 99.2    | 99.3     | 99.0       | 99.1   |
| Heavy smoke low light| 98.3    | 98.4     | 98.2       | 98.3   |
| Fire in open field   | 98.5    | 98.6     | 98.3       | 98.3   |

## Applicability to Pyronear

**Medium relevance -- interesting architecture but limited practical applicability.**

- The ViT + 3D-CNN fusion concept is relevant: Pyronear could use a spatial
  branch (current frame analysis) + temporal branch (sequence of frames)
  with feature fusion on the server side
- Transformer encoder after fusion for temporal dependency modeling is a
  modern alternative to LSTM for the second-stage verifier
- Class-weighted BCE loss for handling fire/non-fire imbalance is directly
  applicable

**Significant limitations:**
- Results are extremely high (99.2% accuracy) on small/easy datasets
  (NASA: 999 images; Kaggle Fire Videos) -- likely not representative of
  real-world wildfire smoke detection performance
- No bounding-box detection -- only image/video classification
  (fire/smoke/non-fire). Not suitable as a detector
- No comparison with modern detection architectures (YOLO, DETR, etc.)
- Computational cost not reported in detail; ViT + 3D-CNN + Transformer
  encoder is very heavy
- Small datasets (25,510 total after augmentation) from simple scenarios
- No testing on wildfire-specific datasets (HPWREN, AlertWildfire, FIgLib)
- Paper quality is low; some methodological concerns

## Takeaways for Implementation

1. **Dual-branch spatial+temporal fusion:** The general idea of parallel branches
   (one for single-frame spatial features, one for multi-frame temporal features)
   fused via cross-attention is sound. For Pyronear: ViT/CNN on current frame +
   lightweight temporal model on sequence, fused on server
2. **3D-CNN for temporal features:** Use 3D convolutions (kernel 3x3x3) on
   stacked consecutive frames to capture smoke growth/motion patterns.
   At 30s cadence, stack 5-10 frames for ~2.5-5 min temporal window
3. **Class-weighted BCE loss:** Weights w1 = N/(2*N1) and w0 = N/(2*N0) to
   balance fire/non-fire classes. Simple and effective for imbalanced
   smoke datasets
4. **Temporal window of 10 frames:** Authors use 10 consecutive frames at 1fps.
   At Pyronear's 30s cadence, 10 frames = 5 minutes -- reasonable window
   for smoke verification
5. **Caution with evaluation:** Very high numbers (99%+) on simple benchmarks
   should not be taken at face value. Real-world wildfire detection with
   diverse backgrounds, lighting, and weather is much harder. Evaluate
   on HPWREN/FIgLib/SKLFS for realistic assessment
