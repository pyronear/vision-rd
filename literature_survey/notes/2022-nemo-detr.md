# Nemo: Transformer-Supercharged Wildfire Smoke Benchmark

- **Title:** Nemo: An Open-Source Transformer-Supercharged Benchmark for Fine-Grained Wildfire Smoke Detection
- **Authors:** Yazdi, M.; Seo, J.; Alipour, K.; Chen, R.; Ren, Y.; Sawhney, R.; et al.
- **Year/Venue:** 2022, Remote Sensing 14(16), 3979
- **PDF:** `pdfs/2022-Nemo-Transformer-Wildfire-Smoke-Benchmark-Yazdi-et-al.pdf`

## Key Idea

Nemo is an open-source benchmark combining labeled wildfire smoke datasets from
AlertWildfire/HPWREN cameras with DETR (Detection Transformer) fine-tuned for
single-class smoke detection and fine-grained smoke density (low/mid/high)
detection. The paper highlights domain-specific challenges: extremely small objects
near the horizon, smoke-like confusers (clouds, fog, dust, glare), and extreme
foreground-background class imbalance. Strategies to reduce false alarms include
adding negative samples, collage images, and dummy annotations.

## Architecture

```
┌──────────────────┐
│ Input image       │
│ (3072x2048 typ.)  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ ResNet50 backbone │
│ (CNN features)    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Transformer      │
│ Encoder (6L)     │ self-attention
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Transformer      │
│ Decoder (6L)     │ + object queries
│ N=10 (sc) / 20   │ (N=10 or 20)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ FFN pred. heads  │
│ class + bbox     │ bipartite match
└──────────────────┘
```

## Results

### Single-class smoke detection (best configs)

| Model            | mAP   | AP50  | FA (%) | FPR_N |
|------------------|-------|-------|--------|-------|
| Nemo-DETR-sc     | 40.6  | 77.2  | 96.4   | 21    |
| Nemo-DETR-sce    | 42.3  | 79.0  | 96.8   | 3     |
| Nemo-FRCNN-sc    | 29.3  | 68.4  | 84.4   | 36    |
| Nemo-RNet-sc     | 28.9  | 68.8  | 82.8   | 25    |

- Adding empty images (sce) dropped FPR_N from 21 to **3** false alarms
- DETR achieves highest frame accuracy: 96.8% vs 84.4% (FRCNN)

### Incipient fire detection (95 HPWREN sequences)

| Model            | Mean detect (min) | Detection rate |
|------------------|-------------------|----------------|
| Nemo-DETR-sc     | 3.58 +/- 4.13     | 97.9%          |
| Nemo-FRCNN-sc    | 9.1 +/- 7.5       | 68.4%          |
| Nemo-RNet-sc     | --                 | 54.7%          |

- 80% of fires detected within 5 minutes of ignition
- 67.3% detected within 3 minutes

## Applicability to Pyronear

**High relevance -- same operational setting (fixed cameras, wildfire smoke).**

- DETR's global self-attention is effective for tiny smoke on large backgrounds,
  exactly the scenario with Pyronear's 360-degree cameras
- The benchmark is open-source (GitHub) -- can directly use the Nemo dataset
  and pretrained models for transfer learning
- Incipient fire detection at ~3.5 min mean is impressive and relevant to
  Pyronear's goal of early detection
- FPR reduction strategies (empty images, collages, dummy annotations) are
  directly applicable to Pyronear's training pipeline

**Limitations for Pyronear:**
- DETR is too heavy for RasPi (41M params, ~100ms on GPU); only for server-side
- No temporal modeling -- single-frame detection only
- 21% FPR on challenging negatives (without mitigation) is high; needs temporal
  verification layer on top

## Takeaways for Implementation

1. **Negative sample strategy:** Add challenging empty images (clouds, fog, glare,
   dust, sunset) to training set. This alone dropped FP from 21 to 3 on the
   challenging test set. Pyronear should curate a "hard negative" gallery
2. **DETR as server-side verifier:** Replace or complement the current second-stage
   with a DETR/Deformable-DETR fine-tuned on wildfire smoke. The global
   attention mechanism excels at tiny-object-on-large-background
3. **Smoke density sub-classes:** Consider low/mid/high density annotations for
   prioritizing alerts (high-density = more urgent). 3-class density model
   achievable with re-annotation effort
4. **IoU threshold 0.33:** Smoke's fuzzy boundaries mean standard IoU=0.5 is too
   strict. Using AP@0.33 gives fairer evaluation for smoke detection
5. **Dataset from AlertWildfire/HPWREN:** Open data source for fixed-camera
   wildfire smoke images. Directly usable for pretraining Pyronear models
6. **Collage + dummy annotation hack:** For frameworks that don't support negative
   samples, add 1x1-pixel dummy bbox on empty images. Simple and effective
