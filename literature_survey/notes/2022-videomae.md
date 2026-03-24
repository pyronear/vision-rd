# VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training

**Authors:** Tong et al.
**Year/Venue:** 2022, NeurIPS
**PDF:** `pdfs/2022-VideoMAE-Masked-Autoencoders-Video-Pre-Training-Tong-et-al.pdf`

## Key Idea

Self-supervised video pre-training via masked autoencoding with extremely high masking ratios (90-95%). Exploits temporal redundancy in video: because adjacent frames are nearly identical, you can mask most of the spatiotemporal volume and still reconstruct it. Uses tube masking (same spatial mask across all frames) to prevent the model from "cheating" by copying from adjacent frames. Data-efficient: works with as few as 3.5k training videos.

## Architecture

```
 Input: T=16 downsampled frames (stride τ=2 or 4)
 ┌─────────────────────────────────────────┐
 │ ■ ■ □ ■ □ □ ■ □ □ □ ■ □ □ □ □ ■       │
 │ ■ ■ □ ■ □ □ ■ □ □ □ ■ □ □ □ □ ■       │  tube masking
 │ ■ ■ □ ■ □ □ ■ □ □ □ ■ □ □ □ □ ■       │  90-95% masked
 │ ...across T/2=8 temporal positions...   │  same mask all
 │ ■=visible □=masked                     │  frames
 └──────────────┬──────────────────────────┘
                │ ~150 visible tokens (out of 1568)
                ▼
 ┌──────────────────────┐
 │ Encoder              │
 │ Vanilla ViT          │  only processes visible tokens
 │ joint space-time     │  (very efficient: 10% of tokens)
 │ self-attention       │
 └──────────┬───────────┘
            │ encoder output + learnable mask tokens
            ▼
 ┌──────────────────────┐
 │ Decoder              │
 │ 4-block Transformer  │  reconstructs full video
 │ (shallow, half-width)│  from visible tokens
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │ MSE Loss             │  on normalized pixel values
 │ (masked tokens only) │  of masked cubes
 └──────────────────────┘


 Cube embedding: 2×16×16 spatiotemporal cube → 1 token
 For T=16, H=W=224: (16/2)×(224/16)×(224/16) = 8×14×14 = 1568 tokens
 At 90% masking: ~150 visible tokens enter encoder
```

## Why Tube Masking?

```
 Without tube masking (random/frame masking):
 ┌─────────┐  ┌─────────┐  ┌─────────┐
 │ ■ □ ■ □ │  │ □ ■ □ ■ │  │ ■ □ □ ■ │  different masks
 │ □ ■ □ ■ │  │ ■ □ ■ □ │  │ □ ■ ■ □ │  per frame
 └─────────┘  └─────────┘  └─────────┘
    → Model can copy masked patch from same
      location in adjacent frame = CHEATING
      (learns temporal correspondence, not semantics)

 With tube masking:
 ┌─────────┐  ┌─────────┐  ┌─────────┐
 │ ■ □ ■ □ │  │ ■ □ ■ □ │  │ ■ □ ■ □ │  SAME mask
 │ □ ■ □ ■ │  │ □ ■ □ ■ │  │ □ ■ □ ■ │  all frames
 └─────────┘  └─────────┘  └─────────┘
    → Masked location is masked EVERYWHERE
      in time → must reason about content
      to reconstruct
```

## Three Key Findings

**1. 90% masking is optimal** (vs 75% for images):

| Mask % | SSV2  | K400  |
|:------:|:-----:|:-----:|
| 50%    | 65.2  | 78.3  |
| 75%    | 68.0  | 79.0  |
| 90%    | 69.6  | 80.0  |
| 95%    | 69.3  | 79.8  |

Video has more redundancy than images → can mask more.

**2. Data-efficient** (works with 3.5k videos):

| Dataset | Videos | Scratch | MoCo v3 | VideoMAE |
|---------|:------:|:-------:|:-------:|:--------:|
| K400    | 240k   | 68.8    | 74.2    | **80.0** |
| SSV2    | 169k   | 32.6    | 54.2    | **69.6** |
| UCF101  | 9.5k   | 51.4    | 81.7    | **91.3** |
| HMDB51  | 3.5k   | 18.0    | 39.2    | **62.6** |

Gap widens as data shrinks. At 3.5k videos: 62.6% vs 39.2% (MoCo) vs 18.0% (scratch).

**3. Domain shift matters:**
Pre-training on same dataset > transfer. K400→SSV2 drops performance vs SSV2→SSV2. Data quality > quantity: 42k videos can match 240k performance.

## Training Details

- Pre-training: 800 epochs, AdamW lr=1.5e-4, batch 1024, 40 warmup epochs
- Pre-training time: 19.5h on 64 V100s (3.2x faster than MoCo v3)
- Model sizes: ViT-S (22M), ViT-B (87M), ViT-L (305M), ViT-H (633M)
- Fine-tuning: lr=1e-3 (SSV2), lr=5e-4 (K400), 40-150 epochs

## Applicability to Pyronear

**Medium relevance -- useful as a pre-training recipe, not as the temporal model itself.**

**What's useful:**
1. **Self-supervised pre-training on unlabeled data.** Pyronear has tons of unlabeled camera footage. VideoMAE could pre-train a strong ViT backbone on this data without any labels.
2. **Data efficiency.** With limited labeled smoke data, having a strong pre-trained backbone is critical. VideoMAE at 3.5k videos suggests pre-training on a few thousand Pyronear clips could work.
3. **The ViT encoder is a strong feature extractor** once pre-trained. Strip the decoder, use the encoder as the CNN replacement in a temporal pipeline.

**What doesn't apply:**
1. **Temporal masking/reconstruction assumes dense video.** The tube masking works because adjacent frames are redundant. At 30s intervals, frames are essentially independent -- no temporal redundancy to exploit. The pre-training objective won't learn temporal features from sparse data.
2. **Joint space-time attention** is overkill for sparse frames. With 4-8 frames at 30s intervals, a simple cross-frame Transformer would suffice.

**Recommended use for Pyronear:**
```
 Pre-training phase (dense video, unlabeled):
 Outdoor/camera footage → VideoMAE → pre-trained ViT encoder

 Deployment phase (sparse 30s frames):
 YOLO crop → ViT encoder (frozen/fine-tuned) → features
                                                    │
 YOLO crop (30s ago) → ViT encoder → features ─────┤
                                                    │
 YOLO crop (60s ago) → ViT encoder → features ─────┤
                                                    ▼
                                          Temporal Transformer
                                          (lightweight, trained
                                           on labeled smoke data)
                                                    │
                                                    ▼
                                          smoke / false positive
```

## Takeaways for Implementation

1. **Pre-train VideoMAE on Pyronear's own camera footage** -- even unlabeled. The domain-specific representations will be stronger than ImageNet/Kinetics pre-trained features.
2. **Use the encoder only** as a feature backbone. Add a separate lightweight temporal model on top.
3. **ViT-S (22M params)** may be sufficient and is much faster to pre-train than ViT-B (87M).
4. **Data quality > quantity** for pre-training. A curated set of diverse Pyronear scenes (different cameras, weather, times of day) is more valuable than massive uncurated data.
