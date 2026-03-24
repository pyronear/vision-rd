# ViViT: A Video Vision Transformer

- **Authors:** Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lucic, Cordelia Schmid
- **Year/Venue:** 2021 / ICCV (arXiv:2103.15691)
- **PDF:** `../pdfs/2021-ViViT-Video-Vision-Transformer-Arnab-et-al.pdf`

## Key Idea

ViViT proposes four pure-transformer architectures for video classification that factorise spatial and temporal dimensions to handle the large number of spatio-temporal tokens. The key variants are: (1) joint spatio-temporal attention (unfactorised), (2) factorised encoder (spatial encoder then temporal encoder in series), (3) factorised self-attention (spatial-then-temporal attention within each block, like TimeSformer's T+S), and (4) factorised dot-product attention (splitting heads into spatial and temporal groups). The paper also introduces tubelet embedding (3D patch tokenization) and effective strategies to initialize from pretrained image ViTs.

## Architecture

```
┌───────────────────────────┐
│ Video: T x H x W x 3     │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Tokenization:             │
│  Uniform frame sampling   │
│  OR Tubelet embedding     │
│  (3D patches: t x h x w) │
└─────────────┬─────────────┘
              ▼
  Model 2: Factorised Encoder
┌───────────────────────────┐
│ Spatial Encoder (Ls lyrs) │
│  per-frame ViT ▶ cls tok  │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Temporal Encoder (Lt lyrs)│
│  across frame cls tokens  │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ MLP head ▶ class          │
└───────────────────────────┘
```

## Results

| Model (ViT-B backbone) | K400 | EK | FLOPs | Params | Runtime |
|------------------------|------|----|-------|--------|---------|
| Model 1: Spatio-temporal | 80.0 | 43.1 | 455G | 88.9M | 58.9ms |
| Model 2: Fact. encoder | 78.8 | 43.7 | 284G | 115.1M | 17.4ms |
| Model 3: Fact. self-attn | 77.4 | 39.1 | 372G | 117.3M | 31.7ms |
| Model 4: Fact. dot-prod | 76.3 | 39.5 | 277G | 88.9M | 22.9ms |

- Tubelet embedding with "central frame" init: 79.2 top-1 (vs 78.5 uniform sampling)
- Factorised encoder (Model 2) best speed/accuracy: 3.4x faster than Model 1
- Temporal transformer depth Lt=4 is sufficient (75.8 at Lt=0 vs 78.8 at Lt=4)
- SOTA on K400, K600, SSv2, Epic Kitchens, Moments in Time

## Applicability to Pyronear

**What transfers:**
- **Factorised encoder (Model 2) is ideal for Pyronear's two-stage pipeline.** The spatial encoder is essentially a per-frame ViT (could reuse ViT features from the edge detection), and the lightweight temporal encoder (Lt=4 layers) aggregates across frames on the server.
- **Tubelet embedding** could capture short-term temporal changes in smoke appearance (opacity, growth) if frames were closer in time.
- **Initialization from pretrained ViT** is well-studied here with three strategies (filter inflation, central frame). Directly applicable when starting from ImageNet-pretrained ViT.
- Model 2 runs at 17.4ms inference -- fast enough for server real-time verification.

**What doesn't transfer:**
- 30s frame interval is too sparse for tubelet embedding (designed for temporally dense video at ~12.5fps).
- Classification-only architecture; no spatial localization/detection head.
- The best accuracy model (Model 1, unfactorised) is too expensive for practical use.
- Trained/evaluated on action recognition, not event detection or anomaly detection.

## Takeaways for Implementation

1. **Adopt Model 2 (Factorised Encoder) for server verifier.** Run a spatial encoder (or reuse YOLOv8 backbone features) per frame, then feed per-frame representations into a small temporal transformer (4 layers) for smoke confirmation.
2. **Lt=4 is sufficient** for the temporal transformer -- no need for deep temporal models. This keeps the server overhead minimal.
3. **Central frame initialization** for tubelet/temporal weights is the best strategy when starting from image ViT pretrained weights.
4. **Frame-level representations can be pooled simply** (global average pooling from spatial encoder works nearly as well as cls token), enabling flexible integration with existing backbones.
5. **Spatial encoder can be shared/frozen** across frames for efficiency -- only the temporal encoder needs to be trained for the fire verification task.
