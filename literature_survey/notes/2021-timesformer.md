# TimeSformer: Is Space-Time Attention All You Need for Video Understanding?

- **Authors:** Gedas Bertasius, Heng Wang, Lorenzo Torresani
- **Year/Venue:** 2021 / ICML (arXiv:2102.05095)
- **PDF:** `../pdfs/2021-TimeSformer-Space-Time-Attention-Video-Bertasius-et-al.pdf`

## Key Idea

TimeSformer is a convolution-free video classification architecture built exclusively on self-attention, extending ViT from images to video. It compares five space-time attention schemes and finds that "Divided Space-Time Attention" (temporal attention then spatial attention applied separately within each block) gives the best accuracy-efficiency trade-off. The model achieves SOTA on Kinetics-400/600 while being faster to train than 3D CNNs and scaling gracefully to longer clips.

## Architecture

```
┌─────────────────────────────┐
│  Video clip: F x H x W x 3 │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ Patch embed (P=16) + PosEmb │
│ N = HW/P^2 patches x F frm │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   L x Divided ST Block:    │
│  ┌────────────────────┐     │
│  │ Temporal Attention  │     │
│  │ (same patch, all F) │     │
│  └────────┬───────────┘     │
│           ▼                 │
│  ┌────────────────────┐     │
│  │ Spatial Attention   │     │
│  │ (same frame, all N) │     │
│  └────────┬───────────┘     │
│           ▼                 │
│  ┌────────────────────┐     │
│  │       MLP          │     │
│  └────────┬───────────┘     │
└───────────┼─────────────────┘
            ▼
┌─────────────────────────────┐
│  CLS token ▶ MLP head       │
└─────────────────────────────┘
```

## Results

| Model | Pretrain | K400 Acc | TFLOPs | Params |
|-------|----------|----------|--------|--------|
| TimeSformer | IN-21K | **78.0** | 0.59 | 121.4M |
| SlowFast 8x8 R50 | IN-1K | 75.6 | 1.97 | 34.6M |
| I3D 8x8 R50 | IN-1K | 73.4 | 1.11 | 28.0M |
| TimeSformer-HR (448px) | IN-21K | 79.7 | -- | 121.4M |
| TimeSformer-L (96 frm) | IN-21K | 80.7 | -- | 121.4M |

- Divided ST attention: K400 78.0, SSv2 59.5 (best among 5 schemes)
- Training cost: 416 V100-hours vs 3840 for SlowFast
- Joint ST attention OOMs at 448px or 32+ frames

## Applicability to Pyronear

**What transfers:**
- Divided space-time attention is directly applicable to Pyronear's server-side temporal verification. With 30s between frames, temporal attention across a window of N frames (e.g., 4-8 frames = 2-4 min) can model smoke evolution patterns.
- The factorized design (temporal then spatial) is efficient -- O(N+F+2) vs O(NF+1) comparisons per patch. This is critical for the server GPU budget.
- ImageNet pretraining strategy works well; Pyronear can leverage existing ViT weights.
- Scales to long videos (96 frames tested), relevant for multi-minute fire confirmation.

**What doesn't transfer:**
- Designed for clip-level classification (action recognition), not detection/localization. Pyronear needs spatial localization of smoke, not just "is there fire in this clip."
- 121M params is too large for RaspPi edge inference. Only suitable for server-side.
- Assumes densely sampled frames (1/32 rate = ~1fps). Pyronear's 30s gap is much sparser -- positional embeddings and temporal attention patterns may need re-tuning.
- No detection head; would need to be combined with a detection framework.

## Takeaways for Implementation

1. **Server temporal verifier:** Use divided space-time attention as the temporal backbone on the server. Feed 4-8 consecutive frames (2-4 min window) as a clip to classify smoke/no-smoke, reducing YOLOv8 false positives.
2. **Factored attention is key:** Always prefer divided (T+S) over joint (ST) attention -- it is both more accurate and scales to longer sequences without OOM.
3. **Pretrain from ImageNet-21K ViT weights:** Initialize spatial attention from ViT, temporal attention from zeros (acts as residual at init). This is efficient and well-validated.
4. **Consider temporal stride:** With 30s frame intervals, a window of 8 frames = 4 minutes. This is a natural fire-confirmation window. Positional embeddings should encode absolute time, not frame index.
5. **Hybrid approach:** Use YOLOv8 on edge for spatial detection, then crop the detected region across T frames and feed to a small TimeSformer on the server for temporal confirmation.
