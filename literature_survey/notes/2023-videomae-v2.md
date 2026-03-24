# VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking

- **Authors:** Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, Yu Qiao
- **Year/Venue:** 2023 / CVPR (arXiv:2303.16727)
- **PDF:** `../pdfs/2023-VideoMAE-V2-Scaling-Dual-Masking-Wang-et-al.pdf`

## Key Idea

VideoMAE V2 scales the video masked autoencoder framework to billion-parameter models (ViT-g, 1B params) through a dual masking strategy: high-ratio tube masking in the encoder (90%) plus additional masking in the decoder, reducing pre-training cost by ~1/3 with no performance loss. It also introduces progressive training: unsupervised pre-training on a 1.35M-clip unlabeled hybrid dataset, then supervised post-pre-training on a 0.66M-clip labeled hybrid dataset, then task-specific fine-tuning. This produces a general video foundation model that achieves SOTA across action recognition, spatial action detection, and temporal action detection.

## Architecture

```
┌───────────────────────────┐
│ Video clip ▶ Cube embed    │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Encoder masking (90%)     │
│ Tube mask ▶ keep 10% toks │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ ViT Encoder (ViT-g 1B)   │
│ vanilla self-attention    │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Decoder masking           │
│ running cell mask on      │
│ combined (enc + mask) toks│
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Lightweight ViT Decoder   │
│ reconstruct masked pixels │
└───────────────────────────┘
   MSE loss on masked cubes
```

## Results

| Task | Dataset | Metric | Score |
|------|---------|--------|-------|
| Action classification | Kinetics-400 | Top-1 | **90.0%** |
| Action classification | Kinetics-600 | Top-1 | **89.9%** |
| Action classification | SSv1 | Top-1 | **68.7%** |
| Action classification | SSv2 | Top-1 | **77.0%** |
| Spatial action det. | AVA | mAP | SOTA |
| Temporal action det. | THUMOS14 | mAP | SOTA |

- First billion-parameter video ViT successfully trained
- Dual masking reduces pre-training time by ~33%
- Progressive training (unsupervised ▶ supervised ▶ fine-tune) prevents overfitting
- Data diversity (1.35M clips from Movies, YouTube, Instagram, etc.) is crucial

## Applicability to Pyronear

**What transfers:**
- **Self-supervised pre-training paradigm.** Pyronear has abundant unlabeled camera footage. VideoMAE-style masked autoencoding can pre-train a temporal model on this data without labels, learning what "normal" landscape patterns look like. Anomalies (smoke) would then be easier to detect.
- **Progressive training strategy** is directly applicable: (1) self-supervised pre-train on unlabeled Pyronear footage, (2) supervised post-pre-train on a mixed fire/no-fire dataset, (3) fine-tune per deployment site.
- **Cube embedding** captures local spatiotemporal structure -- useful for modeling smoke texture evolution.
- **The dual masking idea** (aggressive masking for efficiency) could make pre-training affordable on moderate GPU budgets.

**What doesn't transfer:**
- **ViT-g at 1B parameters** is far too large for deployment, even on the server. But smaller variants (ViT-B at 87M, ViT-S) are available.
- Pre-training requires significant GPU resources (64 A100s for 2 weeks for ViT-g). ViT-B/ViT-S variants are more practical.
- Designed for dense video (16 frames at ~6fps). Pyronear's 30s interval is very sparse.
- Detection tasks require additional heads (not part of the pre-training framework itself).

## Takeaways for Implementation

1. **Self-supervised pre-training on Pyronear footage** using masked autoencoding (VideoMAE approach) with a ViT-S or ViT-B backbone. Use unlabeled camera streams to learn landscape/temporal representations.
2. **Use progressive fine-tuning:** Pre-train on unlabeled Pyronear video, then fine-tune on labeled fire/no-fire clips. This avoids the limited-labeled-data problem.
3. **Dual masking at 90% encoder + decoder masking** makes pre-training feasible on smaller GPU clusters (reduces memory and compute by ~3x).
4. **ViT-B (87M params) is the practical sweet spot** for server-side deployment, balancing capacity and compute.
5. **Data diversity matters:** Collect footage across seasons, weather, day/night, and different camera sites for robust pre-training.
6. **Downstream integration:** After pre-training, the encoder can serve as the backbone for a temporal classification or detection head on the server.
