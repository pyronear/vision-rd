# FIgLib & SmokeyNet: Dataset and Deep Learning Model for Real-Time Wildland Fire Smoke Detection

**Authors:** Dewangan et al.
**Year/Venue:** 2022, Remote Sensing (MDPI)
**PDF:** `pdfs/2022-SmokeyNet-FIgLib-Spatiotemporal-Smoke-Detection-Dewangan-et-al.pdf`

## Key Idea

A three-component spatiotemporal model (CNN + LSTM + ViT) that processes two consecutive camera frames tiled into overlapping patches. Introduces FIgLib, a large-scale wildfire smoke dataset (~25k images) where even humans struggle to detect smoke from single frames, proving the necessity of temporal information.

## Architecture

```
 Frame(t-1)    Frame(t)
     │              │
     ▼              ▼
 ┌────────┐    ┌────────┐
 │ Tile   │    │ Tile   │    45 overlapping 224×224 patches
 │ (45×)  │    │ (45×)  │    per frame (20px overlap)
 └───┬────┘    └───┬────┘
     │              │
     ▼              ▼
 ┌────────┐    ┌────────┐
 │ CNN    │    │ CNN    │    ResNet50, ImageNet pretrained
 │ per    │    │ per    │    extracts features per tile
 │ tile   │    │ tile   │
 └───┬────┘    └───┬────┘
     │              │
     └──────┬───────┘
            │
            ▼
     ┌────────────┐
     │   LSTM     │     Per-tile temporal fusion
     │ (tile_t-1, │     Combines same tile across
     │  tile_t)   │     2 frames
     └─────┬──────┘
           │
           ▼ 45 temporally-fused tile embeddings
     ┌────────────┐
     │   ViT      │     Spatial attention across
     │ (all 45    │     all tiles + CLS token
     │  tiles)    │
     └─────┬──────┘
           │ CLS token
           ▼
     ┌────────────┐
     │ Image Head │     3× FC + ReLU + Sigmoid
     └─────┬──────┘
           │
           ▼
     smoke / no smoke
```

**Intermediate supervision:** Each component (CNN, LSTM, ViT) also has its own per-tile classification head for auxiliary losses.

**Loss:** `L = BCE_image + Σ(BCE_CNN_i + BCE_LSTM_i + BCE_ViT_i)` with 40x positive weight for tiles, 5x for image level.

## Temporal Modeling

- Uses **only 2 frames** (current + previous)
- Time gap: ~0.6 minutes between frames in FIgLib dataset
- LSTM fuses temporal info per tile independently
- ViT then integrates spatial info across tiles

## Results

| Model                         | Params | Time    | Accuracy | F1    | Precision | Recall | TTD (min) |
|-------------------------------|--------|---------|----------|-------|-----------|--------|-----------|
| ResNet34 (1 frame)            | 23.3M  | 29.7ms  | 79.40%   | 78.90 | 81.62     | 76.58  | 2.81      |
| ResNet34 + LSTM               | 38.9M  | 53.3ms  | 79.35%   | 79.21 | 82.00     | 76.74  | 2.64      |
| ResNet34 + ViT (1 frame)      | 40.3M  | 30.8ms  | 82.53%   | 81.30 | 85.58     | 75.19  | 2.95      |
| **ResNet34 + LSTM + ViT (2f)** | 56.9M  | 51.6ms  | **83.49%** | **82.59** | **89.84** | 76.45 | 3.12    |

- CNN alone = 79.4% (poor)
- Adding ViT spatial attention = +3.1% (spatial context across tiles matters a lot)
- Adding LSTM temporal = +1% on top (temporal info helps, but modestly with only 2 frames)
- Average time-to-detection: 3.12 minutes

## Applicability to Pyronear

**High relevance.** This is architecturally the closest to what Pyronear needs.

**What maps directly:**
- The **tiling approach** is relevant for Pyronear's 360° images which are large. YOLO bbox proposals serve the same role as tiles -- defining regions of interest.
- The **two-frame temporal fusion** concept works with 30s intervals. At 30s with a fixed camera, the scene changes are exactly what you want to detect (smoke appearing/growing).
- The **intermediate supervision** trick is useful: train the CNN head to detect smoke per-frame, the temporal head to confirm using multi-frame context.

**What to change:**
- Replace LSTM with a **Transformer** for temporal fusion -- allows more than 2 frames and handles variable-length history.
- Use **YOLO bbox crops** instead of fixed tiling. YOLO already identifies the regions of interest.
- Add a **background subtraction channel** as input alongside the raw frame crops.

**Key insight from ablation:** LSTM alone doesn't help without ViT (79.4% → 79.35%). But LSTM + ViT together push to 83.5%. This suggests that temporal and spatial context are **synergistic** -- you need both.

## Takeaways for Implementation

1. **Use YOLO crops as "tiles"**: instead of tiling the full image, crop around YOLO detections and process them through a CNN backbone.
2. **Two frames is a minimum**: even 2 frames helped (+1% accuracy, +4pp precision). With 30s intervals and more frames (4-8), the temporal signal should be stronger.
3. **Intermediate supervision**: add auxiliary classification heads on each component to improve training.
4. **Positive weighting**: heavy positive weighting (40x) needed due to class imbalance -- most tiles/frames have no smoke.
5. **89.84% precision with 76.45% recall** -- the model is conservative. Pyronear wants high recall; threshold tuning or loss weighting can adjust this tradeoff.
