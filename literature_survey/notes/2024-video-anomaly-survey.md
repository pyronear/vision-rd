# Video Anomaly Detection in 10 Years: A Survey and Outlook

- **Authors:** Moshira Abdalla, Sajid Javed, Muaz Al Radi, Anwaar Ulhaq, Naoufel Werghi
- **Year/Venue:** 2024, IEEE Access (arXiv:2405.19387v2)
- **PDF:** `/mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/papers/pdfs/2024-Video-Anomaly-Detection-10-Years-Survey.pdf`

## Key Idea

A comprehensive survey of video anomaly detection (VAD) methods over the past decade, covering 50+ methods across supervised, weakly supervised, self-supervised, and unsupervised paradigms. The paper introduces a two-dimensional taxonomy based on (1) learning/supervision schemes and (2) feature extraction approaches. A key contribution is the analysis of Vision-Language Models (VLMs) as emerging feature extractors for VAD, alongside thorough coverage of loss functions, regularization techniques, and anomaly score prediction. Performance on UCF-Crime and ShanghaiTech benchmarks has improved from ~63% to ~98% AUC over the decade.

## Architecture / Content (VAD Taxonomy)

```
VIDEO ANOMALY DETECTION
|
+-- Learning & Supervision Schemes
|   +-- Supervised: frame-level labels, direct classification
|   +-- Weakly Supervised (MIL): video-level labels only
|   |   (most practical -- labels are cheap to obtain)
|   +-- Self-Supervised: pretext tasks (prediction, jigsaw, etc.)
|   +-- Unsupervised:
|       +-- One-Class Classification (learn normal only)
|       +-- Reconstruction-based (AE, VAE, U-Net)
|       +-- Future Frame Prediction (predict next frame, flag deviations)
|
+-- Feature Extraction
    +-- Deep Feature Extractors:
    |   +-- CNNs (2D: spatial; 3D: C3D, I3D for spatiotemporal)
    |   +-- Autoencoders (learn compressed representations of "normal")
    |   +-- GANs (learn normal distribution, flag deviations)
    |   +-- Sequential DL (LSTMs, Vision Transformers)
    |   +-- VLMs (CLIP, etc. -- emerging, textual+visual features)
    |   +-- Hybrid (combine multiple extractors)
    |
    +-- Feature Types:
        +-- Spatial (shapes, textures, colors per frame)
        +-- Temporal (motion, optical flow, object tracking)
        +-- Spatiotemporal (combined -- most effective)
        +-- Textual (captions, VLM descriptions)

VAD Pipeline:
  Video -> Frames -> Feature Extraction -> Learning Method ->
  Loss Function -> Regularization -> Anomaly Score S(V_i) -> Threshold T
  -> Y_hat (normal/anomalous)

  S(V_i) = sum of frame-level scores S(f_i,j)
  Y_hat = 1 if S(V_i) >= T, else 0
```

### Key Benchmark Datasets:
| Dataset | Year | #Videos | Length | Anomalies | Train:Test |
|---|---|---|---|---|---|
| Subway | 2008 | 2 | 1.5 hrs | Wrong direction, no payment | 13:87 |
| UCSD Ped1/2 | 2010 | 70/28 | 5 min each | Bikes, carts, wheelchair | 49:51 / 55:45 |
| CUHK Avenue | 2013 | 37 | 30 min | Strange action, wrong direction | 50:50 |
| ShanghaiTech | 2017 | 437 | 3.67 hrs | Running, biking, loitering | 86:14 |
| UCF-Crime | 2018 | 1,900 | 128 hrs | 13 crime types (abuse, arson...) | 85:15 |
| XD-Violence | 2020 | 4,754 | 214 hrs | Abuse, explosion, fighting | 83:17 |
| NWPU Campus | 2023 | 547 | 16 hrs | Single/group/location anomalies | 56:44 |

### Performance evolution (AUC on UCF-Crime):
- 2017: ~63% (early DL methods)
- 2019-2020: ~82-84% (MIL-based: Sultani, Zhong)
- 2021-2022: ~85-97% (improved representations, GCN, RTFM)
- 2023: ~97-98% (VLM-based methods: CLIP features)

## Results

Key findings from the survey:

- **Weakly supervised (MIL) is the sweet spot:** Video-level labels are cheap to obtain; MIL achieves competitive performance without frame-level annotations
- **3D CNNs (I3D, C3D) are dominant feature extractors:** They capture spatiotemporal patterns essential for detecting anomalies that unfold over time
- **VLMs (CLIP) dramatically improve performance:** The most notable AUC jump in recent years comes from using CLIP features, combining visual and textual understanding
- **Reconstruction-based methods** work well for single-scene static cameras (learn "normal" appearance, flag deviations)
- **Temporal context is crucial:** Frame-level features alone are insufficient; temporal aggregation (LSTM, temporal attention) significantly improves detection
- **Dataset imbalance is a core challenge:** Normal vs. anomalous class ratios are extreme

## Applicability to Pyronear

**Moderately applicable -- the wildfire detection task can be framed as video anomaly detection on a static camera.**

- **Static camera + rare events = anomaly detection:** Pyronear's setup (fixed 360-degree cameras with 30s intervals, watching for rare smoke events against a mostly-static background) is structurally similar to surveillance VAD. Smoke is the "anomaly."
- **Reconstruction-based approaches are promising:**
  - Train an autoencoder on normal (no-smoke) frames from each camera position
  - At inference, reconstruction error flags frames that deviate from the learned "normal" background
  - This is inherently camera-specific and handles the static background well
  - Could serve as an additional FP filter: if reconstruction error is low, the frame is likely normal regardless of what YOLO says
- **Weakly supervised MIL is directly relevant:**
  - Pyronear has video-level labels (fire event timestamps) more easily than frame-level bounding boxes
  - MIL can learn to identify which frames within a fire-event video actually contain smoke
  - This could help with semi-automatic annotation of PyroNear2025-V
- **Future frame prediction:**
  - With 30s intervals, predicting the next frame and measuring prediction error could flag sudden changes (smoke appearance)
  - However, 30s is a long interval for optical flow / motion-based methods

## Takeaways for Implementation

1. **Consider an autoencoder-based anomaly detector as a complementary signal** on the server side:
   - Train per-camera or per-viewpoint autoencoders on normal frames
   - Use reconstruction error as an additional feature for the temporal verification model
   - High reconstruction error + YOLO detection = higher confidence; low reconstruction error + YOLO detection = likely FP (cloud, sunlight)
2. **Weakly supervised MIL for semi-automatic annotation:**
   - Use video-level labels (fire event occurred during this period) to train a MIL classifier that localizes fire frames without manual per-frame labels
   - Could accelerate the PyroNear2026 dataset annotation
3. **Background subtraction / change detection** is a classical VAD technique well-suited to Pyronear's static cameras. Combining frame differencing with YOLO detections could filter transient artifacts.
4. **VLM features (CLIP)** on the server side could provide additional discrimination: extract CLIP embeddings for YOLO-cropped regions and compare with text prompts like "smoke plume" vs. "cloud" vs. "fog."
5. **Temporal aggregation of anomaly scores** mirrors what Pyronear already does with the sequential model. The survey confirms this is the correct approach -- single-frame detection is insufficient, temporal consistency is key.
6. **I3D/C3D features** could replace or supplement the ResNet features fed to the LSTM in Pyronear's sequential model, providing built-in temporal encoding. However, the 30s frame interval makes standard video models (designed for 15-30 fps) less applicable without adaptation.
