# Small Object Detection: A Comprehensive Survey on Challenges, Techniques and Real-World Applications

- **Authors:** Mahya Nikouei, Bita Baroutian, Shahabedin Nabavi, Fateme Taraghi, Atefe Aghaei, Ayoob Sajedi, Mohsen Ebrahimi Moghaddam
- **Year/Venue:** 2025 / Submitted to journal (survey of Q1 journal articles 2024-2025)
- **PDF:** `../pdfs/2025-Small-Object-Detection-Comprehensive-Survey.pdf`

## Key Idea

This survey comprehensively reviews small object detection (SOD) techniques from 2024-2025 research, covering definitions, challenges, methods, datasets, and applications. Small objects are defined as <32x32 pixels (COCO) or <1% of image area. Key challenges include limited appearance information, background interference, scale variation, feature loss in deep networks, and computational cost. The survey categorizes solutions into: (1) multi-scale feature extraction (FPN variants, attention mechanisms), (2) super-resolution approaches, (3) data augmentation and synthetic data, (4) transfer learning and knowledge distillation, and (5) lightweight architectures for edge deployment.

## Architecture (Taxonomy of SOD Methods)

```
┌───────────────────────────┐
│   Small Object Detection  │
│        Methods            │
└─────────────┬─────────────┘
              ▼
  ┌───────────┼───────────┐
  ▼           ▼           ▼
┌──────┐ ┌────────┐ ┌────────┐
│Multi-│ │Super-  │ │Data    │
│Scale │ │Resol.  │ │Augment.│
│FPN++ │ │(SR)    │ │Synth.  │
└──┬───┘ └───┬────┘ └───┬────┘
   │         │          │
   ▼         ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐
│Attn  │ │Transfer│ │Lightw. │
│Mech. │ │Learnng │ │Nets /  │
│(CBAM)│ │+ KD    │ │Edge    │
└──────┘ └────────┘ └────────┘
```

## Results (Key Findings from Surveyed Papers)

- COCO defines small objects as <32x32 px; UAV/satellite uses <20x20 px
- Main challenges ranked: (1) low resolution / few pixels, (2) background interference / low SNR, (3) feature loss in deep layers, (4) scale variation
- FPN and its variants remain the dominant approach for multi-scale detection
- Attention mechanisms (CBAM, self-attention) significantly improve SOD
- Super-resolution pre-processing improves detection but adds latency
- Knowledge distillation enables deploying SOD on edge devices
- Key applications: surveillance, UAV, traffic monitoring, remote sensing, medical imaging
- Gap identified: real-time SOD on edge remains underexplored

## Applicability to Pyronear

**What transfers:**
- **Pyronear's core challenge IS small object detection.** Distant smoke plumes are tiny in wide-angle 360-degree camera images. All SOD insights apply directly.
- **Multi-scale feature extraction** via enhanced FPN/PAN is essential. YOLOv8 already uses PAN but could benefit from attention-enhanced variants (BiFPN, CFPAN).
- **Attention mechanisms (CBAM, spatial attention)** help focus on subtle smoke features against complex forest/sky backgrounds -- directly addresses Pyronear's FP problem.
- **Knowledge distillation** from a large server model to the edge YOLOv8 can improve edge detection quality without increasing RaspPi compute.
- **Data augmentation strategies** (copy-paste of smoke objects, synthetic smoke generation) address the scarcity of labeled fire data.
- **Super-resolution as preprocessing:** Could upscale distant smoke regions before detection, but must balance against latency on RaspPi.

**What doesn't transfer:**
- Many surveyed methods target dense small objects (pedestrians, vehicles in crowds). Pyronear's small objects (smoke) are sparse and amorphous.
- Anchor-box challenges for small objects are less relevant with YOLOv8's anchor-free design.
- Some methods require high-resolution input processing, which is expensive on edge.

## Takeaways for Implementation

1. **Enhance YOLOv8's PAN with attention:** Add channel + spatial attention (CBAM-style) to the feature pyramid neck. This is the single most impactful change for small smoke detection per the survey.
2. **Feature fusion matters more than depth** for SOD. Ensure shallow, high-resolution features reach the detection head (skip connections, additional P2-level detection).
3. **Knowledge distillation pipeline:** Train a large, accurate model on the server, then distill to a smaller YOLOv8-nano for edge. This bridges the accuracy gap without increasing edge compute.
4. **Data augmentation for rare small smoke:** Copy-paste smoke patches at various scales onto clean background images. Use GANs or diffusion models to generate synthetic smoke at different distances/sizes.
5. **Add a P2 detection layer** (1/4 resolution) to YOLOv8 for detecting very small/distant smoke. Standard YOLOv8 starts at P3 (1/8), missing the smallest targets.
6. **Tiled inference on edge:** Split high-resolution panoramic images into overlapping tiles, run detection per tile, and merge. This effectively increases relative object size.
7. **Background subtraction as preprocessing:** Since Pyronear cameras are static, background subtraction can highlight changes (new smoke) and suppress static false positives, improving effective SNR for small objects.
