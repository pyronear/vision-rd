# Generalization-Enhanced Few-Shot Object Detection in Remote Sensing

- **Authors:** Hui Lin, Nan Li, Pengjuan Yao, Kexin Dong, Yuhan Guo, Danfeng Hong, Ying Zhang, Congcong Wen
- **Year/Venue:** 2025 / IEEE (arXiv:2501.02474)
- **PDF:** `../pdfs/2025-Few-Shot-Object-Detection-Remote-Sensing-Generalization.pdf`

## Key Idea

GE-FSOD (Generalization-Enhanced Few-Shot Object Detection) addresses FSOD in remote sensing imagery, where objects have large scale variations, complex backgrounds, and diverse appearances. It introduces three innovations: (1) CFPAN -- a Cross-Level Fusion Pyramid Attention Network replacing FPN, using dual attention (channel + spatial via CBAM) and cross-level feature fusion for better multi-scale representation; (2) MRRPN -- a Multi-Stage Refinement Region Proposal Network replacing standard RPN, iteratively refining proposals through dilated+deformable convolutions across multiple stages; and (3) GCL -- a Generalized Classification Loss with placeholder nodes and regularization to improve few-shot classification.

## Architecture

```
┌───────────────────────────┐
│ Backbone (e.g. ResNet)    │
│ ▶ C2, C3, C4, C5          │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ CFPAN (neck):             │
│  CBAM on C5 (chan+spatial) │
│  Cross-level fusion:      │
│  P5 ▶ upsample+fuse ▶ P4  │
│  P4 ▶ upsample+fuse ▶ P3  │
│  P3 ▶ upsample+fuse ▶ P2  │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ MRRPN (head):             │
│  Stage 1: DilConv+Conv    │
│  Stage 2: DilConv+Conv    │
│  ...                      │
│  Stage N: DefConv+Conv    │
│  ▶ refined proposals       │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ ROI Pooling ▶ FC layers    │
│ Classification (GCL loss) │
│ BBox regression           │
└───────────────────────────┘
```

## Results

- Evaluated on DIOR and NWPU VHR-10 remote sensing datasets
- SOTA performance in few-shot detection on remote sensing imagery
- CFPAN improves multi-scale detection vs standard FPN
- MRRPN's multi-stage refinement produces more accurate proposals
- GCL loss with placeholder nodes improves generalization on novel classes
- Two-stage training: base training (full model), then fine-tuning (freeze backbone, tune CFPAN + MRRPN)

## Applicability to Pyronear

**What transfers:**
- **Multi-scale detection is critical for Pyronear.** Smoke appears at vastly different scales depending on distance (distant plume = tiny object, nearby = large). CFPAN's cross-level fusion with attention is directly relevant.
- **CBAM (Channel + Spatial Attention) on FPN** is a lightweight, plug-in improvement applicable to YOLOv8's neck. It helps the model focus on relevant channels (smoke-related features) and spatial regions.
- **Remote sensing shares characteristics with Pyronear:** fixed camera viewpoint, large images, diverse backgrounds (forests, mountains, urban edges), small targets at distance, and lighting/weather variations.
- **Few-shot fine-tuning protocol** (freeze backbone, tune neck+head) is the right approach for site-specific adaptation with limited labeled fire examples.
- **Multi-stage proposal refinement** concept could reduce false positives by progressively filtering candidate regions.

**What doesn't transfer:**
- Two-stage detector (Faster R-CNN based) is slower than YOLOv8 and not ideal for edge deployment.
- Remote sensing imagery (satellite/UAV top-down) has different characteristics from Pyronear's ground-level horizontal views.
- GCL loss is designed for multi-class few-shot scenarios; Pyronear has primarily a single class (smoke).
- The method does not consider temporal information.

## Takeaways for Implementation

1. **Add CBAM attention to YOLOv8's neck (PAN).** This is a lightweight modification (few extra params) that can improve multi-scale smoke detection by focusing on relevant channels and spatial regions.
2. **Cross-level feature fusion** beyond standard PAN: ensure that high-level semantic features (smoke vs cloud discrimination) flow to low-level detection layers, and fine spatial details flow upward.
3. **For new-site deployment:** Freeze YOLOv8 backbone, fine-tune neck + detection head on 5-10 annotated smoke examples from the new site. This follows the paper's proven protocol.
4. **Dilated convolutions in the detection head** can increase receptive field without adding parameters -- useful for detecting smoke plumes that span large image areas.
5. **Consider multi-stage refinement** as a post-processing step on the server: initial YOLOv8 proposals refined through a second lightweight network to reduce FP.
