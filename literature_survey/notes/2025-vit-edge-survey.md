# Vision Transformers on the Edge: A Comprehensive Survey of Model Compression and Acceleration Strategies

- **Authors:** Shaibal Saha, Lanyu Xu
- **Year/Venue:** 2025 / Preprint submitted to Neurocomputing (arXiv:2503.02891)
- **PDF:** `../pdfs/2025-Vision-Transformers-Edge-Compression-Acceleration-Survey.pdf`

## Key Idea

This survey systematically reviews techniques to deploy Vision Transformers (ViTs) on resource-constrained edge devices (GPUs, FPGAs, ASICs, CPUs). It covers three pillars: (1) Model compression -- pruning (structured/unstructured, token pruning), quantization (PTQ, QAT, mixed-precision), and knowledge distillation (logit-based, feature-based, attention-based); (2) Software tools -- TensorRT, ONNX Runtime, OpenVINO, TVM for inference optimization; (3) Hardware-aware acceleration -- efficient implementations of softmax, GELU, LayerNorm on custom hardware, and SW-HW co-design strategies. The survey highlights that compression alone is insufficient for real-time edge inference and must be combined with hardware-aware optimization.

## Architecture (Taxonomy of Techniques)

```
┌───────────────────────────┐
│ ViT Edge Deployment       │
└─────────────┬─────────────┘
              ▼
┌─────────────┼─────────────┐
▼             ▼             ▼
┌───────┐ ┌────────┐ ┌──────────┐
│Compres│ │Software│ │Hardware  │
│-sion  │ │Tools   │ │Accel.    │
└──┬────┘ └───┬────┘ └────┬─────┘
   │          │           │
   ▼          ▼           ▼
┌──────┐ ┌────────┐ ┌──────────┐
│Pruning│ │TensorRT│ │Softmax / │
│Quant. │ │ONNX RT │ │GELU opt  │
│KD     │ │OpenVINO│ │FPGA/ASIC │
│Token  │ │TVM     │ │SW-HW     │
│pruning│ │        │ │co-design │
└──────┘ └────────┘ └──────────┘
```

## Results (Key Findings)

| Technique | Typical Impact |
|-----------|---------------|
| Structured pruning | 30-50% FLOPs reduction, <2% accuracy drop |
| Token pruning | Up to 60% speedup by dropping uninformative tokens |
| INT8 quantization (PTQ) | 4x memory reduction, ~1% accuracy drop |
| INT4 quantization (QAT) | 8x memory reduction, 2-3% accuracy drop |
| Knowledge distillation | Smaller student matches 95-98% of teacher accuracy |
| TensorRT optimization | 2-4x inference speedup on NVIDIA GPUs |
| FPGA implementations | 10-100x energy efficiency vs GPU |

- ViT-Huge has 632M+ params; edge deployment requires aggressive compression
- Publication trend: ViT compression papers growing from 35 (2020) to 1018 (2024)
- Pruning + quantization + KD can be combined for multiplicative gains
- Token pruning is ViT-specific: drop tokens (patches) that contribute little to the output
- Mixed-precision quantization: keep attention heads at higher precision, MLPs at lower
- SW-HW co-design achieves best results but requires custom hardware

## Applicability to Pyronear

**What transfers:**
- **Knowledge distillation is the highest-impact technique for Pyronear.** Train a large, accurate ViT or transformer-based detector on the server, then distill to YOLOv8-nano for edge. This directly improves edge detection quality.
- **INT8 quantization of YOLOv8 on RaspPi** is standard practice and validated here -- 4x memory reduction with minimal accuracy loss. Already likely used, but could push to INT4 with QAT.
- **Token pruning concept** is adaptable to Pyronear's static background: tokens corresponding to known static regions (mountains, buildings) can be pruned, focusing compute on sky/treeline regions where smoke would appear.
- **TensorRT / ONNX Runtime** are directly applicable inference optimization tools for both edge (TensorRT on Jetson, ONNX on RaspPi) and server deployment.
- **Structured pruning** of the YOLOv8 backbone can further reduce edge compute while maintaining detection quality on the specific task (smoke only, not 80 COCO classes).

**What doesn't transfer:**
- Much of the survey focuses on ViT-specific optimizations (attention head pruning, token pruning). YOLOv8 on the edge is CNN-based, not ViT-based.
- FPGA/ASIC discussion is relevant for future custom hardware but not for current RaspPi deployment.
- Some techniques require retraining or QAT, which needs the full training pipeline.

## Takeaways for Implementation

1. **Knowledge distillation pipeline:** Train a large fire detection model (e.g., YOLOv8-L or ViT-based detector) on server, distill to YOLOv8-nano for RaspPi. Use feature-based KD (not just logit matching) for better transfer of spatial awareness.
2. **Quantize YOLOv8 to INT8 (or INT4 with QAT)** for RaspPi deployment. Use TensorRT if on Jetson, or ONNX Runtime / TFLite for Raspberry Pi.
3. **Task-specific pruning:** Since YOLOv8 is pretrained on 80 COCO classes but only needs to detect smoke, prune channels/filters that are irrelevant to the fire domain. This can reduce model size by 30-50%.
4. **Token/region pruning for static cameras:** Pre-compute a mask of "interesting regions" (sky, treeline) for each camera position. Only run detection on those regions, skipping static foreground (ground, buildings). This is a spatial analogy to token pruning.
5. **Software stack:** Use ONNX Runtime with INT8 on RaspPi. For server GPU, use TensorRT with FP16 for the temporal verification model.
6. **Combine compression techniques:** Pruning + quantization + KD together can achieve 4-10x total speedup. Apply in sequence: prune first, then distill, then quantize.
