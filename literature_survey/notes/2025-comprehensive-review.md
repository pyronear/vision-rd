# Early Fire and Smoke Detection Using Deep Learning: A Comprehensive Review of Models, Datasets, and Challenges

- **Authors:** Abdussalam Elhanashi, Siham Essahraui, Pierpaolo Dini, Sergio Saponara
- **Year/Venue:** 2025, Applied Sciences 15, 10255 (MDPI)
- **PDF:** `/mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/papers/pdfs/2025-Early-Fire-Smoke-Detection-Comprehensive-Review.pdf`

## Key Idea

A comprehensive survey (80 studies, 2020-2025) reviewing deep learning techniques for early fire and smoke detection. The paper covers the full spectrum from traditional sensor-based methods to modern DL architectures (CNNs, YOLO variants, Faster R-CNN, transformers, spatiotemporal models), with particular emphasis on edge AI deployment, lightweight architectures, and multimodal fusion strategies. It provides a structured taxonomy of detection approaches by environment (forest, urban, indoor) and technical parameters (modality, real-time constraints).

## Architecture / Content (Survey Taxonomy)

```
FIRE & SMOKE DETECTION METHODS
|
+-- Traditional Methods
|   +-- Heat detectors (fixed temp, slow response)
|   +-- Smoke detectors (ionization/photoelectric, false alarms)
|   +-- Flame detectors (UV/IR, false triggers)
|   +-- Gas detectors (CO/hydrocarbons, calibration drift)
|   +-- Optical beam / aspirating detectors
|   +-- Remote monitoring (satellite: AVHRR, MODIS, VIIRS)
|       Common limitation: threshold-based, 30-300s delay
|
+-- Deep Learning Methods
    +-- Classification (CNNs)
    |   +-- VGG, ResNet, EfficientNet, MobileNet
    |   +-- Binary (fire/no-fire) or multi-class
    |
    +-- Object Detection (one-stage)
    |   +-- YOLO family (v5 through v11)
    |   +-- YOLOv8 consistently best speed-accuracy tradeoff
    |   +-- YOLO + U-Net hybrids reduce FP in wildfire
    |
    +-- Object Detection (two-stage)
    |   +-- Faster R-CNN + LSTM for temporal stabilization
    |   +-- Higher accuracy, higher compute cost
    |
    +-- Transformers & Attention
    |   +-- DETR-style for small-object smoke (high recall)
    |   +-- Swin Transformer + FPN for forest monitoring
    |   +-- Hierarchical transformers for smoke video
    |
    +-- Spatiotemporal Models
    |   +-- CNN+LSTM for flame flicker & smoke motion
    |   +-- 3D CNNs (C3D, I3D) for video fire detection
    |   +-- ConvLSTM for satellite-based monitoring
    |
    +-- Multimodal Fusion
        +-- RGB + infrared + thermal
        +-- Reduces FP vs. unimodal systems
```

### Key challenges identified:
1. **Dataset scarcity:** Most public datasets <10,000 samples, lack night/fog coverage
2. **False alarms:** Sunlight, headlights, fog, steam misclassified as fire
3. **Computational demands:** Pruning, quantization, knowledge distillation needed for edge
4. **Latency:** Emergency apps need <100ms; current embedded implementations >1s/frame
5. **Interpretability:** Grad-CAM helps but no standardized solutions
6. **Generalization:** Models overfit to training domain

### Paper organization:
- Section 3: Traditional methods
- Section 4: DL architectures (CNNs, YOLO, R-CNN, hybrid)
- Section 5: Datasets review
- Section 6: Taxonomy by deployment context
- Section 7: Edge AI deployment strategies
- Section 8: Open challenges
- Section 9: Future directions

## Results

This is a survey paper -- no original experimental results. Key findings from reviewed works:

- **YOLOv8 consistently achieves best speed-accuracy tradeoff** for wildfire on constrained platforms
- YOLOv5 enabled real-time UAV wildfire monitoring
- **CNN+LSTM** captures spatiotemporal dynamics for early smoke recognition in video
- **Bayesian Faster R-CNN + LSTM** enhances reliability in video-based detection
- **EfficientNet + DeepLabv3+** reduced FP in outdoor smoke monitoring
- YOLO + U-Net hybrids refine smoke plume localization, reducing FP
- RGB-infrared fusion with transformers achieved >90% precision+recall in parks
- GA-optimized YOLOv5 variants: strong indoor smoke detection
- **YOLOv11n** achieved optimal recall-precision balance on D-Fire dataset

## Applicability to Pyronear

**Highly applicable as a reference guide for architectural decisions and future directions.**

- **Validates Pyronear's two-stage approach:** The survey confirms that combining a lightweight first-stage detector (YOLO) with temporal verification (LSTM/sequential) is the established best practice for balancing speed and accuracy.
- **Edge AI is critical but under-explored:** The paper highlights that practical edge deployment (pruning, quantization, knowledge distillation) is under-represented in research, which is exactly Pyronear's challenge with RaspPi.
- **CNN+LSTM is proven for temporal verification:** Multiple reviewed works confirm this architecture captures spatiotemporal dynamics of smoke motion effectively.
- **YOLO + U-Net for FP reduction:** Hybrid approaches combining detection with segmentation could help Pyronear's server-side model better delineate smoke from clouds/fog.
- **Multimodal fusion is a future direction:** If Pyronear adds IR/thermal cameras, the survey provides a roadmap for fusion architectures.
- **Dataset challenges apply directly:** Small datasets, class imbalance, and lack of negative hard examples (fog, clouds) are Pyronear's actual problems.

## Takeaways for Implementation

1. **Stick with YOLO family for edge detection** -- the survey confirms it as the dominant choice for real-time fire/smoke detection across the literature.
2. **Consider knowledge distillation** to compress a larger YOLO (e.g., YOLOv10-M trained on full data) into a smaller model (YOLOv10-N) for the RaspPi, preserving accuracy while reducing compute.
3. **Pruning + quantization (INT8)** should be applied to the edge model. The survey notes this is critical but under-explored for fire detection specifically.
4. **CNN+LSTM remains the go-to temporal model.** However, consider exploring 3D CNN alternatives (C3D/I3D) or ConvLSTM if the server has sufficient GPU resources.
5. **Address false alarm sources explicitly:** Include hard negatives (fog, clouds, sunlight reflections, headlights) in training data. The survey identifies these as the primary FP sources.
6. **Grad-CAM for interpretability** could help operators understand why a detection was triggered, building trust in the system.
7. **Federated learning** is flagged as a future direction for distributed camera networks -- relevant for Pyronear's multi-site deployment where data cannot easily be centralized.
