# Wildfire Smoke Detection Papers -- Collection Summary

This collection of 28 research papers covers deep learning approaches for wildfire smoke detection and adjacent domains, with a particular focus on **temporal/video-based models** that leverage sequential frames to distinguish smoke from visually similar phenomena (clouds, fog, haze). These papers inform the design of detection systems for [Pyronear](https://pyronear.org/), where early and reliable smoke detection from fixed cameras is the core challenge.

---

## 1. Temporal / Video Models for Smoke Detection

These papers exploit sequential frames to capture the spatiotemporal dynamics of smoke -- its growth, motion, and texture evolution over time. This is the primary area of interest.

### SmokeyNet (Dewangan et al., 2022)
**FIgLib & SmokeyNet: Dataset and Deep Learning Model for Real-Time Wildland Fire Smoke Detection**
*Remote Sensing (MDPI)* | [arXiv](https://arxiv.org/abs/2112.08598)

Three-component spatiotemporal architecture combining a CNN (ResNet34) for per-frame feature extraction, an LSTM for temporal fusion across consecutive frames, and a Vision Transformer for global context. Introduces the FIgLib dataset (~25k labeled images from HPWREN cameras) where even human annotators struggle to detect smoke from single frames, demonstrating the necessity of temporal information. Achieves 3.12 min average time-to-detection with 89.84% precision. **Directly relevant to Pyronear's camera-based setup.**

### SlowFastMTB (Choi et al., 2022)
**A Video-based SlowFastMTB Model for Detection of Small Amounts of Smoke from Incipient Forest Fires**
*Journal of Computational Design and Engineering (Oxford)* | [DOI](https://doi.org/10.1093/jcde/qwac027)

Adapts the SlowFast dual-pathway video architecture (slow pathway for semantics, fast pathway for motion) to incipient smoke detection. Introduces the MTB (Moving object pixels To Bounding box pixels) algorithm for automatic annotation of smoke bounding boxes with fuzzy boundaries. Particularly effective for small, early-stage smoke plumes. **Useful reference for leveraging action recognition architectures.**

### Lightweight Student LSTM (Jeong et al., 2020)
**Light-Weight Student LSTM for Real-Time Wildfire Smoke Detection**
*Sensors (MDPI)* | [DOI](https://doi.org/10.3390/s20195508)

Combines YOLOv3 for spatial detection with a deep LSTM for temporal classification on "smoke-tube" features extracted from sequential frames. Applies teacher-student knowledge distillation to compress the LSTM, achieving 8.4x parameter reduction while maintaining accuracy. **Demonstrates that temporal models can be made lightweight for real-time deployment.**

### ViT + 3D-CNN Spatiotemporal (Lilhore et al., 2026)
**Real Time Fire and Smoke Detection Using Vision Transformers and Spatiotemporal Learning**
*Scientific Reports (Nature)* | [DOI](https://doi.org/10.1038/s41598-026-36687-9)

Integrates Vision Transformers with 3D-CNNs for joint spatiotemporal feature extraction, using a Transformer encoder to track the temporal evolution of fire and smoke. Reports 99.2% accuracy on the NASA Space Apps Challenge dataset. **Most recent temporal approach in the collection.**

### ELASTIC-YOLOv3 + Fire-Tube (Park & Ko, 2020)
**Two-Step Real-Time Night-Time Fire Detection in an Urban Environment Using Static ELASTIC-YOLOv3 and Temporal Fire-Tube**
*Sensors (MDPI)* | [DOI](https://doi.org/10.3390/s20082202)

Two-step pipeline for night-time fire detection: ELASTIC-YOLOv3 detects fire candidates regardless of size, then N frames are accumulated into a temporal "fire-tube" from which optical flow is computed and converted into a Bag-of-Features (BoF) histogram, classified by a random forest. Addresses the specific challenge of night-time urban fires where neon signs, headlights, and streetlights cause false positives. **Relevant temporal concept: the fire-tube as a lightweight way to capture motion dynamics without heavy sequence models. Also from the same lab (Keimyung University) as the Student LSTM paper.**

### FLAME (Gragnaniello et al., 2024)
**FLAME: Fire Detection in Videos Combining a Deep Neural Network with a Model-based Motion Analysis**
*Neural Computing and Applications (Springer)* | [DOI](https://doi.org/10.1007/s00521-024-10963-z)

Hybrid approach that pairs a CNN-based frame-level detector with a physics-informed motion analysis module. The motion component analyzes flame/smoke movement patterns to filter out false positives from the neural network. **Interesting angle: using motion priors as a post-processing filter rather than an end-to-end temporal model.**

---

## 2. Single-Frame Detection Architectures

These papers focus on improving spatial detection from individual images, relevant as the per-frame backbone within a temporal pipeline.

### Nemo / DETR (Yazdi et al., 2022)
**Nemo: An Open-Source Transformer-Supercharged Benchmark for Fine-Grained Wildfire Smoke Detection**
*Remote Sensing (MDPI)* | [DOI](https://doi.org/10.3390/rs14163979)

First open-source DETR-based benchmark for wildfire smoke. Adapts Facebook's Detection Transformer to smoke detection, where the encoder's self-attention captures long-range dependencies critical for small, distant smoke plumes. Detects 97.9% of incipient fires within 5 min on HPWREN sequences with average detection at 3.6 min. **Provides open-source models and datasets.**

### Smoke-DETR (Sun & Cheng, 2024)
**Smoke Detection Transformer: An Improved Real-Time Detection Transformer Smoke Detection Model for Early Fire Warning**
*Fire (MDPI)* | [DOI](https://doi.org/10.3390/fire7120488)

Improves RT-DETR with three components: Enhanced Channel-wise Partial Convolution (ECPConv) for efficient feature extraction, Efficient Multi-Scale Attention (EMA) for the backbone, and a Multi-Scale Foreground-Focus Fusion Pyramid Network (MFFPN). Achieves +3.6pp precision over baseline RT-DETR. **Good reference for smoke-specific architectural modifications.**

### RT-DETR-Smoke (Wang et al., 2025)
**RT-DETR-Smoke: A Real-Time Transformer for Forest Smoke Detection**
*Fire (MDPI)* | [DOI](https://doi.org/10.3390/fire8050170)

Another RT-DETR variant with a hybrid CNN-Transformer encoder, coordinate attention for smoke-edge localization, and WShapeIoU loss. Achieves 87.75% mAP@0.5 at 445.50 FPS. **Demonstrates that transformer detectors can be made very fast.**

### CCPE Swin Transformer (Wang et al., 2025)
**Wildfire Smoke Detection System: Model Architecture, Training Mechanism, and Dataset**
*International Journal of Intelligent Systems (Wiley)* | [arXiv](https://arxiv.org/abs/2311.10116)

Cross Contrast Patch Embedding (CCPE) module on Swin Transformer to capture low-level features (color, transparency, texture) that vanilla Transformers miss. Introduces Separable Negative Sampling Mechanism for hard-negative mining. Tested on FIgLib and the SKLFS-WildFire Test dataset (largest real-world test set). **Addresses the problem of smoke-like false positives (fog, clouds).**

### Ultra-lightweight Conv-Transformer (Chaturvedi et al., 2024)
**Ultra-lightweight Convolution-Transformer Network for Early Fire Smoke Detection**
*Fire Ecology (SpringerOpen)* | [DOI](https://doi.org/10.1186/s42408-024-00304-9)

Hybrid convolution-transformer with only 0.6M parameters and 0.4B FLOPs, designed for resource-constrained edge deployment. Achieves >99% accuracy on multispectral satellite imagery. **Relevant for edge deployment constraints similar to Pyronear's embedded cameras.**

### YOLOv10 (Wang et al., 2024)
**YOLOv10: Real-Time End-to-End Object Detection**
*NeurIPS 2024* | [arXiv](https://arxiv.org/abs/2405.14458)

General-purpose real-time detector introducing NMS-free dual label assignments and holistic efficiency-accuracy driven architecture design. Not smoke-specific, but represents the YOLO state-of-the-art. **Potential backbone for Pyronear's per-frame detection stage.**

---

## 3. Datasets and Benchmarks

### PyroNear2025 (Lostanlen et al., 2024)
**Constructing a Real-World Benchmark for Early Wildfire Detection with the New PYRONEAR-2025 Dataset**
*arXiv* | [arXiv](https://arxiv.org/abs/2402.05349)

~150k annotations on 50k images covering 640 wildfires from France, Spain, Chile, and USA. Crucially, includes **video sequences** enabling temporal model training and evaluation. Baseline F1 ~70% indicates a challenging benchmark. **Pyronear's own dataset -- the primary training/evaluation resource.**

### SmokeBench (Qi et al., 2024)
**SmokeBench: Evaluating Multimodal Large Language Models for Wildfire Smoke Detection**
*arXiv* | [arXiv](https://arxiv.org/abs/2512.11215)

First benchmark evaluating multimodal LLMs (GPT-4V, Claude, Gemini) on wildfire smoke classification and localization tasks using the FIgLib dataset. **Explores whether foundation models can be applied to smoke detection without task-specific training.**

### Fire and Smoke Datasets: 20 Years (Haeri Boroujeni et al., 2025)
**Eyes on the Environment: AI-Driven Analysis for Fire and Smoke Classification, Segmentation, and Detection**
*arXiv* | [arXiv](https://arxiv.org/abs/2503.14552)

Systematic review of 20 years of fire/smoke datasets across RGB, thermal, and infrared modalities. Evaluates ResNet-50, DeepLab-V3, and YOLOv8 across datasets. **Useful for identifying additional training data sources.**

---

## 4. Surveys

### Comprehensive DL Review (Elhanashi et al., 2025)
**Early Fire and Smoke Detection Using Deep Learning: A Comprehensive Review of Models, Datasets, and Challenges**
*Applied Sciences (MDPI)* | [DOI](https://doi.org/10.3390/app151810255)

Covers the full landscape: CNNs, RNNs, YOLO variants, Faster R-CNN, spatiotemporal models (SlowFast, Video Swin, TimeSformer), and deployment challenges. **Best single reference for mapping the field.**

### Video Anomaly Detection Survey (Liu et al., 2024)
**Video Anomaly Detection in 10 Years: A Survey and Outlook**
*arXiv* | [arXiv](https://arxiv.org/abs/2405.19387)

Comprehensive survey of temporal anomaly detection in surveillance video. Covers reconstruction-based, prediction-based, and classification-based methods. Many techniques (temporal consistency modeling, future frame prediction, change detection) are transferable to smoke detection. **Cross-domain reference for temporal modeling ideas.**

---

## 5. Adjacent Domains -- Video Foundation Models

The foundational architectures behind the temporal backbones used in the `time-wildfire` repo. Understanding these is key to choosing and tuning the right model.

### VideoMAE (Tong et al., 2022)
**VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training**
*NeurIPS 2022* | [arXiv](https://arxiv.org/abs/2203.12602)

Self-supervised video pre-training via masked autoencoding with extremely high masking ratios (90-95%). Critically **data-efficient** -- achieves strong results with only ~3.5k training videos, which is important given limited wildfire video data. The `time-wildfire` repo uses this as the `videomae` backbone. **Pre-training on unlabeled outdoor camera footage then fine-tuning on smoke data is a promising direction.**

### VideoMAE V2 (Wang et al., 2023)
**VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking**
*CVPR 2023* | [arXiv](https://arxiv.org/abs/2303.16727)

Scales VideoMAE to billion parameters with dual masking (encoder + decoder). Introduces a progressive pre-training paradigm: first pre-train on diverse unlabeled video, then post-pre-train on labeled data. **The progressive training could be adapted: pre-train on generic outdoor/surveillance video, then fine-tune on Pyronear wildfire sequences.**

### TimeSformer (Bertasius et al., 2021)
**Is Space-Time Attention All You Need for Video Understanding?**
*ICML 2021* | [arXiv](https://arxiv.org/abs/2102.05095)

Pure attention architecture with "divided space-time attention" -- applies temporal and spatial attention separately. 3x faster to train than 3D CNNs, 10x less inference compute than SlowFast. **Its efficiency makes it attractive for Pyronear's real-time constraints, and it's not yet in the `time-wildfire` repo -- worth adding as a backbone.**

### ViViT (Arnab et al., 2021)
**ViViT: A Video Vision Transformer**
*ICCV 2021* | [arXiv](https://arxiv.org/abs/2103.15691)

Explores four factorizations of spatial and temporal attention for video classification. The factorised encoder variant (spatial attention per frame, then temporal attention across frames) is the most efficient. Used as the `vivit` backbone in `time-wildfire`. **Reference paper for understanding the architecture choices.**

---

## 6. Adjacent Domains -- Online / Streaming Detection

Pyronear cameras run continuously. These papers solve the problem of real-time detection in streaming video without waiting for a clip to complete.

### TeSTra (Zhao & Krahenbuhl, 2022)
**Real-time Online Video Detection with Temporal Smoothing Transformers**
*ECCV 2022* | [arXiv](https://arxiv.org/abs/2209.09236)

Reformulates cross-attention with temporal smoothing kernels (box or Laplace) so each new frame requires only O(1) computation instead of reprocessing the full temporal window. Runs 6x faster than sliding-window transformers on 2048-frame sequences. **Directly applicable to Pyronear's continuous camera feeds where latency matters.**

### MATR (Song et al., 2024)
**Online Temporal Action Localization with Memory-Augmented Transformer**
*ECCV 2024* | [arXiv](https://arxiv.org/abs/2408.02957)

Memory-augmented transformer that selectively preserves past segment features in a memory queue, enabling long-term context for online detection. Uses dual decoders: one for detecting action ends, another for scanning memory to find action starts. **The memory mechanism maps well to tracking smoke evolution over minutes -- smoke "starts" are exactly what Pyronear needs to detect.**

### LSTR (Xu et al., 2021)
**Long Short-Term Transformer for Online Action Detection**
*NeurIPS 2021* | [arXiv](https://arxiv.org/abs/2107.03377)

Dual-memory architecture with long-term and short-term memory processed by a Transformer. The long-term memory captures extended context while the short-term memory focuses on recent dynamics. **Relevant for smoke that evolves slowly (minutes) while fast-changing false positives (clouds moving, lighting changes) need short-term discrimination.**

---

## 7. Adjacent Domains -- Small Object Detection

Incipient smoke plumes are often just a few pixels on high-resolution camera images, far from the camera.

### Small Object Detection Survey (2025)
**Small Object Detection: A Comprehensive Survey on Challenges, Techniques and Real-World Applications**
*arXiv* | [arXiv](https://arxiv.org/abs/2503.20516)

Comprehensive survey covering multi-scale feature extraction, super-resolution techniques, attention mechanisms, and transformer architectures specifically for tiny objects. **Directly applicable to detecting distant smoke plumes that occupy very few pixels in Pyronear camera images.**

---

## 8. Adjacent Domains -- Few-Shot Detection & Domain Adaptation

Wildfire data is scarce and domain-specific (different cameras, geographies, seasons, lighting conditions).

### Few-Shot Object Detection in Remote Sensing (Zhang et al., 2025)
**Generalization-Enhanced Few-Shot Object Detection in Remote Sensing**
*arXiv* | [arXiv](https://arxiv.org/abs/2501.02474)

Addresses rapid adaptation to novel classes with limited samples in remote sensing while retaining base class performance. **The domain gap between different camera networks (Pyronear France vs. HPWREN US) is exactly this problem -- models trained on one network must generalize to another with minimal new labels.**

### Beyond Few-Shot Object Detection Survey (Li et al., 2024)
**Beyond Few-shot Object Detection: A Detailed Survey**
*arXiv* | [arXiv](https://arxiv.org/abs/2408.14249)

Comprehensive survey covering five categories: standard FSOD, generalized FSOD, incremental FSOD, open-set FSOD, and few-shot domain-adaptive object detection (FSDAOD). **The FSDAOD category is most relevant for adapting smoke detectors across camera domains.**

---

## 9. Adjacent Domains -- Edge Deployment & Model Compression

Pyronear runs on embedded cameras with limited compute. Temporal models are expensive -- compression is essential.

### Vision Transformers on the Edge (2025)
**Vision Transformers on the Edge: A Comprehensive Survey of Model Compression and Acceleration Strategies**
*arXiv* | [arXiv](https://arxiv.org/abs/2503.02891)

Covers pruning, quantization, and knowledge distillation specifically for Vision Transformers. **Directly relevant to deploying VideoMAE, ViViT, or TimeSformer-based smoke detectors on Pyronear's edge hardware.**

---

## Key Takeaways

1. **Temporal information is essential** -- FIgLib/SmokeyNet showed that even humans cannot reliably detect smoke from single frames. Sequential models consistently outperform single-frame detectors.

2. **Promising temporal architectures** for Pyronear (ordered by relevance):
   - **CNN + LSTM/Transformer**: SmokeyNet-style per-frame CNN features fed into a temporal module (matches `cnn_transformer` in `time-wildfire` repo)
   - **3D CNNs**: ResNet3D / SlowFast directly process video volumes (matches `3dresnet` in `time-wildfire` repo)
   - **Video Transformers**: VideoMAE / ViViT for long-range temporal dependencies (matches `videomae` / `vivit` in `time-wildfire` repo)
   - **Motion analysis + detection**: FLAME-style post-filtering with motion priors

3. **Streaming/online detection is underexplored in the fire domain** -- TeSTra, MATR, and LSTR solve the continuous camera feed problem that Pyronear faces, but none have been applied to smoke detection yet. This is a gap worth investigating.

4. **Lightweight deployment matters** -- teacher-student distillation (Jeong 2020), ultra-lightweight architectures (Chaturvedi 2024), and ViT compression techniques (pruning, quantization, KD) demonstrate that temporal models can be compressed for edge cameras.

5. **Data efficiency is critical** -- VideoMAE's self-supervised pre-training works with very few videos, and few-shot detection methods can help adapt models across camera domains (e.g., France to US).

6. **Small object detection techniques** are directly relevant -- incipient smoke plumes are tiny and distant, making multi-scale features and super-resolution approaches valuable.

7. **Open datasets available** -- PyroNear2025 (with video), FIgLib, Nemo, and HPWREN provide training data. The 20-year dataset review identifies additional sources.

8. **The `time-wildfire` repo** already implements the key architectures from the literature (EfficientNet motion, 3D ResNet, VideoMAE, ViViT, CNN+Transformer), making it a solid experimental platform.
