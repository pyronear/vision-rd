# vision-rd

Research papers on temporal ML models for wildfire smoke detection and related topics.

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/) for Python dependency management
- AWS credentials configured for access to the S3 bucket (`s3://pyro-survey-research/dvc/`)

### Installation

```bash
git clone <repo-url>
cd papers
make install        # installs DVC and dependencies from uv.lock
make pull           # downloads PDFs and notes from S3
```

### Adding a new paper

1. Drop the PDF into `pdfs/` using the naming convention `Year-Short-Title-Author.pdf`
2. Add a row to `papers.csv`
3. Write reading notes in `notes/` as `year-short-title.md`
4. Update `SUMMARY.md` with a description
5. Track and push the changes:

```bash
uv run dvc add pdfs/ notes/
make push
git add pdfs.dvc notes.dvc papers.csv SUMMARY.md README.md
git commit -m "Add paper: <title>"
```

### Available commands

```
make install        Install dependencies from uv.lock
make pull           Pull PDF data and notes from S3 via DVC
make push           Push PDF data and notes to S3 via DVC
```

## Structure

```
vision-rd/
├── README.md        # This file
├── SUMMARY.md       # Narrative summary grouped by theme
├── CLAUDE.md        # Project context for AI assistants
├── papers.csv       # Structured metadata for all papers
├── pdfs/            # PDF files (DVC-tracked)
├── notes/           # Per-paper reading notes (DVC-tracked)
├── Makefile         # install / pull / push
├── pyproject.toml   # Python dependencies (dvc[s3])
└── uv.lock          # Lockfile
```

## Papers (28)

| Year | Paper | Category | Architecture / Focus | PDF | Notes |
|------|-------|----------|---------------------|-----|-------|
| 2020 | Lightweight Student LSTM (Jeong et al.) | Temporal | YOLOv3 + LSTM, teacher-student distillation | [pdf](pdfs/2020-Lightweight-Student-LSTM-Wildfire-Smoke-Detection-Jeong-et-al.pdf) | [notes](notes/2020-student-lstm.md) |
| 2020 | ELASTIC-YOLOv3 + Fire-Tube (Park & Ko) | Temporal | YOLOv3 + fire-tube + BoF + random forest | [pdf](pdfs/2020-Night-Time-Fire-Detection-ELASTIC-YOLOv3-Fire-Tube-Park-Ko.pdf) | [notes](notes/2020-fire-tube.md) |
| 2021 | TimeSformer (Bertasius et al.) | Video Foundation | Divided space-time attention | [pdf](pdfs/2021-TimeSformer-Space-Time-Attention-Video-Bertasius-et-al.pdf) | [notes](notes/2021-timesformer.md) |
| 2021 | ViViT (Arnab et al.) | Video Foundation | Video Vision Transformer, 4 factorizations | [pdf](pdfs/2021-ViViT-Video-Vision-Transformer-Arnab-et-al.pdf) | [notes](notes/2021-vivit.md) |
| 2021 | LSTR (Xu et al.) | Online Detection | Long short-term memory Transformer | [pdf](pdfs/2021-LSTR-Long-Short-Term-Transformer-Online-Action-Detection-Xu-et-al.pdf) | [notes](notes/2021-lstr.md) |
| 2022 | Nemo / DETR (Yazdi et al.) | Spatial | DETR for wildfire smoke, open-source benchmark | [pdf](pdfs/2022-Nemo-Transformer-Wildfire-Smoke-Benchmark-Yazdi-et-al.pdf) | [notes](notes/2022-nemo-detr.md) |
| 2022 | SlowFastMTB (Choi et al.) | Temporal | SlowFast + MTB bounding box algorithm | [pdf](pdfs/2022-SlowFastMTB-Incipient-Forest-Fire-Smoke-Choi-et-al.pdf) | [notes](notes/2022-slowfastmtb.md) |
| 2022 | SmokeyNet (Dewangan et al.) | Temporal | CNN (ResNet34) + LSTM + ViT on tiled frames | [pdf](pdfs/2022-SmokeyNet-FIgLib-Spatiotemporal-Smoke-Detection-Dewangan-et-al.pdf) | [notes](notes/2022-smokeynet.md) |
| 2022 | TeSTra (Zhao & Krahenbuhl) | Online Detection | Temporal smoothing kernels, O(1) per frame | [pdf](pdfs/2022-TeSTra-Temporal-Smoothing-Transformer-Online-Detection-Zhao-Krahenbuhl.pdf) | [notes](notes/2022-testra.md) |
| 2022 | VideoMAE (Tong et al.) | Video Foundation | Masked video autoencoder, data-efficient | [pdf](pdfs/2022-VideoMAE-Masked-Autoencoders-Video-Pre-Training-Tong-et-al.pdf) | [notes](notes/2022-videomae.md) |
| 2023 | VideoMAE V2 (Wang et al.) | Video Foundation | Dual masking, billion-scale, progressive training | [pdf](pdfs/2023-VideoMAE-V2-Scaling-Dual-Masking-Wang-et-al.pdf) | [notes](notes/2023-videomae-v2.md) |
| 2024 | Beyond Few-Shot OD Survey (Li et al.) | Few-Shot | 5 categories of few-shot detection | [pdf](pdfs/2024-Beyond-Few-Shot-Object-Detection-Survey.pdf) | [notes](notes/2024-few-shot-od-survey.md) |
| 2024 | FLAME (Gragnaniello et al.) | Temporal | DNN + GMM background subtraction + tracking FSM | [pdf](pdfs/2024-FLAME-Fire-Detection-Video-DNN-Motion-Analysis.pdf) | [notes](notes/2024-flame.md) |
| 2024 | MATR (Song et al.) | Online Detection | Memory-augmented Transformer for streaming | [pdf](pdfs/2024-MATR-Memory-Augmented-Transformer-Online-Action-Localization-Song-et-al.pdf) | [notes](notes/2024-matr.md) |
| 2024 | PyroNear2025 Dataset (Lostanlen et al.) | Dataset | 150k annotations, 50k images, 640 wildfires | [pdf](pdfs/2024-PyroNear2025-Dataset-Benchmark-Early-Wildfire-Lostanlen-et-al.pdf) | [notes](notes/2024-pyronear2025-dataset.md) |
| 2024 | Smoke-DETR (Sun & Cheng) | Spatial | RT-DETR + ECPConv + EMA + MFFPN | [pdf](pdfs/2024-Smoke-Detection-Transformer-RT-DETR-Sun-Cheng.pdf) | [notes](notes/2024-smoke-detr.md) |
| 2024 | SmokeBench (Qi et al.) | Benchmark | Multimodal LLM evaluation on FIgLib | [pdf](pdfs/2024-SmokeBench-Multimodal-LLM-Wildfire-Smoke-Detection.pdf) | [notes](notes/2024-smokebench.md) |
| 2024 | Ultra-lightweight (Chaturvedi et al.) | Spatial | Conv-Transformer, 0.6M params, edge deploy | [pdf](pdfs/2024-Ultra-Lightweight-Conv-Transformer-Fire-Smoke-Detection.pdf) | [notes](notes/2024-ultra-lightweight.md) |
| 2024 | Video Anomaly Survey (Liu et al.) | Survey | 10-year survey, reconstruction + MIL methods | [pdf](pdfs/2024-Video-Anomaly-Detection-10-Years-Survey.pdf) | [notes](notes/2024-video-anomaly-survey.md) |
| 2024 | YOLOv10 (Wang et al.) | General | NMS-free YOLO, edge-friendly | [pdf](pdfs/2024-YOLOv10-Real-Time-End-to-End-Object-Detection-Wang-et-al.pdf) | [notes](notes/2024-yolov10.md) |
| 2025 | CCPE Swin (Wang et al.) | Spatial | Swin + Cross Contrast Patch Embedding | [pdf](pdfs/2025-Wildfire-Smoke-Detection-CCPE-Swin-Transformer-Wang-et-al.pdf) | [notes](notes/2025-ccpe-swin.md) |
| 2025 | Comprehensive DL Review (Elhanashi et al.) | Survey | CNNs, RNNs, YOLO, transformers, spatiotemporal | [pdf](pdfs/2025-Early-Fire-Smoke-Detection-Comprehensive-Review.pdf) | [notes](notes/2025-comprehensive-review.md) |
| 2025 | Datasets 20-Year Review (Haeri Boroujeni et al.) | Survey | 29 fire/smoke datasets across modalities | [pdf](pdfs/2025-Fire-Smoke-Datasets-20-Years-Review.pdf) | [notes](notes/2025-datasets-review.md) |
| 2025 | Few-Shot Remote Sensing (Zhang et al.) | Few-Shot | Domain adaptation with limited labels | [pdf](pdfs/2025-Few-Shot-Object-Detection-Remote-Sensing-Generalization.pdf) | [notes](notes/2025-few-shot-remote-sensing.md) |
| 2025 | RT-DETR-Smoke (Wang et al.) | Spatial | RT-DETR + CoordAtt + WShapeIoU, 445 FPS | [pdf](pdfs/2025-RT-DETR-Smoke-Real-Time-Forest-Smoke-Detection.pdf) | [notes](notes/2025-rt-detr-smoke.md) |
| 2025 | Small Object Detection Survey | Survey | Multi-scale, super-resolution, attention | [pdf](pdfs/2025-Small-Object-Detection-Comprehensive-Survey.pdf) | [notes](notes/2025-small-object-survey.md) |
| 2025 | ViT on the Edge Survey | Survey | Pruning, quantization, knowledge distillation | [pdf](pdfs/2025-Vision-Transformers-Edge-Compression-Acceleration-Survey.pdf) | [notes](notes/2025-vit-edge-survey.md) |
| 2026 | ViT + 3D-CNN (Lilhore et al.) | Temporal | ViT + 3D-CNN + Transformer encoder | [pdf](pdfs/2026-Real-Time-Fire-Smoke-ViT-Spatiotemporal-Learning.pdf) | [notes](notes/2026-vit-3dcnn-spatiotemporal.md) |

## Related Repos

- [time-wildfire](https://github.com/rensortino/time-wildfire) -- Temporal smoke detection with EfficientNet, 3D ResNet, VideoMAE, ViViT, CNN+Transformer backbones + SAM3 tracking
