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
make pull           # downloads PDFs from S3 into pdfs/
```

### Adding a new paper

1. Drop the PDF into `pdfs/` using the naming convention `Year-Short-Title-Author.pdf`
2. Add a row to `papers.csv`
3. Update `SUMMARY.md` with a description
4. Track and push the changes:

```bash
uv run dvc add pdfs/
make push
git add pdfs.dvc papers.csv SUMMARY.md README.md
git commit -m "Add paper: <title>"
```

### Available commands

```
make install        Install dependencies from uv.lock
make pull           Pull PDF data from S3 via DVC
make push           Push PDF data to S3 via DVC
```

## Structure

```
papers/
├── README.md          # This file
├── SUMMARY.md         # Narrative summary of all papers, grouped by theme
├── papers.csv         # Structured metadata (title, authors, year, architecture, etc.)
└── pdfs/              # PDF files (named as Year-Short-Title-Author.pdf)
```

## Papers (28)

### Temporal / Video Models for Smoke Detection
| Year | Paper | Architecture |
|------|-------|-------------|
| 2020 | Lightweight Student LSTM (Jeong et al.) | YOLOv3 + LSTM (teacher-student) |
| 2020 | ELASTIC-YOLOv3 + Fire-Tube (Park & Ko) | YOLOv3 + temporal fire-tube + BoF + random forest |
| 2022 | SmokeyNet (Dewangan et al.) | CNN (ResNet34) + LSTM + ViT |
| 2022 | SlowFastMTB (Choi et al.) | SlowFast + MTB annotation |
| 2024 | FLAME (Gragnaniello et al.) | CNN + model-based motion analysis |
| 2026 | ViT + 3D-CNN (Lilhore et al.) | ViT + 3D-CNN + Transformer encoder |

### Single-Frame Detection
| Year | Paper | Architecture |
|------|-------|-------------|
| 2022 | Nemo (Yazdi et al.) | DETR |
| 2024 | Smoke-DETR (Sun & Cheng) | RT-DETR + ECPConv + EMA + MFFPN |
| 2024 | Ultra-lightweight (Chaturvedi et al.) | Conv-Transformer (0.6M params) |
| 2024 | YOLOv10 (Wang et al.) | YOLOv10 |
| 2025 | RT-DETR-Smoke (Wang et al.) | RT-DETR + hybrid encoder |
| 2025 | CCPE Swin (Wang et al.) | Swin Transformer + CCPE |

### Datasets & Benchmarks
| Year | Paper | Dataset |
|------|-------|---------|
| 2024 | PyroNear2025 (Lostanlen et al.) | 150k annotations, 50k images, 640 wildfires |
| 2024 | SmokeBench (Qi et al.) | Multimodal LLM evaluation on FIgLib |
| 2025 | 20-Year Dataset Review (Haeri Boroujeni et al.) | Survey of RGB / thermal / IR datasets |

### Surveys
| Year | Paper | Scope |
|------|-------|-------|
| 2024 | Video Anomaly Detection Survey (Liu et al.) | 10-year survey of temporal anomaly detection |
| 2025 | Comprehensive DL Review (Elhanashi et al.) | CNNs, RNNs, YOLO, transformers, spatiotemporal |

### Adjacent: Video Foundation Models
| Year | Paper | Architecture |
|------|-------|-------------|
| 2021 | TimeSformer (Bertasius et al.) | Divided space-time attention |
| 2021 | ViViT (Arnab et al.) | Video Vision Transformer (4 factorizations) |
| 2022 | VideoMAE (Tong et al.) | Masked video autoencoder |
| 2023 | VideoMAE V2 (Wang et al.) | Dual masking, billion-scale |

### Adjacent: Online / Streaming Detection
| Year | Paper | Architecture |
|------|-------|-------------|
| 2021 | LSTR (Xu et al.) | Long short-term memory Transformer |
| 2022 | TeSTra (Zhao & Krahenbuhl) | Temporal smoothing kernels, O(1) per frame |
| 2024 | MATR (Song et al.) | Memory-augmented Transformer |

### Adjacent: Small Object / Few-Shot / Edge Deployment
| Year | Paper | Focus |
|------|-------|-------|
| 2024 | Beyond Few-Shot OD Survey (Li et al.) | 5 categories of few-shot detection |
| 2025 | Few-Shot OD in Remote Sensing (Zhang et al.) | Domain adaptation with limited labels |
| 2025 | Small Object Detection Survey | Multi-scale, super-resolution, attention |
| 2025 | ViT on the Edge Survey | Pruning, quantization, knowledge distillation |

## Related Repos

- [time-wildfire](https://github.com/rensortino/time-wildfire) -- Temporal smoke detection with EfficientNet, 3D ResNet, VideoMAE, ViViT, CNN+Transformer backbones + SAM3 tracking
