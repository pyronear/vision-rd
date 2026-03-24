# Constructing a Real-World Benchmark for Early Wildfire Detection with the New PyroNear2025 Dataset

- **Authors:** Mateo Lostanlen, Nicolas Isla, Jose Guillen, Renzo Zanca, Felix Veith, Cristian Buc, Valentin Barriere
- **Year/Venue:** 2024 (arXiv:2402.05349v3, Oct 2025)
- **PDF:** `/mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/papers/pdfs/2024-PyroNear2025-Dataset-Benchmark-Early-Wildfire-Lostanlen-et-al.pdf`

## Key Idea

PyroNear2025 is a new open-source dataset for early wildfire detection composed of both images and videos, with ~150,000 manual annotations on ~50,000 images covering 640 wildfires from France, Spain, Chile, and the USA. It is specifically designed for *early* smoke plume detection (small bounding boxes, avg ~2% of image area), making it far more challenging and realistic than existing datasets that focus on advanced-stage fires. It provides both a single-image dataset (PyroNear2025-I) and a video dataset (PyroNear2025-V) for training sequential/temporal models.

## Architecture / Content

This is a dataset paper, not a model paper. The benchmarking pipeline:

```
                         PYRONEAR2025 DATASET
                               |
            +------------------+------------------+
            |                                     |
    PyroNear2025-I                        PyroNear2025-V
    (4228 images, 4041 smoke)             (1049 videos, 640 fires)
            |                                     |
     YOLOv8-small                          YOLOv8 + ResNet + LSTM
     (single-frame detection)             (sequential verification)
            |                                     |
     Precision/Recall/F1                   Improved Recall + earlier
     @ optimal threshold tau_d             detection time
```

**Data sources:**
- HPWREN + ALERTWildfire camera networks (web-scraped)
- In-house PyroNear cameras (15 towers, 51 cameras in FR/ES/CL)
- Google image scraping (442 images)
- Synthetic smoke plumes via Blender (200 images)
- FIgLib re-annotated with bounding boxes (24,800 images)

**Annotation:** 5x cross-labeling by volunteers, Krippendorff's alpha for quality, custom collaborative annotation platform.

## Results

### Single-frame (YOLOv8-small) per-dataset F1:
| Dataset | F1 | Optimal tau_d |
|---|---|---|
| SmokeFrames-2.4k | 0.828 | 0.19 |
| SmokeFrames-50k | 0.576 | 0.05 |
| Nemo | 0.868 | 0.09 |
| AiForMankind | 0.817 | 0.09 |
| Fuego | 0.721 | 0.04 |
| **PyroNear2025-I** | **0.683** | **0.11** |

### Combined training (all datasets merged):
| Dataset | Precision | Recall | F1 |
|---|---|---|---|
| Overall | 0.880 | 0.836 | 0.852 |
| PyroNear2025-I | 0.838 | 0.745 | 0.789 |

### Video-based (sequential model vs. vanilla):
| Model | Precision | Recall | F1 | Time Elapsed (min) |
|---|---|---|---|---|
| Vanilla (1 frame) | **0.805** | 0.775 | 0.790 | 1.76 |
| Sequential | 0.793 | **0.853** | **0.822** | **1.17** |

- Sequential model improves recall by +10% relative with only -1.5% relative precision loss.
- Average detection time drops from 1.76 to 1.17 minutes.
- Synthetic images contribute +2% F1 on the test split.
- PyroNear2025-I yields the best overall cross-dataset F1 (68.8%).

## Applicability to Pyronear

**Directly applicable -- this is the Pyronear dataset paper.**

- **Primary training/eval dataset:** This is the canonical dataset for the Pyronear system. All future model development should use PyroNear2025-I for single-frame and PyroNear2025-V for temporal models.
- **Confirms the two-stage architecture:** The paper validates that using a low-threshold first-stage detector (YOLOv8) followed by a sequential model (ResNet+LSTM) on the server is the right approach -- it increases recall (+10%) with negligible precision loss.
- **Small bounding boxes (~2% image area):** Matches the real operational scenario of detecting distant smoke plumes. Detection threshold tau_d is highly sensitive (ranges 0.04-0.19), confirming the need for careful threshold tuning.
- **30s between frames aligns with video dataset:** PyroNear2025-V was built with images saved around detection events including 15-minute context windows, compatible with the 30s capture interval.
- **Cross-dataset training improves generalization:** Training on the combined dataset (PyroNear2025 + others) gives the best overall performance (F1=0.852), suggesting data augmentation with external datasets is valuable.

## Takeaways for Implementation

1. **Use low detection threshold at edge:** Set tau_d low on the RaspPi YOLOv8 (e.g., 0.04-0.11) to maximize recall; let the server-side sequential model filter FPs.
2. **Train on combined datasets:** Merge PyroNear2025 with FIgLib, SmokeFrames-2.4k, Nemo, AiForMankind, and Fuego for best generalization.
3. **Sequential model architecture:** YOLO bbox crops -> ResNet feature extraction -> LSTM binary classification is proven effective and lightweight enough for server-side.
4. **Include synthetic data:** The +2% F1 from 200 synthetic Blender images shows even small synthetic augmentation helps.
5. **Keep wildfire events disjoint across train/val/test splits** to avoid data leakage.
6. **Balance dataset by selecting ~7 images per incident** to avoid redundancy bias.
7. **Code available at:** https://github.com/joseg20/wildfires2025
