# Eyes on the Environment: AI-Driven Analysis for Fire and Smoke Classification, Segmentation, and Detection

- **Authors:** Sayed Pedram Haeri Boroujeni, Niloufar Mehrabi, Fatemeh Afghah, Connor Peter McGrath, Danish Bhatkar, Mithilesh Anil Biradar, Abolfazl Razi
- **Year/Venue:** 2025 (arXiv:2503.14552v2, Jul 2025)
- **PDF:** `/mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/papers/pdfs/2025-Fire-Smoke-Datasets-20-Years-Review.pdf`

## Key Idea

A comprehensive review of fire and smoke datasets collected over the past 20 years, systematically analyzing 29 datasets by type, size, format, collection method, geographical diversity, imaging modality (RGB, thermal, infrared), and applicability to classification, segmentation, and detection tasks. The paper also performs extensive experimental benchmarking using ResNet-50 (classification), DeepLab-V3 (segmentation), and YOLOv8 (detection) across multiple datasets, providing comparative performance evaluations highlighting each dataset's strengths, limitations, and generalizability.

## Architecture / Content (Dataset Catalog)

```
FIRE & SMOKE DATASETS (29 datasets, 2006-2024)
|
+-- By Category:
|   +-- Original (collected by authors): FLAME1, FLAME2, FLAME3,
|   |   BA-UAV, DataCluster, FireFront, FESB MLID, Corsican, FireSense
|   +-- Aggregated (from existing sources): FireDetn, Paddle, Kaggle,
|   |   FD-Dataset, FF-Det, FireNet, AIDER, FIRE, CAIR, FiSmo, BoWFire
|   +-- Mixed (both): DFS, D-Fire, ForestryImage, MIVIA, VisiFire, FireClips
|   +-- Generated (synthetic): FLAME-SD, FireFly
|
+-- By Perspective:
|   +-- Aerial/UAV: FLAME1/2/3, BA-UAV, FireFront, FireFly
|   +-- Terrestrial (ground cameras): DFS, FireDetn, Paddle, etc.
|   +-- Mixed: D-Fire, ForestryImage
|
+-- By Modality:
|   +-- Image only: most datasets
|   +-- Image + Video: DFS, FLAME2, FireFront, D-Fire, FD-Dataset,
|   |   FESB MLID, FiSmo, Corsican, FireSense, MIVIA, VisiFire, FireClips
|
+-- By Size (images/frames):
    +-- Large (>50K): FLAME1 (47,992), FD-Dataset (50,000),
    |   ForestryImage (317,921), MIVIA (62,690)
    +-- Medium (5K-50K): most datasets
    +-- Small (<5K): Paddle (3,701), DeepFire (1,900), Kaggle (1,900),
        FF-Det (1,900), FireNet (502), BoWFire (466), FIRE (999)
```

### Key dataset characteristics table:
| Dataset | Year | Size | Type | Labeling | Location | Source |
|---|---|---|---|---|---|---|
| FLAME3 | 2024 | 13,997 | Image | Fire,Smoke,None | OR/FL, USA | UAV |
| FLAME-SD | 2024 | 10,000 | Image | Fire,Smoke,None | Forest | Synthetic |
| DFS | 2023 | 9,462 | Image+Video | Fire,Smoke,Other | Rural,Urban | Mixed |
| D-Fire | 2020 | 21,527 | Image+Video | Fire,Smoke,None,Both | Brazil | Mixed |
| FLAME1 | 2020 | 47,992 | Image+Video | Fire,No Fire | AZ, USA | UAV |
| FD-Dataset | 2020 | 50,000 | Image+Video | Fire,No Fire | Rural,Urban | Search |
| ForestryImage | 2018 | 317,921 | Image | Fire,No Fire | Rural | Mixed |

## Results

No original model proposed; benchmarking results across datasets using standard models:

### Key experimental findings:
- **ResNet-50** (classification): Performance varies significantly across datasets, with aerial/UAV datasets (FLAME series) showing higher accuracy than terrestrial datasets
- **DeepLab-V3** (segmentation): Works best on datasets with pixel-level annotations; struggles with datasets providing only bounding boxes
- **YOLOv8** (detection): Best performance on datasets with clear, well-annotated bounding boxes; struggles with small or low-contrast smoke
- **Cross-dataset generalization is poor:** Models trained on one dataset often fail on others, especially across different perspectives (aerial vs. ground) and modalities
- **Synthetic datasets** (FLAME-SD, FireFly) help as training augmentation but cannot replace real data

## Applicability to Pyronear

**High applicability as a dataset selection guide for training data augmentation.**

- **Identifies best complementary datasets for Pyronear training:**
  - **D-Fire** (21,527 images, fire+smoke+both labels, bounding boxes): Good mixed dataset for augmentation
  - **DFS** (9,462 images+video, multi-class): Modern, diverse dataset
  - **FD-Dataset** (50,000 images+video): Large scale, rural+urban coverage
  - **FESB MLID** (400 images+video, Mediterranean): Geographically relevant to Pyronear's EU deployments
- **Confirms gaps in existing datasets:** Most datasets lack early-stage smoke (small plumes), night conditions, and fog/cloud confounders -- exactly the gaps PyroNear2025 addresses.
- **Terrestrial perspective datasets are most relevant:** Aerial/UAV datasets (FLAME series) show different viewpoints and fire stages than Pyronear's fixed tower cameras. Prioritize terrestrial datasets for training.
- **Video datasets are scarce:** Only 13 of 29 datasets include video, confirming PyroNear2025-V's value as a rare video resource for temporal model training.
- **Synthetic data is emerging:** FLAME-SD and FireFly show synthetic generation is viable. Pyronear already uses Blender for synthetic smoke; the review confirms this is a valid approach.

## Takeaways for Implementation

1. **Prioritize these datasets for training augmentation** (in order of relevance to Pyronear):
   - D-Fire (mixed, bounding boxes, large)
   - DFS (multi-class, video, recent)
   - FD-Dataset (large, video)
   - FESB MLID (Mediterranean, camera+sensor, relevant geography)
2. **Avoid aerial-only datasets** (FLAME1/2/3, BA-UAV) for primary training -- perspective mismatch with tower-mounted cameras. Use them for diversity only.
3. **Cross-dataset evaluation is essential:** The review shows models overfit to source datasets. Always evaluate on held-out data from different sources/geographies.
4. **Data format standardization:** Different datasets use different formats (YOLO, COCO, VOC). Build a unified data loading pipeline.
5. **Gap analysis for PyroNear2025:** The review confirms that PyroNear2025 fills critical gaps (early smoke, small bboxes, video, multi-country), validating the dataset effort.
6. **Domain adaptation techniques** should be explored for cross-dataset training, as the review highlights poor generalization across domains.
