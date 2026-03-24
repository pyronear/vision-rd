# Ultra-Lightweight Convolution-Transformer Network for Early Fire Smoke Detection

- **Authors:** Shubhangi Chaturvedi, Chandravanshi Shubham Arun, Poornima Singh Thakur, Pritee Khanna, Aparajita Ojha
- **Year/Venue:** 2024, Fire Ecology 20:83 (Springer Open)
- **PDF:** `/mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/papers/pdfs/2024-Ultra-Lightweight-Conv-Transformer-Fire-Smoke-Detection.pdf`

## Key Idea

A hybrid dual-path classification network combining CNN and Vision Transformer (ViT) paths for fire smoke detection in satellite/remote sensing imagery, achieving >99% accuracy on multiple datasets with only 0.6M parameters and 0.4 GFLOPs. The model classifies images into four categories (clear, foggy, smokey, foggy-and-smokey) and is designed for resource-constrained IoT deployment. The combination of CNN's local feature learning with ViT's global self-attention captures smoke characteristics more effectively than either alone.

## Architecture / Content

```
Input Image (224x224x3)
         |
   MobileNetV2 Backbone
   (3 pretrained bottleneck blocks)
         |
   Output: 56x56x144
         |
    +-----+-----+
    |             |
  CNN Path      ViT Path
    |             |
  RBC x2        7x7 patches -> flatten
  RBDC x2       4 Transformer blocks
  GAP            (4 MHA + MLP each)
  Dense(64,32,16) Flatten
    |             Dense(64,32,16)
    +-----+-----+
          |
      Add/Fuse
          |
      Dense(16,8)
          |
      Softmax (4 classes)

RBC  = Residual Block with Conv (3x3)
RBDC = Residual Block with Depthwise Conv
MHA  = Multi-Head Attention (4 heads)

Total: 0.6M parameters, 0.4 GFLOPs
```

**Datasets used (all classification, 4 categories):**
- IIITDMJ_Smoke: 23,644 MODIS satellite images
- USTC_SmokeRS: 4,059 MODIS satellite images (256x256)
- Khan et al. (2019): 72,012 CCTV images
- He et al. (2021): 33,710 CCTV images

## Results

### Best model (ViT + RBC + RBDC combined):
| Dataset | Accuracy | Precision | Recall | F1 | FAR |
|---|---|---|---|---|---|
| IIITDMJ_Smoke | 99.62% | 99.66% | 99.62% | 99.64% | 0.125% |
| USTC_SmokeRS | 93.90% | 93.87% | 93.41% | 93.64% | 2.030% |
| Khan et al. | 99.94% | 99.97% | 99.97% | 99.94% | 0.018% |
| He et al. | 99.91% | 99.91% | 99.91% | 99.91% | 0.029% |

### Ablation study shows both paths needed:
- ViT alone: 99.03% on IIITDMJ_Smoke, but FAR=0.335%
- CNN alone (RBC+RBDC): 97.21%, FAR=0.909%
- Combined: 99.62%, FAR=0.125% (best of both)

### Comparison with SOTA:
- Outperforms 7 prior methods (VGG16, MobileNetV2, EfficientNet, etc.)
- Achieves this with only 0.6M params vs. 46.7M (VGG16) or 134M (Khan 2019)
- Detects smoke covering just 2% of satellite image area

## Applicability to Pyronear

**Moderate applicability -- architecture ideas relevant, but task differs.**

- **Classification vs. Detection:** This paper solves a 4-class classification problem (is smoke present in the image?), not object detection (where is the smoke?). Pyronear needs bounding boxes for downstream verification, so this model cannot directly replace YOLOv8.
- **Dual-path CNN+ViT concept is transferable:** The idea of combining CNN (local features) and ViT (global context) in parallel paths could be adapted for a lightweight classification head on the server side to verify YOLO detections.
- **0.6M parameters is very small:** This model could potentially run on a RaspPi as a binary "is there smoke anywhere in this image?" pre-screener before the heavier YOLO runs. However, YOLO-small is already designed for this.
- **Satellite imagery focus:** The datasets are mostly satellite (MODIS) and CCTV images, which differ from Pyronear's fixed watchtower camera perspective. The USTC_SmokeRS lower accuracy (93.9%) shows generalization challenges.
- **FAR reduction is relevant:** The dual-path approach consistently reduces False Alarm Rate compared to single-path, which aligns with Pyronear's goal of minimizing FP.

## Takeaways for Implementation

1. **Dual-path (CNN+ViT) fusion** reduces false alarm rates significantly. Consider this architecture for the server-side verification model if moving beyond ResNet+LSTM.
2. **MobileNetV2 backbone + lightweight heads** achieves excellent efficiency (0.6M params). If Pyronear needs a binary "smoke/no-smoke" classifier on edge, this architecture is a strong candidate.
3. **Depthwise separable convolutions** (RBDC blocks) are key to keeping the model lightweight -- same principle used in YOLOv8/v10 efficient designs.
4. **The combination of local (CNN) and global (ViT) features** is particularly effective for smoke, which is amorphous and lacks clear edges. This could improve the server-side temporal model.
5. **Cross-dataset validation is essential** -- the model was only tested on same-distribution data. Pyronear's diverse multi-country deployment requires cross-domain robustness.
