# Aggregation-rule ablation

Target recall for threshold search: **0.95**.

One threshold is chosen per (variant, split, rule) to hit the target recall;
precision/FPR/etc. are reported at that threshold.

| variant | split | rule | k | threshold | precision | recall | F1 | FPR | TP | FP | FN | TN |
|---------|-------|------|---|-----------|-----------|--------|----|----|----|----|----|----|
| gru_convnext_finetune | train | max | 1 | 1.0280 | 0.8816 | 0.9504 | 0.9147 | 0.1276 | 1475 | 198 | 77 | 1354 |
| gru_convnext_finetune | train | top_k_mean | 2 | -inf | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 1552 | 1552 | 0 | 0 |
| gru_convnext_finetune | train | top_k_mean | 3 | -inf | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 1552 | 1552 | 0 | 0 |
| gru_convnext_finetune | val | max | 1 | 0.2431 | 0.8588 | 0.9560 | 0.9048 | 0.1572 | 152 | 25 | 7 | 134 |
| gru_convnext_finetune | val | top_k_mean | 2 | -inf | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 159 | 159 | 0 | 0 |
| gru_convnext_finetune | val | top_k_mean | 3 | -inf | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 159 | 159 | 0 | 0 |
| vit_dinov2_finetune | train | max | 1 | 1.5495 | 0.8409 | 0.9504 | 0.8923 | 0.1798 | 1475 | 279 | 77 | 1273 |
| vit_dinov2_finetune | train | top_k_mean | 2 | -inf | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 1552 | 1552 | 0 | 0 |
| vit_dinov2_finetune | train | top_k_mean | 3 | -inf | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 1552 | 1552 | 0 | 0 |
| vit_dinov2_finetune | val | max | 1 | 1.5322 | 0.8216 | 0.9560 | 0.8837 | 0.2075 | 152 | 33 | 7 | 126 |
| vit_dinov2_finetune | val | top_k_mean | 2 | -inf | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 159 | 159 | 0 | 0 |
| vit_dinov2_finetune | val | top_k_mean | 3 | -inf | 0.5000 | 1.0000 | 0.6667 | 1.0000 | 159 | 159 | 0 | 0 |
