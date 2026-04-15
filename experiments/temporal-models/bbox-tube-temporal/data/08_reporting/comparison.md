# Variant comparison

> A variant must beat the baseline mean by more than the seed-to-seed spread on FP count at target recall to count as signal. The `train_gru`, `train_gru_seed43`, and `train_gru_seed44` rows below provide that spread.

| variant | F1 @ 0.5 | PR-AUC | ROC-AUC | FP @ recall 0.90 | FP @ recall 0.95 | FP @ recall 0.97 | FP @ recall 0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru | 0.930 | 0.978 | 0.980 | 7 | 14 | 20 | 26 |
| gru_convnext | 0.959 | 0.989 | 0.991 | 4 | 5 | 5 | 16 |
| gru_finetune | 0.915 | 0.971 | 0.976 | 13 | 18 | 20 | 27 |
| gru_convnext_finetune | 0.963 | 0.993 | 0.994 | 2 | 4 | 4 | 18 |
| gru_convnext_base_finetune | 0.967 | 0.989 | 0.991 | 3 | 5 | 6 | 10 |
| vit_dinov2_frozen | 0.960 | 0.993 | 0.994 | 2 | 6 | 9 | 13 |
| vit_dinov2_finetune | 0.971 | 0.991 | 0.993 | 5 | 5 | 6 | 6 |
| vit_in21k_finetune | 0.960 | 0.989 | 0.991 | 5 | 6 | 7 | 10 |
