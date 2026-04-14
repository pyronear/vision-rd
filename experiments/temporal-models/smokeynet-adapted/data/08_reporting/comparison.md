# Variant comparison

> A variant must beat the baseline mean by more than the seed-to-seed spread on FP count at target recall to count as signal. The `train_gru`, `train_gru_seed43`, and `train_gru_seed44` rows below provide that spread.

| variant | F1 @ 0.5 | PR-AUC | ROC-AUC | FP @ recall 0.90 | FP @ recall 0.95 | FP @ recall 0.97 | FP @ recall 0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru | 0.937 | 0.976 | 0.981 | 8 | 13 | 16 | 16 |
| gru_seed43 | 0.942 | 0.968 | 0.976 | 10 | 11 | 15 | 21 |
| gru_seed44 | 0.929 | 0.972 | 0.976 | 9 | 15 | 19 | 32 |
| gru_convnext | 0.953 | 0.982 | 0.985 | 7 | 9 | 9 | 17 |
| gru_finetune | 0.918 | 0.962 | 0.969 | 10 | 17 | 18 | 38 |
| gru_convnext_finetune | 0.964 | 0.989 | 0.991 | 4 | 6 | 7 | 13 |
