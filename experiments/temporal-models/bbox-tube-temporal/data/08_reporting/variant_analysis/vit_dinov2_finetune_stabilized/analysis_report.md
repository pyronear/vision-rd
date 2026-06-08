# Variant analysis report

Target: P >= 0.93 and R >= 0.95

## 1. Baseline (max-logit aggregation)

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] all tubes, max                          |  0.9329 |  0.9623 |  0.9474 |  153 |   11 |    6 |  |
| [train] all tubes, max                        |  0.9168 |  0.9581 |  0.9370 | 1487 |  135 |   65 |  |

## 2. Training-label confidence floor

- Detections scanned: 2198
- Min: 0.1001, P01: 0.1044, Median: 0.4847

## 3. Confidence filter simulation

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] conf>=0.05                              |  0.9329 |  0.9623 |  0.9474 |  153 |   11 |    6 | smoke_drop=0 fp_drop=0 |
| [train] conf>=0.05                            |  0.9168 |  0.9581 |  0.9370 | 1487 |  135 |   65 | smoke_drop=0 fp_drop=0 |
| [val] conf>=0.10                              |  0.9329 |  0.9623 |  0.9474 |  153 |   11 |    6 | smoke_drop=0 fp_drop=0 |
| [train] conf>=0.10                            |  0.9168 |  0.9581 |  0.9370 | 1487 |  135 |   65 | smoke_drop=0 fp_drop=0 |
| [val] conf>=0.15                              |  0.9325 |  0.9560 |  0.9441 |  152 |   11 |    7 | smoke_drop=2 fp_drop=3 |
| [train] conf>=0.15                            |  0.9235 |  0.9568 |  0.9399 | 1485 |  123 |   67 | smoke_drop=1 fp_drop=33 |
| [val] conf>=0.20                              |  0.9379 |  0.9497 |  0.9437 |  151 |   10 |    8 | smoke_drop=3 fp_drop=6 |
| [train] conf>=0.20                            |  0.9286 |  0.9549 |  0.9416 | 1482 |  114 |   70 | smoke_drop=5 fp_drop=58 |
| [val] conf>=0.25                              |  0.9497 |  0.9497 |  0.9497 |  151 |    8 |    8 | smoke_drop=3 fp_drop=11 |
| [train] conf>=0.25                            |  0.9343 |  0.9523 |  0.9432 | 1478 |  104 |   74 | smoke_drop=8 fp_drop=77 |

## 4. Tube selection sweep

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] all                                     |  0.9329 |  0.9623 |  0.9474 |  153 |   11 |    6 |  |
| [train] all                                   |  0.9168 |  0.9581 |  0.9370 | 1487 |  135 |   65 |  |
| [val] top-1                                   |  0.9557 |  0.9497 |  0.9527 |  151 |    7 |    8 |  |
| [train] top-1                                 |  0.9351 |  0.9472 |  0.9411 | 1470 |  102 |   82 |  |
| [val] top-2                                   |  0.9387 |  0.9623 |  0.9503 |  153 |   10 |    6 |  |
| [train] top-2                                 |  0.9235 |  0.9562 |  0.9395 | 1484 |  123 |   68 |  |
| [val] top-3                                   |  0.9387 |  0.9623 |  0.9503 |  153 |   10 |    6 |  |
| [train] top-3                                 |  0.9190 |  0.9581 |  0.9382 | 1487 |  131 |   65 |  |

## 5. Aggregation rule sweep

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] agg=max                                 |  0.9329 |  0.9623 |  0.9474 |  153 |   11 |    6 |  |
| [train] agg=max                               |  0.9168 |  0.9581 |  0.9370 | 1487 |  135 |   65 |  |
| [val] agg=mean                                |  0.9441 |  0.9560 |  0.9500 |  152 |    9 |    7 |  |
| [train] agg=mean                              |  0.9407 |  0.9401 |  0.9404 | 1459 |   92 |   93 |  |
| [val] agg=length_weighted_mean                |  0.9560 |  0.9560 |  0.9560 |  152 |    7 |    7 |  |
| [train] agg=length_weighted_mean              |  0.9411 |  0.9465 |  0.9438 | 1469 |   92 |   83 |  |

## 6. Logistic calibration (fit on train)

Weights: logit=0.761, log_len=1.856, mean_conf=2.717, n_tubes=-0.269, intercept=-5.968

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] logistic thr=0.40                       |  0.9560 |  0.9560 |  0.9560 |  152 |    7 |    7 |  |
| [train] logistic thr=0.40                     |  0.9403 |  0.9749 |  0.9573 | 1513 |   96 |   39 |  |
| [val] logistic thr=0.50                       |  0.9682 |  0.9560 |  0.9620 |  152 |    5 |    7 |  |
| [train] logistic thr=0.50                     |  0.9489 |  0.9684 |  0.9585 | 1503 |   81 |   49 |  |
| [val] logistic thr=0.60                       |  0.9682 |  0.9560 |  0.9620 |  152 |    5 |    7 |  |
| [train] logistic thr=0.60                     |  0.9587 |  0.9568 |  0.9578 | 1485 |   64 |   67 |  |
| [val] logistic thr=0.70                       |  0.9742 |  0.9497 |  0.9618 |  151 |    4 |    8 |  |
| [train] logistic thr=0.70                     |  0.9676 |  0.9427 |  0.9550 | 1463 |   49 |   89 |  |

## 7. Recommendation

**Target cleared** by **logistic thr=0.50**: P=0.9682 R=0.9560 F1=0.9620

Top 5 configs by val F1:

| rank | config | P | R | F1 |
|---|---|---|---|---|
| 1 | ** logistic thr=0.50** | 0.9682 | 0.9560 | 0.9620 |
| 2 | ** logistic thr=0.60** | 0.9682 | 0.9560 | 0.9620 |
| 3 | logistic thr=0.70 | 0.9742 | 0.9497 | 0.9618 |
| 4 | ** agg=length_weighted_mean** | 0.9560 | 0.9560 | 0.9560 |
| 5 | ** logistic thr=0.40** | 0.9560 | 0.9560 | 0.9560 |
