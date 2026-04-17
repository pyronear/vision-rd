# Variant analysis report

Target: P >= 0.93 and R >= 0.95

## 1. Baseline (max-logit aggregation)

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] all tubes, max                          |  0.8947 |  0.9623 |  0.9273 |  153 |   18 |    6 |  |
| [train] all tubes, max                        |  0.8897 |  0.9723 |  0.9292 | 1509 |  187 |   43 |  |

## 2. Training-label confidence floor

- Detections scanned: 2198
- Min: 0.1001, P01: 0.1044, Median: 0.4847

## 3. Confidence filter simulation

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] conf>=0.05                              |  0.8947 |  0.9623 |  0.9273 |  153 |   18 |    6 | smoke_drop=0 fp_drop=0 |
| [train] conf>=0.05                            |  0.8897 |  0.9723 |  0.9292 | 1509 |  187 |   43 | smoke_drop=0 fp_drop=0 |
| [val] conf>=0.10                              |  0.8947 |  0.9623 |  0.9273 |  153 |   18 |    6 | smoke_drop=0 fp_drop=0 |
| [train] conf>=0.10                            |  0.8897 |  0.9723 |  0.9292 | 1509 |  187 |   43 | smoke_drop=0 fp_drop=0 |
| [val] conf>=0.15                              |  0.8941 |  0.9560 |  0.9240 |  152 |   18 |    7 | smoke_drop=2 fp_drop=3 |
| [train] conf>=0.15                            |  0.8949 |  0.9710 |  0.9314 | 1507 |  177 |   45 | smoke_drop=1 fp_drop=33 |
| [val] conf>=0.20                              |  0.9096 |  0.9497 |  0.9292 |  151 |   15 |    8 | smoke_drop=3 fp_drop=6 |
| [train] conf>=0.20                            |  0.9016 |  0.9684 |  0.9338 | 1503 |  164 |   49 | smoke_drop=5 fp_drop=58 |
| [val] conf>=0.25                              |  0.9264 |  0.9497 |  0.9379 |  151 |   12 |    8 | smoke_drop=3 fp_drop=11 |
| [train] conf>=0.25                            |  0.9084 |  0.9652 |  0.9360 | 1498 |  151 |   54 | smoke_drop=8 fp_drop=77 |

## 4. Tube selection sweep

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] all                                     |  0.8947 |  0.9623 |  0.9273 |  153 |   18 |    6 |  |
| [train] all                                   |  0.8897 |  0.9723 |  0.9292 | 1509 |  187 |   43 |  |
| [val] top-1                                   |  0.9317 |  0.9434 |  0.9375 |  150 |   11 |    9 |  |
| [train] top-1                                 |  0.9137 |  0.9613 |  0.9369 | 1492 |  141 |   60 |  |
| [val] top-2                                   |  0.9107 |  0.9623 |  0.9358 |  153 |   15 |    6 |  |
| [train] top-2                                 |  0.8960 |  0.9710 |  0.9320 | 1507 |  175 |   45 |  |
| [val] top-3                                   |  0.9053 |  0.9623 |  0.9329 |  153 |   16 |    6 |  |
| [train] top-3                                 |  0.8907 |  0.9710 |  0.9291 | 1507 |  185 |   45 |  |

## 5. Aggregation rule sweep

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] agg=max                                 |  0.8947 |  0.9623 |  0.9273 |  153 |   18 |    6 |  |
| [train] agg=max                               |  0.8897 |  0.9723 |  0.9292 | 1509 |  187 |   43 |  |
| [val] agg=mean                                |  0.9317 |  0.9434 |  0.9375 |  150 |   11 |    9 |  |
| [train] agg=mean                              |  0.9182 |  0.9549 |  0.9362 | 1482 |  132 |   70 |  |
| [val] agg=length_weighted_mean                |  0.9379 |  0.9497 |  0.9437 |  151 |   10 |    8 |  |
| [train] agg=length_weighted_mean              |  0.9198 |  0.9607 |  0.9398 | 1491 |  130 |   61 |  |

## 6. Platt re-calibration (fit on train)

Weights: logit=0.670, log_len=1.692, mean_conf=2.685, n_tubes=0.059, intercept=-6.364

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] platt thr=0.40                          |  0.9620 |  0.9560 |  0.9590 |  152 |    6 |    7 |  |
| [train] platt thr=0.40                        |  0.9293 |  0.9736 |  0.9509 | 1511 |  115 |   41 |  |
| [val] platt thr=0.50                          |  0.9742 |  0.9497 |  0.9618 |  151 |    4 |    8 |  |
| [train] platt thr=0.50                        |  0.9398 |  0.9665 |  0.9530 | 1500 |   96 |   52 |  |
| [val] platt thr=0.60                          |  0.9803 |  0.9371 |  0.9582 |  149 |    3 |   10 |  |
| [train] platt thr=0.60                        |  0.9518 |  0.9536 |  0.9527 | 1480 |   75 |   72 |  |
| [val] platt thr=0.70                          |  0.9799 |  0.9182 |  0.9481 |  146 |    3 |   13 |  |
| [train] platt thr=0.70                        |  0.9641 |  0.9356 |  0.9496 | 1452 |   54 |  100 |  |

## 7. Recommendation

**Target cleared** by **platt thr=0.40**: P=0.9620 R=0.9560 F1=0.9590

Top 5 configs by val F1:

| rank | config | P | R | F1 |
|---|---|---|---|---|
| 1 | platt thr=0.50 | 0.9742 | 0.9497 | 0.9618 |
| 2 | ** platt thr=0.40** | 0.9620 | 0.9560 | 0.9590 |
| 3 | platt thr=0.60 | 0.9803 | 0.9371 | 0.9582 |
| 4 | platt thr=0.70 | 0.9799 | 0.9182 | 0.9481 |
| 5 | agg=length_weighted_mean | 0.9379 | 0.9497 | 0.9437 |
