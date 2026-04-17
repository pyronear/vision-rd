# Variant analysis report

Target: P >= 0.93 and R >= 0.95

## 1. Baseline (max-logit aggregation)

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] all tubes, max                          |  0.9560 |  0.9560 |  0.9560 |  152 |    7 |    7 |  |
| [train] all tubes, max                        |  0.9252 |  0.9716 |  0.9478 | 1508 |  122 |   44 |  |

## 2. Training-label confidence floor

- Detections scanned: 2198
- Min: 0.1001, P01: 0.1044, Median: 0.4847

## 3. Confidence filter simulation

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] conf>=0.05                              |  0.9560 |  0.9560 |  0.9560 |  152 |    7 |    7 | smoke_drop=0 fp_drop=0 |
| [train] conf>=0.05                            |  0.9252 |  0.9716 |  0.9478 | 1508 |  122 |   44 | smoke_drop=0 fp_drop=0 |
| [val] conf>=0.10                              |  0.9560 |  0.9560 |  0.9560 |  152 |    7 |    7 | smoke_drop=0 fp_drop=0 |
| [train] conf>=0.10                            |  0.9252 |  0.9716 |  0.9478 | 1508 |  122 |   44 | smoke_drop=0 fp_drop=0 |
| [val] conf>=0.15                              |  0.9557 |  0.9497 |  0.9527 |  151 |    7 |    8 | smoke_drop=2 fp_drop=3 |
| [train] conf>=0.15                            |  0.9308 |  0.9710 |  0.9505 | 1507 |  112 |   45 | smoke_drop=1 fp_drop=33 |
| [val] conf>=0.20                              |  0.9554 |  0.9434 |  0.9494 |  150 |    7 |    9 | smoke_drop=3 fp_drop=6 |
| [train] conf>=0.20                            |  0.9359 |  0.9691 |  0.9522 | 1504 |  103 |   48 | smoke_drop=5 fp_drop=58 |
| [val] conf>=0.25                              |  0.9615 |  0.9434 |  0.9524 |  150 |    6 |    9 | smoke_drop=3 fp_drop=11 |
| [train] conf>=0.25                            |  0.9410 |  0.9665 |  0.9536 | 1500 |   94 |   52 | smoke_drop=8 fp_drop=77 |

## 4. Tube selection sweep

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] all                                     |  0.9560 |  0.9560 |  0.9560 |  152 |    7 |    7 |  |
| [train] all                                   |  0.9252 |  0.9716 |  0.9478 | 1508 |  122 |   44 |  |
| [val] top-1                                   |  0.9728 |  0.8994 |  0.9346 |  143 |    4 |   16 |  |
| [train] top-1                                 |  0.9473 |  0.9607 |  0.9539 | 1491 |   83 |   61 |  |
| [val] top-2                                   |  0.9618 |  0.9497 |  0.9557 |  151 |    6 |    8 |  |
| [train] top-2                                 |  0.9314 |  0.9710 |  0.9508 | 1507 |  111 |   45 |  |
| [val] top-3                                   |  0.9557 |  0.9497 |  0.9527 |  151 |    7 |    8 |  |
| [train] top-3                                 |  0.9269 |  0.9716 |  0.9487 | 1508 |  119 |   44 |  |

## 5. Aggregation rule sweep

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] agg=max                                 |  0.9560 |  0.9560 |  0.9560 |  152 |    7 |    7 |  |
| [train] agg=max                               |  0.9252 |  0.9716 |  0.9478 | 1508 |  122 |   44 |  |
| [val] agg=mean                                |  0.9733 |  0.9182 |  0.9450 |  146 |    4 |   13 |  |
| [train] agg=mean                              |  0.9565 |  0.9497 |  0.9531 | 1474 |   67 |   78 |  |
| [val] agg=length_weighted_mean                |  0.9795 |  0.8994 |  0.9377 |  143 |    3 |   16 |  |
| [train] agg=length_weighted_mean              |  0.9528 |  0.9626 |  0.9577 | 1494 |   74 |   58 |  |

## 6. Logistic calibration (fit on train)

Weights: logit=0.647, log_len=1.957, mean_conf=2.588, n_tubes=-0.013, intercept=-6.354

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] logistic thr=0.40                       |  0.9737 |  0.9308 |  0.9518 |  148 |    4 |   11 |  |
| [train] logistic thr=0.40                     |  0.9530 |  0.9794 |  0.9660 | 1520 |   75 |   32 |  |
| [val] logistic thr=0.50                       |  0.9735 |  0.9245 |  0.9484 |  147 |    4 |   12 |  |
| [train] logistic thr=0.50                     |  0.9620 |  0.9781 |  0.9700 | 1518 |   60 |   34 |  |
| [val] logistic thr=0.60                       |  0.9863 |  0.9057 |  0.9443 |  144 |    2 |   15 |  |
| [train] logistic thr=0.60                     |  0.9665 |  0.9678 |  0.9672 | 1502 |   52 |   50 |  |
| [val] logistic thr=0.70                       |  0.9859 |  0.8805 |  0.9302 |  140 |    2 |   19 |  |
| [train] logistic thr=0.70                     |  0.9744 |  0.9562 |  0.9652 | 1484 |   39 |   68 |  |

## 7. Recommendation

**Target cleared** by **baseline**: P=0.9560 R=0.9560 F1=0.9560

Top 5 configs by val F1:

| rank | config | P | R | F1 |
|---|---|---|---|---|
| 1 | ** baseline** | 0.9560 | 0.9560 | 0.9560 |
| 2 | ** conf>=0.05** | 0.9560 | 0.9560 | 0.9560 |
| 3 | ** conf>=0.10** | 0.9560 | 0.9560 | 0.9560 |
| 4 | ** sel=all** | 0.9560 | 0.9560 | 0.9560 |
| 5 | ** agg=max** | 0.9560 | 0.9560 | 0.9560 |
