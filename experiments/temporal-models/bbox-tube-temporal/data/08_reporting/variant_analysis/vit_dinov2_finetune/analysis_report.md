# Variant analysis report

Target: P >= 0.93 and R >= 0.95

## 1. Baseline (max-logit aggregation)

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] all tubes, max                          |  0.8895 |  0.9623 |  0.9245 |  153 |   19 |    6 |  |
| [train] all tubes, max                        |  0.8790 |  0.9781 |  0.9259 | 1518 |  209 |   34 |  |

## 2. Training-label confidence floor

- Detections scanned: 2198
- Min: 0.1001, P01: 0.1044, Median: 0.4847

## 3. Confidence filter simulation

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] conf>=0.05                              |  0.8895 |  0.9623 |  0.9245 |  153 |   19 |    6 | smoke_drop=0 fp_drop=0 |
| [train] conf>=0.05                            |  0.8790 |  0.9781 |  0.9259 | 1518 |  209 |   34 | smoke_drop=0 fp_drop=0 |
| [val] conf>=0.10                              |  0.8895 |  0.9623 |  0.9245 |  153 |   19 |    6 | smoke_drop=0 fp_drop=0 |
| [train] conf>=0.10                            |  0.8790 |  0.9781 |  0.9259 | 1518 |  209 |   34 | smoke_drop=0 fp_drop=0 |
| [val] conf>=0.15                              |  0.8889 |  0.9560 |  0.9212 |  152 |   19 |    7 | smoke_drop=2 fp_drop=3 |
| [train] conf>=0.15                            |  0.8840 |  0.9768 |  0.9281 | 1516 |  199 |   36 | smoke_drop=1 fp_drop=33 |
| [val] conf>=0.20                              |  0.9042 |  0.9497 |  0.9264 |  151 |   16 |    8 | smoke_drop=3 fp_drop=6 |
| [train] conf>=0.20                            |  0.8915 |  0.9742 |  0.9310 | 1512 |  184 |   40 | smoke_drop=5 fp_drop=58 |
| [val] conf>=0.25                              |  0.9207 |  0.9497 |  0.9350 |  151 |   13 |    8 | smoke_drop=3 fp_drop=11 |
| [train] conf>=0.25                            |  0.8992 |  0.9710 |  0.9337 | 1507 |  169 |   45 | smoke_drop=8 fp_drop=77 |

## 4. Tube selection sweep

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] all                                     |  0.8895 |  0.9623 |  0.9245 |  153 |   19 |    6 |  |
| [train] all                                   |  0.8790 |  0.9781 |  0.9259 | 1518 |  209 |   34 |  |
| [val] top-1                                   |  0.9259 |  0.9434 |  0.9346 |  150 |   12 |    9 |  |
| [train] top-1                                 |  0.9033 |  0.9691 |  0.9350 | 1504 |  161 |   48 |  |
| [val] top-2                                   |  0.9053 |  0.9623 |  0.9329 |  153 |   16 |    6 |  |
| [train] top-2                                 |  0.8850 |  0.9768 |  0.9286 | 1516 |  197 |   36 |  |
| [val] top-3                                   |  0.9000 |  0.9623 |  0.9301 |  153 |   17 |    6 |  |
| [train] top-3                                 |  0.8799 |  0.9768 |  0.9258 | 1516 |  207 |   36 |  |

## 5. Aggregation rule sweep

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] agg=max                                 |  0.8895 |  0.9623 |  0.9245 |  153 |   19 |    6 |  |
| [train] agg=max                               |  0.8790 |  0.9781 |  0.9259 | 1518 |  209 |   34 |  |
| [val] agg=mean                                |  0.9321 |  0.9497 |  0.9408 |  151 |   11 |    8 |  |
| [train] agg=mean                              |  0.9073 |  0.9646 |  0.9350 | 1497 |  153 |   55 |  |
| [val] agg=length_weighted_mean                |  0.9383 |  0.9560 |  0.9470 |  152 |   10 |    7 |  |
| [train] agg=length_weighted_mean              |  0.9053 |  0.9665 |  0.9349 | 1500 |  157 |   52 |  |

## 6. Logistic calibration (fit on train)

Weights: logit=0.670, log_len=1.692, mean_conf=2.685, n_tubes=0.059, intercept=-6.364

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] logistic thr=0.40                       |  0.9620 |  0.9560 |  0.9590 |  152 |    6 |    7 |  |
| [train] logistic thr=0.40                     |  0.9293 |  0.9736 |  0.9509 | 1511 |  115 |   41 |  |
| [val] logistic thr=0.50                       |  0.9742 |  0.9497 |  0.9618 |  151 |    4 |    8 |  |
| [train] logistic thr=0.50                     |  0.9398 |  0.9665 |  0.9530 | 1500 |   96 |   52 |  |
| [val] logistic thr=0.60                       |  0.9803 |  0.9371 |  0.9582 |  149 |    3 |   10 |  |
| [train] logistic thr=0.60                     |  0.9518 |  0.9536 |  0.9527 | 1480 |   75 |   72 |  |
| [val] logistic thr=0.70                       |  0.9799 |  0.9182 |  0.9481 |  146 |    3 |   13 |  |
| [train] logistic thr=0.70                     |  0.9641 |  0.9356 |  0.9496 | 1452 |   54 |  100 |  |

## 7. Recommendation

**Target cleared** by **logistic thr=0.40**: P=0.9620 R=0.9560 F1=0.9590

Top 5 configs by val F1:

| rank | config | P | R | F1 |
|---|---|---|---|---|
| 1 | logistic thr=0.50 | 0.9742 | 0.9497 | 0.9618 |
| 2 | ** logistic thr=0.40** | 0.9620 | 0.9560 | 0.9590 |
| 3 | logistic thr=0.60 | 0.9803 | 0.9371 | 0.9582 |
| 4 | logistic thr=0.70 | 0.9799 | 0.9182 | 0.9481 |
| 5 | ** agg=length_weighted_mean** | 0.9383 | 0.9560 | 0.9470 |
