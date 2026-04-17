# Variant analysis report

Target: P >= 0.93 and R >= 0.95

## 1. Baseline (max-logit aggregation)

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] all tubes, max                          |  0.8844 |  0.9623 |  0.9217 |  153 |   20 |    6 |  |
| [train] all tubes, max                        |  0.8696 |  0.9755 |  0.9195 | 1514 |  227 |   38 |  |

## 2. Training-label confidence floor

- Detections scanned: 2198
- Min: 0.1001, P01: 0.1044, Median: 0.4847

## 3. Confidence filter simulation

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] conf>=0.05                              |  0.8844 |  0.9623 |  0.9217 |  153 |   20 |    6 | smoke_drop=0 fp_drop=0 |
| [train] conf>=0.05                            |  0.8696 |  0.9755 |  0.9195 | 1514 |  227 |   38 | smoke_drop=0 fp_drop=0 |
| [val] conf>=0.10                              |  0.8844 |  0.9623 |  0.9217 |  153 |   20 |    6 | smoke_drop=0 fp_drop=0 |
| [train] conf>=0.10                            |  0.8696 |  0.9755 |  0.9195 | 1514 |  227 |   38 | smoke_drop=0 fp_drop=0 |
| [val] conf>=0.15                              |  0.8889 |  0.9560 |  0.9212 |  152 |   19 |    7 | smoke_drop=2 fp_drop=2 |
| [train] conf>=0.15                            |  0.8805 |  0.9729 |  0.9244 | 1510 |  205 |   42 | smoke_drop=1 fp_drop=45 |
| [val] conf>=0.20                              |  0.9096 |  0.9497 |  0.9292 |  151 |   15 |    8 | smoke_drop=3 fp_drop=5 |
| [train] conf>=0.20                            |  0.8880 |  0.9704 |  0.9273 | 1506 |  190 |   46 | smoke_drop=5 fp_drop=80 |
| [val] conf>=0.25                              |  0.9207 |  0.9497 |  0.9350 |  151 |   13 |    8 | smoke_drop=3 fp_drop=11 |
| [train] conf>=0.25                            |  0.8988 |  0.9671 |  0.9317 | 1501 |  169 |   51 | smoke_drop=8 fp_drop=104 |

## 4. Tube selection sweep

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] all                                     |  0.8844 |  0.9623 |  0.9217 |  153 |   20 |    6 |  |
| [train] all                                   |  0.8696 |  0.9755 |  0.9195 | 1514 |  227 |   38 |  |
| [val] top-1                                   |  0.9375 |  0.9434 |  0.9404 |  150 |   10 |    9 |  |
| [train] top-1                                 |  0.9016 |  0.9620 |  0.9308 | 1493 |  163 |   59 |  |
| [val] top-2                                   |  0.9053 |  0.9623 |  0.9329 |  153 |   16 |    6 |  |
| [train] top-2                                 |  0.8790 |  0.9736 |  0.9239 | 1511 |  208 |   41 |  |
| [val] top-3                                   |  0.8947 |  0.9623 |  0.9273 |  153 |   18 |    6 |  |
| [train] top-3                                 |  0.8715 |  0.9742 |  0.9200 | 1512 |  223 |   40 |  |

## 5. Aggregation rule sweep

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] agg=max                                 |  0.8844 |  0.9623 |  0.9217 |  153 |   20 |    6 |  |
| [train] agg=max                               |  0.8696 |  0.9755 |  0.9195 | 1514 |  227 |   38 |  |
| [val] agg=mean                                |  0.9379 |  0.9497 |  0.9437 |  151 |   10 |    8 |  |
| [train] agg=mean                              |  0.9095 |  0.9581 |  0.9332 | 1487 |  148 |   65 |  |
| [val] agg=length_weighted_mean                |  0.9441 |  0.9560 |  0.9500 |  152 |    9 |    7 |  |
| [train] agg=length_weighted_mean              |  0.9070 |  0.9620 |  0.9337 | 1493 |  153 |   59 |  |

## 6. Platt re-calibration (fit on train)

Weights: logit=0.691, log_len=1.673, mean_conf=2.450, n_tubes=-0.013, intercept=-6.173

| experiment | P | R | F1 | TP | FP | FN | notes |
|---|---|---|---|---|---|---|---|
| [val] platt thr=0.40                          |  0.9742 |  0.9497 |  0.9618 |  151 |    4 |    8 |  |
| [train] platt thr=0.40                        |  0.9315 |  0.9723 |  0.9515 | 1509 |  111 |   43 |  |
| [val] platt thr=0.50                          |  0.9804 |  0.9434 |  0.9615 |  150 |    3 |    9 |  |
| [train] platt thr=0.50                        |  0.9390 |  0.9626 |  0.9507 | 1494 |   97 |   58 |  |
| [val] platt thr=0.60                          |  0.9866 |  0.9245 |  0.9545 |  147 |    2 |   12 |  |
| [train] platt thr=0.60                        |  0.9540 |  0.9485 |  0.9512 | 1472 |   71 |   80 |  |
| [val] platt thr=0.70                          |  0.9931 |  0.9057 |  0.9474 |  144 |    1 |   15 |  |
| [train] platt thr=0.70                        |  0.9639 |  0.9291 |  0.9462 | 1442 |   54 |  110 |  |

## 7. Recommendation

**Target cleared** by **agg=length_weighted_mean**: P=0.9441 R=0.9560 F1=0.9500

Top 5 configs by val F1:

| rank | config | P | R | F1 |
|---|---|---|---|---|
| 1 | platt thr=0.40 | 0.9742 | 0.9497 | 0.9618 |
| 2 | platt thr=0.50 | 0.9804 | 0.9434 | 0.9615 |
| 3 | platt thr=0.60 | 0.9866 | 0.9245 | 0.9545 |
| 4 | ** agg=length_weighted_mean** | 0.9441 | 0.9560 | 0.9500 |
| 5 | platt thr=0.70 | 0.9931 | 0.9057 | 0.9474 |
