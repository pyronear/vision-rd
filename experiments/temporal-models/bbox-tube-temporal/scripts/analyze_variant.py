"""Automated post-training analysis for bbox-tube-temporal variants.

Runs a battery of offline simulations on evaluate_packaged predictions
and produces a comprehensive report with recommended inference config.

Usage::

    uv run python scripts/analyze_variant.py \\
        --train-predictions \\
        data/08_reporting/train/packaged/gru_convnext_finetune/predictions.json \\
        --val-predictions \\
        data/08_reporting/val/packaged/gru_convnext_finetune/predictions.json \\
        --training-labels-dir data/01_raw/datasets/val/fp \\
        --output-dir data/08_reporting/variant_analysis/gru_convnext_finetune
"""

import argparse
import contextlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from bbox_tube_temporal.aggregation_analysis import load_predictions

CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]
TUBE_SELECTIONS = [("all", None), ("top-1", 1), ("top-2", 2), ("top-3", 3)]
AGGREGATION_RULES = ["max", "mean", "length_weighted_mean"]
PLATT_THRESHOLDS = [0.40, 0.50, 0.60, 0.70]


def tube_len(t: dict) -> int:
    return t["end_frame"] - t["start_frame"] + 1


def tube_mean_conf(t: dict) -> float:
    confs = [e["confidence"] for e in t["entries"] if e["confidence"] is not None]
    return mean(confs) if confs else 0.0


def tube_max_conf(t: dict) -> float:
    confs = [e["confidence"] for e in t["entries"] if e["confidence"] is not None]
    return max(confs) if confs else 0.0


def _evaluate(
    records: list[dict],
    *,
    filter_fn=None,
    score_fn=None,
) -> dict:
    tp = fp = fn = tn = 0
    for r in records:
        cls = r["label"]
        threshold = r["threshold"]
        kept = r["kept_tubes"]
        surviving = filter_fn(kept) if filter_fn and kept else (kept or [])
        if not surviving:
            if cls == "smoke":
                fn += 1
            else:
                tn += 1
            continue
        score = score_fn(surviving) if score_fn else max(t["logit"] for t in surviving)
        fires = score >= threshold
        if fires:
            if cls == "smoke":
                tp += 1
            else:
                fp += 1
        else:
            if cls == "smoke":
                fn += 1
            else:
                tn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r_ = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r_, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def scan_confidence_floor(labels_dir: Path) -> dict:
    confs: list[float] = []
    for lf in labels_dir.rglob("labels/*.txt"):
        for line in lf.read_text().splitlines():
            parts = line.split()
            if len(parts) == 6:
                with contextlib.suppress(ValueError):
                    confs.append(float(parts[5]))
    if not confs:
        return {"n_detections": 0, "min": None, "p01": None, "median": None}
    confs_sorted = sorted(confs)
    return {
        "n_detections": len(confs),
        "min": round(confs_sorted[0], 4),
        "p01": round(confs_sorted[max(0, len(confs) // 100)], 4),
        "median": round(confs_sorted[len(confs) // 2], 4),
    }


def simulate_confidence_filter(records: list[dict], tau: float) -> dict:
    def _filter(tubes):
        return [t for t in tubes if tube_max_conf(t) >= tau]

    result = _evaluate(records, filter_fn=_filter)

    smoke_recs = [r for r in records if r["label"] == "smoke"]
    fp_recs = [r for r in records if r["label"] == "fp"]
    smoke_all_drop = sum(
        1
        for r in smoke_recs
        if r["kept_tubes"] and all(tube_max_conf(t) < tau for t in r["kept_tubes"])
    )
    fp_all_drop = sum(
        1
        for r in fp_recs
        if r["kept_tubes"] and all(tube_max_conf(t) < tau for t in r["kept_tubes"])
    )
    result["smoke_all_drop"] = smoke_all_drop
    result["fp_all_drop"] = fp_all_drop
    return result


def simulate_tube_selection(records: list[dict], n: int | None) -> dict:
    if n is None:
        return _evaluate(records)

    def _filter(tubes):
        return sorted(tubes, key=lambda t: -tube_len(t))[:n]

    return _evaluate(records, filter_fn=_filter)


def simulate_aggregation(records: list[dict], rule: str) -> dict:
    if rule == "max":
        return _evaluate(records)
    if rule == "mean":
        return _evaluate(
            records,
            score_fn=lambda tubes: mean(t["logit"] for t in tubes),
        )
    if rule == "length_weighted_mean":

        def _lwm(tubes):
            total = sum(tube_len(t) for t in tubes)
            if total == 0:
                return -math.inf
            return sum(t["logit"] * tube_len(t) for t in tubes) / total

        return _evaluate(records, score_fn=_lwm)
    raise ValueError(f"unknown rule {rule!r}")


def fit_platt_model(
    train_records: list[dict],
) -> tuple[LogisticRegression, list[str]]:
    features = ["logit", "log_len", "mean_conf", "n_tubes"]
    X, y = [], []
    for r in train_records:
        label = 1 if r["label"] == "smoke" else 0
        kept = r["kept_tubes"]
        if not kept:
            X.append([0.0, 0.0, 0.0, 0.0])
        else:
            best = max(kept, key=lambda t: t["logit"])
            X.append(
                [
                    best["logit"],
                    math.log1p(tube_len(best)),
                    tube_mean_conf(best),
                    len(kept),
                ]
            )
        y.append(label)
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(np.array(X), np.array(y))
    return lr, features


def evaluate_platt(
    model: LogisticRegression,
    records: list[dict],
    threshold: float,
) -> dict:
    tp = fp = fn = tn = 0
    for r in records:
        cls = r["label"]
        kept = r["kept_tubes"]
        if not kept:
            features = [0.0, 0.0, 0.0, 0.0]
        else:
            best = max(kept, key=lambda t: t["logit"])
            features = [
                best["logit"],
                math.log1p(tube_len(best)),
                tube_mean_conf(best),
                len(kept),
            ]
        prob = model.predict_proba(np.array([features]))[0, 1]
        fires = prob >= threshold
        if fires:
            if cls == "smoke":
                tp += 1
            else:
                fp += 1
        else:
            if cls == "smoke":
                fn += 1
            else:
                tn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r_ = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r_, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _row(label: str, m: dict, extra: str = "") -> str:
    return (
        f"| {label:45s} | {m['precision']:7.4f} | {m['recall']:7.4f} "
        f"| {m['f1']:7.4f} | {m['tp']:>4d} | {m['fp']:>4d} "
        f"| {m['fn']:>4d} | {extra} |"
    )


def _table_header() -> str:
    return (
        "| experiment | P | R | F1 | TP | FP | FN | notes |\n"
        "|---|---|---|---|---|---|---|---|"
    )


def build_report(
    *,
    train_records: list[dict],
    val_records: list[dict],
    conf_floor: dict,
    target_p: float,
    target_r: float,
) -> tuple[str, list[dict]]:
    lines: list[str] = []
    all_configs: list[dict] = []

    lines.append("# Variant analysis report\n")
    lines.append(f"Target: P >= {target_p} and R >= {target_r}\n")

    # 1. Baseline
    lines.append("## 1. Baseline (max-logit aggregation)\n")
    lines.append(_table_header())
    for name, recs in [("val", val_records), ("train", train_records)]:
        m = _evaluate(recs)
        lines.append(_row(f"[{name}] all tubes, max", m))
        if name == "val":
            all_configs.append({"name": "baseline", "split": "val", **m})

    # 2. Confidence floor
    lines.append(
        f"\n## 2. Training-label confidence floor\n\n"
        f"- Detections scanned: {conf_floor['n_detections']}\n"
        f"- Min: {conf_floor['min']}, P01: {conf_floor['p01']}, "
        f"Median: {conf_floor['median']}\n"
    )

    # 3. Confidence filter sweep
    lines.append("## 3. Confidence filter simulation\n")
    lines.append(_table_header())
    for tau in CONF_THRESHOLDS:
        for name, recs in [("val", val_records), ("train", train_records)]:
            m = simulate_confidence_filter(recs, tau)
            extra = f"smoke_drop={m['smoke_all_drop']} fp_drop={m['fp_all_drop']}"
            lines.append(_row(f"[{name}] conf>={tau:.2f}", m, extra))
            if name == "val":
                all_configs.append({"name": f"conf>={tau:.2f}", "split": "val", **m})

    # 4. Tube selection sweep
    lines.append("\n## 4. Tube selection sweep\n")
    lines.append(_table_header())
    for sel_name, n in TUBE_SELECTIONS:
        for name, recs in [("val", val_records), ("train", train_records)]:
            m = simulate_tube_selection(recs, n)
            lines.append(_row(f"[{name}] {sel_name}", m))
            if name == "val":
                all_configs.append({"name": f"sel={sel_name}", "split": "val", **m})

    # 5. Aggregation rule sweep
    lines.append("\n## 5. Aggregation rule sweep\n")
    lines.append(_table_header())
    for rule in AGGREGATION_RULES:
        for name, recs in [("val", val_records), ("train", train_records)]:
            m = simulate_aggregation(recs, rule)
            lines.append(_row(f"[{name}] agg={rule}", m))
            if name == "val":
                all_configs.append({"name": f"agg={rule}", "split": "val", **m})

    # 6. Platt re-calibration
    lines.append("\n## 6. Platt re-calibration (fit on train)\n")
    platt_model, feature_names = fit_platt_model(train_records)
    coefs = platt_model.coef_[0]
    intercept = platt_model.intercept_[0]
    weights_str = ", ".join(
        f"{n}={c:.3f}" for n, c in zip(feature_names, coefs, strict=True)
    )
    lines.append(f"Weights: {weights_str}, intercept={intercept:.3f}\n")
    lines.append(_table_header())
    for thr in PLATT_THRESHOLDS:
        for name, recs in [("val", val_records), ("train", train_records)]:
            m = evaluate_platt(platt_model, recs, thr)
            lines.append(_row(f"[{name}] platt thr={thr:.2f}", m))
            if name == "val":
                all_configs.append(
                    {"name": f"platt thr={thr:.2f}", "split": "val", **m}
                )

    # 7. Recommendation
    lines.append("\n## 7. Recommendation\n")
    val_configs = [c for c in all_configs if c["split"] == "val"]
    passing = [
        c for c in val_configs if c["precision"] >= target_p and c["recall"] >= target_r
    ]
    by_f1 = sorted(val_configs, key=lambda c: -c["f1"])

    if passing:
        best = max(passing, key=lambda c: c["f1"])
        lines.append(
            f"**Target cleared** by **{best['name']}**: "
            f"P={best['precision']:.4f} R={best['recall']:.4f} "
            f"F1={best['f1']:.4f}\n"
        )
    else:
        lines.append("**Target NOT cleared** by any single config on val.\n")

    lines.append("Top 5 configs by val F1:\n")
    lines.append("| rank | config | P | R | F1 |")
    lines.append("|---|---|---|---|---|")
    for i, c in enumerate(by_f1[:5], 1):
        clears = c.get("precision", 0) >= target_p and c.get("recall", 0) >= target_r
        marker = " **" if clears else ""
        end = "**" if marker else ""
        lines.append(
            f"| {i} |{marker} {c['name']}{end} "
            f"| {c['precision']:.4f} | {c['recall']:.4f} "
            f"| {c['f1']:.4f} |"
        )

    platt_weights = {
        "features": feature_names,
        "coefficients": [round(float(c), 6) for c in coefs],
        "intercept": round(float(intercept), 6),
    }

    return "\n".join(lines) + "\n", platt_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--training-labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-precision", type=float, default=0.93)
    parser.add_argument("--target-recall", type=float, default=0.95)
    args = parser.parse_args()

    train_recs = load_predictions(args.train_predictions)
    val_recs = load_predictions(args.val_predictions)

    assert train_recs and "kept_tubes" in train_recs[0], (
        "predictions.json missing 'kept_tubes'; "
        "run evaluate_packaged with post-B1 model code"
    )

    conf_floor = scan_confidence_floor(args.training_labels_dir)

    report_md, platt_weights = build_report(
        train_records=train_recs,
        val_records=val_recs,
        conf_floor=conf_floor,
        target_p=args.target_precision,
        target_r=args.target_recall,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "analysis_report.md").write_text(report_md)
    (args.output_dir / "platt_model.json").write_text(
        json.dumps(platt_weights, indent=2) + "\n"
    )

    best_config = {
        "package": {
            "infer": {
                "confidence_threshold": conf_floor["min"],
                "pad_to_min_frames": 20,
                "pad_strategy": "symmetric",
            }
        },
        "note": (
            "Auto-generated by analyze_variant.py. "
            "Review analysis_report.md for the full sweep."
        ),
    }
    (args.output_dir / "recommended_config.yaml").write_text(
        yaml.dump(best_config, default_flow_style=False)
    )

    print(f"Wrote analysis to {args.output_dir}/")
    print(report_md[-500:])


if __name__ == "__main__":
    main()
