#!/usr/bin/env bash
# Follow-up to the paper-recipe pipeline: retrain LW-DETR tiny with the SAME
# paper recipe (2e-4 base / 1e-4 backbone, EMA 0.9999, no WD on norm/bias,
# grad-clip 0.1) but with early stopping DISABLED — all 132 epochs run to
# completion — to test whether the early-stopped lwdetr-tiny-paper (F1 0.878,
# stopped well before 132 epochs) left accuracy on the table.
#
# New model: lwdetr-tiny-paper-full (kept alongside lwdetr-tiny-paper so the two
# are directly comparable). BestF1Callback still selects the best-val-F1 epoch,
# so we report the best checkpoint even though the full schedule runs.
#
# Waits for the current pipeline's last model (rtdetrv2-r18-paper) to finish and
# the GPU to free before starting, so there's no contention. Launch detached:
#   setsid nohup bash scripts/train_lwdetr_tiny_full.sh > <log> 2>&1 < /dev/null &
set -u

cd "$(dirname "$0")/.." || exit 1
LOG="$(pwd)/data/06_models/_paper_recipe_logs"
mkdir -p "$LOG"

name=lwdetr-tiny-paper-full
mdir="data/06_models/$name"
out="data/07_model_output/$name"

if [ -f "$out/metrics.json" ]; then echo "[$name] metrics.json exists -> skip"; exit 0; fi

# 1) Wait for the current paper-recipe pipeline to finish (its last model's
#    metrics exist) AND the GPU to be free (no training process running).
echo "[$name] waiting for rtdetrv2-r18-paper to finish + GPU to free..."
while [ ! -f data/07_model_output/rtdetrv2-r18-paper/metrics.json ]; do sleep 60; done
while pgrep -f 'python scripts/train.py' >/dev/null 2>&1; do sleep 30; done
echo "[$name] current pipeline done, GPU free — starting $(date '+%F %T')"

# 2) Train all 132 epochs, no early stop (patience 999 never triggers within 132).
uv run python scripts/train.py --checkpoint AnnaZhang/lwdetr_tiny_60e_coco --output-dir "$mdir" \
    --wandb-run-name "$name" \
    --train-dir data/01_raw/datasets/train --val-dir data/01_raw/datasets/val \
    --image-size 1024 --epochs 132 --batch-size 8 \
    --learning-rate 2e-4 --backbone-lr 1e-4 --weight-decay 1e-4 --max-grad-norm 0.1 \
    --no-wd-on-norm-bias --ema-decay 0.9999 --early-stop-patience 999 --seed 42 \
    --wandb-project detector-leaderboard --log-images \
    > "$LOG/$name.train.log" 2>&1 || { echo "[$name] TRAIN FAILED"; exit 1; }

# 3) infer -> evaluate -> profile (LW-DETR uses the hf_detr backend)
uv run python scripts/infer.py --backend hf_detr --data-dir data/01_raw/datasets/val \
    --model-dir "$mdir" --output-file "data/02_intermediate/$name/val_predictions.json" \
    --confidence-threshold 0.01 > "$LOG/$name.infer_val.log" 2>&1 || { echo "[$name] INFER val FAILED"; exit 1; }
uv run python scripts/infer.py --backend hf_detr --data-dir data/01_raw/datasets/test \
    --model-dir "$mdir" --output-file "data/02_intermediate/$name/test_predictions.json" \
    --confidence-threshold 0.01 > "$LOG/$name.infer_test.log" 2>&1 || { echo "[$name] INFER test FAILED"; exit 1; }
uv run python scripts/evaluate.py --model-name "$name" \
    --val-predictions "data/02_intermediate/$name/val_predictions.json" --val-dir data/01_raw/datasets/val \
    --test-predictions "data/02_intermediate/$name/test_predictions.json" --test-dir data/01_raw/datasets/test \
    --output-dir "$out" --iou-threshold 0.1 --conf-min 0.01 --conf-max 0.95 --conf-step 0.01 \
    > "$LOG/$name.eval.log" 2>&1 || { echo "[$name] EVAL FAILED"; exit 1; }
uv run python scripts/profile_model.py --backend hf_detr --model-name "$name" \
    --model-dir "$mdir" --image-size 1024 --warmup 10 --iters 50 --output-dir "$out" \
    > "$LOG/$name.profile.log" 2>&1 || { echo "[$name] PROFILE FAILED"; exit 1; }
echo "[$name] DONE $(date '+%F %T')  -> $(grep -o '\"f1\": [0-9.]*' "$out/metrics.json" | head -1)"

# 4) refresh reports
uv run python scripts/leaderboard.py --results-dir data/07_model_output --output-dir data/08_reporting \
    > "$LOG/leaderboard.full.log" 2>&1 && echo "leaderboard OK" || echo "leaderboard FAILED"
uv run python scripts/build_baseline_comparison.py \
    > "$LOG/baseline_comparison.full.log" 2>&1 && echo "baseline_comparison OK" || echo "baseline_comparison FAILED"
echo "[$name] PIPELINE COMPLETE $(date '+%F %T')"
