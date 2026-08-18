#!/usr/bin/env bash
# Retrain the Group-1 HF DETR-family detectors with the D-FINE-style *paper*
# optimizer recipe instead of the shared flat recipe — discriminative backbone
# LR, EMA (0.9999), no weight decay on norm/bias, and tight gradient clipping
# (D-FINE's 0.1). Plus LW-DETR small (a new architecture, similar param count to
# the baseline), trained with the same recipe.
#
# LR note: this uses D-FINE's actual pairing of a high base LR with tight
# gradient clipping — base 2e-4 / backbone 1e-4 (matching the successful
# dfine-small-paper) plus max-grad-norm 0.1. The clip is what keeps 2e-4 from
# diverging to NaN on the small dfine-nano backbone (it did without a clip);
# dropping the LR to 1e-4 instead just starved learning (nano undertrained to
# F1 0.54 vs the flat recipe's 0.77), so the LR stays high and the clip stays
# tight — the combination the paper is tuned around.
#
# One GPU, models run sequentially. Resumable: a model whose metrics.json
# already exists is skipped, so relaunching after an interruption continues.
# Each model runs the full pipeline: train -> infer(val,test) -> evaluate ->
# profile. The leaderboard + baseline_comparison.md are regenerated at the end.
#
#   bash scripts/train_paper_recipe.sh
set -u

cd "$(dirname "$0")/.." || exit 1
LOG="$(pwd)/data/06_models/_paper_recipe_logs"
mkdir -p "$LOG"

# Shared paper recipe (matches dfine-small-paper).
COMMON=(--train-dir data/01_raw/datasets/train --val-dir data/01_raw/datasets/val
        --image-size 1024 --epochs 132 --batch-size 8
        --learning-rate 2e-4 --backbone-lr 1e-4 --weight-decay 1e-4 --max-grad-norm 0.1
        --no-wd-on-norm-bias --ema-decay 0.9999 --early-stop-patience 20 --seed 42
        --wandb-project detector-leaderboard --log-images)

run_model () {
  local name="$1" ckpt="$2" backend="$3"
  local mdir="data/06_models/$name"
  local out="data/07_model_output/$name"
  if [ -f "$out/metrics.json" ]; then echo "[$name] metrics.json exists -> skip"; return 0; fi

  echo "[$name] TRAIN start $(date '+%F %T')"
  uv run python scripts/train.py --checkpoint "$ckpt" --output-dir "$mdir" \
      --wandb-run-name "$name" "${COMMON[@]}" \
      > "$LOG/$name.train.log" 2>&1 || { echo "[$name] TRAIN FAILED (see $LOG/$name.train.log)"; return 1; }

  echo "[$name] INFER val+test"
  uv run python scripts/infer.py --backend "$backend" --data-dir data/01_raw/datasets/val \
      --model-dir "$mdir" --output-file "data/02_intermediate/$name/val_predictions.json" \
      --confidence-threshold 0.01 > "$LOG/$name.infer_val.log" 2>&1 || { echo "[$name] INFER val FAILED"; return 1; }
  uv run python scripts/infer.py --backend "$backend" --data-dir data/01_raw/datasets/test \
      --model-dir "$mdir" --output-file "data/02_intermediate/$name/test_predictions.json" \
      --confidence-threshold 0.01 > "$LOG/$name.infer_test.log" 2>&1 || { echo "[$name] INFER test FAILED"; return 1; }

  echo "[$name] EVALUATE"
  uv run python scripts/evaluate.py --model-name "$name" \
      --val-predictions "data/02_intermediate/$name/val_predictions.json" --val-dir data/01_raw/datasets/val \
      --test-predictions "data/02_intermediate/$name/test_predictions.json" --test-dir data/01_raw/datasets/test \
      --output-dir "$out" --iou-threshold 0.1 --conf-min 0.01 --conf-max 0.95 --conf-step 0.01 \
      > "$LOG/$name.eval.log" 2>&1 || { echo "[$name] EVAL FAILED"; return 1; }

  echo "[$name] PROFILE"
  uv run python scripts/profile_model.py --backend "$backend" --model-name "$name" \
      --model-dir "$mdir" --image-size 1024 --warmup 10 --iters 50 --output-dir "$out" \
      > "$LOG/$name.profile.log" 2>&1 || { echo "[$name] PROFILE FAILED"; return 1; }

  echo "[$name] DONE $(date '+%F %T')  -> $(grep -o '\"f1\": [0-9.]*' "$out/metrics.json" | head -1)"
}

# name                checkpoint                          backend
run_model dfine-nano-paper    ustc-community/dfine-nano-coco   dfine
run_model lwdetr-tiny-paper   AnnaZhang/lwdetr_tiny_60e_coco   hf_detr
run_model lwdetr-small-paper  AnnaZhang/lwdetr_small_60e_coco  hf_detr
run_model rtdetrv2-r18-paper  PekingU/rtdetr_v2_r18vd          hf_detr

echo "=== regenerating leaderboard + baseline_comparison.md ==="
uv run python scripts/leaderboard.py --results-dir data/07_model_output --output-dir data/08_reporting \
    > "$LOG/leaderboard.log" 2>&1 && echo "leaderboard OK" || echo "leaderboard FAILED"
uv run python scripts/build_baseline_comparison.py \
    > "$LOG/baseline_comparison.log" 2>&1 && echo "baseline_comparison OK" || echo "baseline_comparison FAILED"
echo "=== PIPELINE COMPLETE $(date '+%F %T') ==="
