#!/usr/bin/env bash
#
# Publish review + export artifacts to DVC and commit, per split.
#
# Detects which splits have changes (via dvc status), shows a summary,
# prompts y/N per split, and for each confirmed split:
#   1. dvc add data/09_review/<m>/<s> data/10_export/<m>/<s>
#   2. dvc push (targeted to that split's .dvc files)
#   3. git add the split's .dvc files (+ .gitignore on first publish)
#   4. git commit -e -m "review(<s>): bbox corrections + export"
#
# Each split is atomic: data on the DVC remote + git commit, or neither.
# git's commit editor (-e) is the message confirmation step.
#
# Usage:
#   make audit-publish            # uses default model
#   MODEL=other-model make audit-publish

set -euo pipefail

MODEL="${MODEL:-yolo11s-nimble-narwhal}"
SPLITS=(train val test)

if [[ -f data/09_review.dvc || -f data/10_export.dvc ]]; then
    echo "ERROR: data/09_review and data/10_export are still tracked at the"
    echo "umbrella level (data/09_review.dvc / data/10_export.dvc exist)."
    echo "This branch expects per-split DVC tracking — check git log for the"
    echo "commit that converted tracking, and rebase/merge it in."
    exit 1
fi

if [[ ! -d data/09_review/$MODEL ]]; then
    echo "ERROR: data/09_review/$MODEL/ not found. Wrong MODEL or no reviews yet?"
    exit 1
fi

dirty=()
for s in "${SPLITS[@]}"; do
    review="data/09_review/$MODEL/$s/review.json"
    [[ -f "$review" ]] || continue
    if ! uv run dvc status -q \
        "data/09_review/$MODEL/$s" "data/10_export/$MODEL/$s" >/dev/null 2>&1; then
        dirty+=("$s")
    fi
done

if [[ ${#dirty[@]} -eq 0 ]]; then
    echo "Nothing to publish — all tracked splits match the DVC cache."
    exit 0
fi

echo "Detected changes in: ${dirty[*]}"
echo

published=()
for s in "${dirty[@]}"; do
    echo "▸ $s"
    uv run python scripts/summarize_split.py --model "$MODEL" --split "$s"
    read -r -p "  Publish $s? [y/N]: " ans
    if [[ ! "$ans" =~ ^[Yy]$ ]]; then
        echo "  Skipped $s."
        echo
        continue
    fi

    uv run dvc add "data/09_review/$MODEL/$s" "data/10_export/$MODEL/$s"
    uv run dvc push \
        "data/09_review/$MODEL/$s.dvc" "data/10_export/$MODEL/$s.dvc"

    git add \
        "data/09_review/$MODEL/$s.dvc" \
        "data/10_export/$MODEL/$s.dvc"
    # .gitignore only changes on first publish after migration; ignore failure.
    git add -- \
        "data/09_review/$MODEL/.gitignore" \
        "data/10_export/$MODEL/.gitignore" 2>/dev/null || true

    if ! git commit -e -m "review($s): bbox corrections + export"; then
        echo
        echo "Commit aborted for $s. Data is on the DVC remote but uncommitted;"
        echo "files remain staged. Resolve manually (git restore --staged ...)"
        echo "or re-run this command."
        exit 1
    fi
    published+=("$s")
    echo
done

if [[ ${#published[@]} -eq 0 ]]; then
    echo "Nothing was published."
    exit 0
fi

echo "Published: ${published[*]}"
echo "Run 'git push' to share with the team."
