# Export flow — design

**Status:** draft
**Date:** 2026-05-05
**Owner:** Arthur

## 1. Background

The review app ([`2026-05-05-review-app-design.md`](2026-05-05-review-app-design.md))
already produces a per-`(model, split)` export at
`data/10_export/<model>/<split>/` containing corrected YOLO `.txt`
files and a single `manifest.json`. That export is enough to *describe*
the corrections but glosses over two practical needs:

1. **Unclear-status frames have no destination.** Reviewers who flag a
   frame `unclear` get no hand-off — those frames stay in the local
   `review.json` only.
2. **No provenance.** A patch sitting in someone's checkout doesn't
   record *how* it was produced (which audit experiment commit, which
   thresholds, which predictions file). When patches accumulate or a
   reviewer wants to reproduce an audit, that context is missing.

This spec extends the export to address both, while keeping the apply
side (in `pyro-dataset`) completely out of scope.

## 2. Goals

- Export structure is **self-documenting**: a fresh consumer can read
  the files and apply the patch without out-of-band knowledge.
- **Provenance is captured** so any patch is traceable back to the
  audit run that produced it.
- **Unclear-status frames** flow through the export as a separate
  hand-off file consumed by a second reviewer (not by the apply step).
- The apply contract is **stable and minimal**: `pyro-dataset` only
  needs `manifest.json` + `labels/<stem>.txt` to perform the patch;
  the other files are informational.

## 3. Non-goals

- The apply step itself. Lives in `pyro-dataset`, separately
  versioned. This spec describes the contract it consumes; the
  implementation is out of scope.
- Image-level decisions (e.g., "this frame should be removed
  entirely"). Out of scope for this iteration; would require new
  status states (`drop`) in the review data model and corresponding
  UI work.
- Programmatic PR creation against `pyro-dataset`. The reviewer drives
  the integration manually (clone + apply + commit + PR).
- DVC / S3 routing of the export between machines. Reviewers hand off
  exports via local checkouts; DVC tracking on the audit side is
  archival.

## 4. Output layout

```
data/10_export/<model>/<split>/
  labels/<stem>.txt        # corrected YOLO labels (only-changed frames)
  manifest.json            # the apply contract — pyro-dataset reads this
  pending.json             # unclear-status frames for second-opinion review
  provenance.json          # audit-side context for reproducibility
```

Three sibling JSON files instead of one because each has a different
audience and lifecycle:

| File              | Audience                       | Required for apply |
|-------------------|--------------------------------|--------------------|
| `manifest.json`   | `pyro-dataset` apply script    | yes                |
| `pending.json`    | another reviewer for triage    | no                 |
| `provenance.json` | the audit team, reproducibility| no                 |

Files are emitted by `make review-export` (and the equivalent
`scripts/export_review_app.py` CLI) and DVC-tracked under
`data/10_export/`.

## 5. File schemas

### 5.1 `manifest.json` — the apply contract

```json
{
  "version": 1,
  "model": "yolo11s-nimble-narwhal",
  "split": "val",
  "exported_at": "2026-05-05T14:30:00Z",
  "changed": [
    {
      "stem": "hpwren-figlib_rmwmoboc_999_2018-07-29T00-19-06",
      "added": 1,
      "removed": 0,
      "modified": 0,
      "reviewer": "arthur",
      "note": "moved bbox up; pred was on cloud, removed FP"
    }
  ],
  "totals": {"changed": 17, "added": 8, "removed": 4, "modified": 9}
}
```

Properties:

- `changed` is sorted by `stem` for stable diffs.
- Every entry's `stem` corresponds to a sibling
  `labels/<stem>.txt` file in the export directory. The presence of
  the `.txt` file is the source of truth for the new content; the
  manifest's `added`/`removed`/`modified` counts are advisory metadata
  for the apply step's PR description.
- **No source-routing field** (`wildfire` vs `fp`). The apply script
  on the `pyro-dataset` side resolves which upstream a stem belongs to
  by checking `processed/wildfire_yolo/labels/` vs
  `processed/fp_yolo/labels/`. Keeps the audit's exporter independent
  of `pyro-dataset`'s internal layout.
- `unclear`-status samples are **not** included here — they live in
  `pending.json` (§5.2).
- `reviewer` and `note` are passed through verbatim from the
  `review.json` sample. Both may be absent if the reviewer didn't set
  them.

### 5.2 `pending.json` — second-opinion hand-off

```json
{
  "version": 1,
  "model": "yolo11s-nimble-narwhal",
  "split": "val",
  "exported_at": "2026-05-05T14:30:00Z",
  "pending": [
    {
      "stem": "hpwren-figlib_losmoboc_999_2019-07-16T00-18-24",
      "reviewer": "arthur",
      "note": "could be smoke or thin cloud — second pair of eyes please"
    }
  ]
}
```

Properties:

- Includes every sample whose `status == "unclear"` in `review.json`.
- Sorted by `stem`.
- The apply script **must not** consume this file. It exists so a
  second reviewer can pick up the queue, look at each frame in the
  review app, and either resolve it (mark `reviewed`) or escalate.

### 5.3 `provenance.json` — audit-side context

```json
{
  "version": 1,
  "audit_repo": "pyronear/vision-rd",
  "audit_commit": "abc123de4567890abc",
  "audit_branch": "arthur/data-quality-frame-level-workflow",
  "experiment": "experiments/data-quality/frame-level",
  "model": "yolo11s-nimble-narwhal",
  "split": "val",
  "thresholds": {"conf": 0.05, "iou": 0.05, "review_conf": 0.35},
  "predictions_path": "data/07_model_output/yolo11s-nimble-narwhal/val/predictions.json",
  "predictions_md5": "f3a9...c4e7",
  "exported_at": "2026-05-05T14:30:00Z"
}
```

Properties:

- `audit_commit` is the HEAD SHA of `vision-rd` at export time. If the
  working tree is dirty, append `+dirty`.
- `audit_branch` helps trace which feature branch produced the patch.
- `thresholds` capture the values from `params.yaml` at the time of
  the audit (not the live UI sliders — apply target is the canonical
  configured thresholds, not whatever a reviewer was looking at).
- `predictions_md5` is the MD5 of the `predictions.json` file
  contents, computed at export time. Identical to whatever DVC would
  store, but computed directly so we don't depend on DVC's lockfile
  format. A future consumer can verify they're looking at the same
  predictions artifact.

The provenance file is informational only — `pyro-dataset` does not
need to read it to apply the patch. It's preserved alongside the patch
so anyone reading the export later can answer "where did this come
from?".

## 6. Apply contract (informative)

`pyro-dataset` will eventually ship a script roughly like:

```bash
cd ../pyro-dataset
uv run python scripts/apply_audit.py \
    /abs/path/to/vision-rd/experiments/data-quality/frame-level/data/10_export/yolo11s-nimble-narwhal/val/
```

That script's responsibility is to:

1. Read `manifest.json`.
2. For each `changed[].stem`, locate the upstream label by checking
   `processed/wildfire_yolo/labels/<stem>.txt` then
   `processed/fp_yolo/labels/<stem>.txt`.
3. Replace the upstream label with the corresponding
   `labels/<stem>.txt` from the export.
4. Optionally run `dvc repro merge_yolo_dataset` to propagate to
   `yolo_train_val/yolo_test`.
5. Print a summary suitable for a PR description (using the manifest
   `totals` and `changed[].note`).

The above is not part of this spec — it's documented here to make the
contract concrete. Implementation lives entirely in `pyro-dataset`.

## 7. Implementation changes

This spec requires modifying the existing exporter in
`src/data_quality_frame_level/review_app/export.py` and the CLI in
`scripts/export_review_app.py`. Specifically:

- Rename the existing `export_corrections` → `export_manifest` (it
  produces what the spec now calls `manifest.json`).
- Add `export_pending` that builds `pending.json` from `unclear`
  samples in `review.json`.
- Add `export_provenance` that gathers git SHA, branch, threshold
  values from `params.yaml`, and the `.dvc` md5 of the predictions
  file.
- Update the top-level `export_corrections` orchestrator (called from
  the CLI) to write all three files.
- Replace the old single-file `manifest.json` with the three-file
  layout described in §4. Each new file carries its own
  `"version": 1` marker. Existing exports under `data/10_export/`
  were never consumed downstream (no apply script exists yet), so no
  migration is needed — `dvc repro` regenerates them.

Test additions:

- `test_export_manifest_has_changed_only`
- `test_export_pending_includes_only_unclear`
- `test_export_pending_excludes_reviewed_and_untouched`
- `test_export_provenance_captures_git_and_predictions_md5`
- `test_export_provenance_marks_dirty_when_working_tree_unclean`

The CLI in `scripts/export_review_app.py` keeps the same surface
(`make review-export`, no flags) — it just emits more files.

## 8. Open questions

None blocking. The hand-off mechanism (reviewers running the apply
script in a sibling `pyro-dataset` checkout) is the simplest possible
choice and can be revisited later if a more automated path is wanted
(e.g., gh-driven PR creation, S3-mediated patches between machines).
