# Self-hosted raw datasets — design

**Status:** draft
**Date:** 2026-05-14
**Owner:** Arthur

## 1. Background

`data/01_raw/datasets/{train,val,test}/{images,labels}` is currently
tracked by six DVC frozen-import files (`images.dvc` / `labels.dvc` per
split) pointing at `pyronear/pyro-dataset` `v4.0.0`. The intent was to
get version-pinned, reproducible inputs and one-command upgrades via
`dvc update`.

In practice the setup has two problems:

1. **Pyro-dataset's S3 remote is incomplete at v4.0.0.** The parent
   manifest `3f5c9c37…dir` for the whole `yolo_test` split is not in
   `s3://pyro-dataset-dvc-v2/dvc/`. Colleagues cloning this experiment
   and running `dvc pull` get
   `Checkout failed for following targets: data/01_raw/datasets/{test,train,val}/{images,labels}`
   with a list of `No file hash info found for <path>.jpg` warnings.
2. **The frozen imports reference sub-paths.** Each `.dvc` file's
   `outs.md5` (e.g. `20a4c8a3…dir` for `test/images`) is a hash derived
   *locally* during the original `dvc import`. It only ever existed in
   the importer's cache and is not present in either pyro-dataset's
   remote or this experiment's remote. Even if pyro-dataset's remote
   were healthy, only the original importer could re-host these
   sub-path manifests.

Net effect: nobody but the original importer can pull the raw datasets,
even though the data exists on the importer's disk.

## 2. Goals

- Anyone with read access to `s3://pyro-vision-rd/` can `dvc pull` the
  raw datasets without needing access to pyro-dataset's bucket.
- Updating to a new pyro-dataset version is a single command
  (`make refresh-datasets PYRO_DATASET_VERSION=v5.0.0`).
- The provenance link to which pyro-dataset version the data came from
  is preserved as a checked-in file, not implicit in DVC metadata.
- The audit-app header continues to display the pyro-dataset version
  (currently sourced from `.dvc` files' `repo.rev` field, which goes
  away under this design).
- No changes to the predict / review / export pipeline stages.

## 3. Non-goals

- Fixing pyro-dataset's S3 remote. Out of scope for this experiment.
- Generalizing the refresh script across experiments. If another
  experiment needs the same pattern, copy and adapt later.
- Preserving the `dvc update` workflow for these datasets. We are
  intentionally trading it for `dvc pull` reliability.

## 4. Target layout

```
data/01_raw/datasets/
├── SOURCE.json              # provenance (committed to git)
├── train.dvc                # tracks the whole train/ dir (images + labels)
├── val.dvc
├── test.dvc
├── train/{images,labels}/   # gitignored, dvc-managed
├── val/{images,labels}/
└── test/{images,labels}/
```

- Three `.dvc` files instead of six. Each tracks one split's full
  directory (images + labels together), matching pyro-dataset's own
  tracking shape (`data/processed/yolo_test`, `data/processed/yolo_train_val`).
- Data is stored in `s3://pyro-vision-rd/dvc/experiments/data-quality/frame-level/`
  (this experiment's existing remote) via plain `dvc add`. No
  `repo:` / frozen-import field.
- `SOURCE.json` records the upstream version, commit, last-refresh
  timestamp, and an optional `note`:
  ```json
  {
    "pyro_dataset_version": "v4.0.0",
    "pyro_dataset_commit": "4e16c464edda7400b0ac738c4f45f8d8e50fa735",
    "refreshed_at": "2026-05-14T12:30:00+02:00",
    "note": "optional free-form caveat"
  }
  ```
  The audit-app version badge reads `pyro_dataset_version` only;
  `note` is informational for humans reading the file.

## 4a. Audit-app version badge

`src/data_quality_frame_level/audit_app/dataset_version.py` currently
parses `deps[0].repo.rev` from each `.dvc` file and returns either a
single rev string or a `"mixed: <r1>, <r2>"` marker. Both go away
because plain `dvc add` files have no `repo:` block.

Replace the implementation: `read_dataset_version(datasets_root)` reads
`datasets_root / "SOURCE.json"` and returns `pyro_dataset_version`
from it (or `None` when the file is missing or unreadable). The
`"mixed"` case disappears entirely — there is now a single source of
truth, so the function reduces to one JSON read.

The function's signature and return type (`str | None`) stay the same.
Callers (`audit_app/main.py`, `static/app.js`) and the rendered badge
in `static/index.html` are unchanged.

Tests in `tests/test_audit_app_dataset_version.py` are rewritten to
exercise SOURCE.json instead of `.dvc` templates: missing file → None,
malformed JSON → None, valid file → version string.

## 5. Refresh flow

`make refresh-datasets PYRO_DATASET_VERSION=v5.0.0` runs
`scripts/refresh_datasets.py`, which does:

1. Validate `PYRO_DATASET_VERSION` is set; abort if missing.
2. Clone `pyronear/pyro-dataset` shallow at that tag into a temp dir
   (deleted on script exit). Always a fresh clone — no
   user-environment assumptions about pre-existing checkouts.
3. `dvc pull` inside the pyro-dataset clone to materialize
   `data/processed/yolo_test` and `data/processed/yolo_train_val`.
   Fails fast if pyro-dataset's remote is missing data at this version.
4. Sync files into this experiment:
   - `pyro-dataset/data/processed/yolo_test/{images,labels}/test/*`
     → `data/01_raw/datasets/test/{images,labels}/`
   - `pyro-dataset/data/processed/yolo_train_val/{images,labels}/{train,val}/*`
     → `data/01_raw/datasets/{train,val}/{images,labels}/`
   - Exact source paths confirmed against pyro-dataset's `dvc.yaml`
     during implementation; this is the expected layout based on the
     existing `images.dvc` / `labels.dvc` deps but the script verifies
     before copying.
5. `uv run dvc add data/01_raw/datasets/train data/01_raw/datasets/val data/01_raw/datasets/test`
6. `uv run dvc push` the three resulting `.dvc` files.
7. Write `data/01_raw/datasets/SOURCE.json` with version, resolved
   commit hash, and current timestamp (ISO-8601, local tz).
8. Print a summary of what changed (new file counts per split, source
   commit) and remind the user to `git add` the `.dvc` files +
   `SOURCE.json` and run `dvc repro` if the data changed.

The script does **not** commit or push to git — that stays manual so
the user reviews diffs.

## 6. Migration (one-time)

To convert the current setup on this branch:

1. Delete the six `*.dvc` import files under
   `data/01_raw/datasets/{train,val,test}/{images,labels}.dvc`.
2. Update `.gitignore` files: the current per-split
   `data/01_raw/datasets/{train,val,test}/.gitignore` ignores
   `/images` and `/labels` (correct for the old setup and still
   correct under the new one — `dvc add` of the parent dir keeps
   ignoring the contents). The new `.dvc` files sit at
   `data/01_raw/datasets/<split>.dvc`, and `dvc add` will generate
   `data/01_raw/datasets/.gitignore` ignoring `/train`, `/val`,
   `/test`. No manual `.gitignore` edits needed; DVC manages them.
3. Run the refresh script with `PYRO_DATASET_VERSION=v4.0.0` (matching
   the current rev_lock) to re-add and push under the new shape.
4. Verify `dvc.lock` deps still match — the predict stages reference
   `data/01_raw/datasets/{train,val,test}` as directories, which the
   new `.dvc` files cover. The dir hashes in `dvc.lock` may need
   updating via `dvc commit` if the directory contents differ.
5. Commit the new `.dvc` files, `SOURCE.json`, the DVC-managed
   `.gitignore` (if changed), and any `dvc.lock` changes.

Pyro-dataset's S3 was missing some v4.0.0 blobs (notably the
`yolo_test` parent manifest) at the time this spec was drafted; a
colleague is pushing the missing data in parallel. The migration runs
the refresh script as written and depends on that push completing
first. If the refresh fails with `No file hash info found` warnings,
that is the diagnostic signal that the upstream push is incomplete —
do not work around it locally; report back and wait.

## 7. Testing

- Manual: clone the experiment fresh into a sibling directory, run
  `uv run dvc pull`, verify all three splits materialize and warnings
  / errors are absent.
- The refresh script is exercised by running it once during migration.
  No unit tests — it's a thin shell around `git clone` + `dvc pull` +
  `dvc add` + `dvc push`.

## 8. Risks

- **Pipeline breakage from dir-hash changes.** `dvc.lock` currently
  has per-stage dataset deps with specific dir hashes
  (e.g. `data/01_raw/datasets/test` md5: `3b8457851c…`). If the new
  `.dvc` files produce a different overall dir hash (because the new
  layout omits or adds files), DVC will mark predict stages as
  out-of-date. Mitigation: run `dvc commit` after migration, accept
  the new hashes, and rerun `dvc repro` only if the data truly
  changed.
- **Loss of `dvc update` ergonomics.** Refreshing now requires a
  refresh script run rather than a single `dvc update` command. At
  every-few-weeks cadence this is acceptable; not at daily.
- **Bucket cost / size.** Storing the raw datasets in this
  experiment's S3 duplicates them across pyro-dataset's bucket. ~500MB
  total per snapshot. Acceptable.

## 9. Open questions

- Should `SOURCE.json` also record a hash of the resulting tree (e.g.
  `tree_md5: <hash>`) to detect drift between checked-in metadata and
  actual data? Maybe v2. For now the DVC dir hashes serve that role.
