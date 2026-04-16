# YOLO weights from HuggingFace Hub for bbox-tube-temporal

Status: design (not yet implemented)
Date: 2026-04-16

## Goal

Source the YOLO companion-detector weights for the `bbox-tube-temporal`
experiment from the HuggingFace Hub
(`pyronear/yolo11s_nimble-narwhal_v6.0.0`, file `best.pt`) via a new DVC
`prepare` stage, so the exact pinned model version is reproducible from
`params.yaml` alone. Today the file sits at
`data/01_raw/models/best.pt` with no machine-checked provenance — its
contents happen to be byte-identical to the HF artifact
(SHA256 `0bf3c7ee…8ac9d`, 19,225,626 bytes at HF commit
`5f20f6bea14f76b964df740c38af27b96b93f36a`), but nothing in the repo
records that fact.

## Context

The `package` stage in `dvc.yaml` depends on
`data/01_raw/models/best.pt` as an external file. The directory
`data/01_raw/models/` is git-ignored and has no standalone `.dvc`
sidecar; DVC hashes the file as a plain dep under `package` in
`dvc.lock`. A new user cloning the repo has no way to recover the file
from the codebase alone.

Two sibling experiments already implement exactly this pattern:

- `experiments/temporal-models/tracking-fsm-baseline/scripts/prepare.py`
  — near-identical use case (single `.pt` from HF Hub).
- `experiments/temporal-models/pyro-detector-baseline/scripts/prepare.py`
  — same pattern, plus tarball extraction.

Both use `huggingface_hub.hf_hub_download`, parameterize the repo and
filename via `params.yaml`, and expose the directory as a DVC stage
output.

## Scope

- In scope: a new `prepare` stage that downloads one file
  (`best.pt`) from one HF repo into `data/01_raw/models/`.
- Out of scope: any change to `package_model.py`, `scripts/evaluate*`,
  or the classifier training/eval stages. The file path
  (`data/01_raw/models/best.pt`) that downstream consumers hardcode
  does not change.
- Out of scope: fetching ONNX/NCNN variants or the `manifest.yaml` from
  the HF repo. We only need the PyTorch weights for the Ultralytics
  loader used by `BboxTubeTemporalModel`.

## Design

### 1. New script: `scripts/prepare.py`

Copy of `tracking-fsm-baseline/scripts/prepare.py` verbatim. It accepts
`--model-repo`, `--model-filename`, `--output-model-dir`; calls
`hf_hub_download(repo_id=..., filename=..., local_dir=...)`; and skips
the download when the target file already exists. Kept as a separate
script (not merged into `package_model.py`) so it's a clean DVC stage
boundary and so re-running the packager doesn't require network
access.

### 2. `params.yaml` additions

```yaml
prepare:
  model_repo: "pyronear/yolo11s_nimble-narwhal_v6.0.0"
  model_filename: "best.pt"
```

### 3. `dvc.yaml` additions

A new first stage, placed before `truncate`:

```yaml
prepare:
  desc: "Download YOLO detector weights from HuggingFace Hub"
  cmd: >-
    uv run python scripts/prepare.py
    --model-repo ${prepare.model_repo}
    --model-filename ${prepare.model_filename}
    --output-model-dir data/01_raw/models
  deps:
    - scripts/prepare.py
  params:
    - prepare
  outs:
    - data/01_raw/models
```

### 4. `pyproject.toml` dependency

Add `huggingface_hub>=0.24` to `[project].dependencies`.

### 5. No changes elsewhere

`scripts/package_model.py` keeps its
`--yolo-weights-path data/01_raw/models/best.pt` default. The existing
`package` stage in `dvc.yaml` already lists
`data/01_raw/models/best.pt` as a dep; once `prepare` exists, DVC
chains `prepare → package` implicitly via the shared path.

## DVC repro impact

On first `dvc repro` after the change:

- **`prepare`**: runs once. The script short-circuits because
  `best.pt` already exists on disk, so no network I/O. DVC records the
  output hash in `dvc.lock` and copies the file into the DVC cache
  (previously it was only an external dep hash).
- **`package`**: does **not** re-run. Its dep
  `data/01_raw/models/best.pt` is byte-identical, its md5 in
  `dvc.lock` is unchanged, and its `deps:` list in `dvc.yaml` is
  unchanged.
- All other stages: untouched.

Verification path before committing: `uv run dvc status` should show
`prepare` as new and `package` as cached; `uv run dvc repro --dry`
confirms `prepare` is the only command to execute.

## Non-goals / explicitly deferred

- Verifying the HF file's SHA256 against a pinned value inside
  `prepare.py`. The HF Hub's resolve endpoint is content-addressed and
  `hf_hub_download` validates the `x-linked-etag`, so a second check
  would be redundant. If we later want a commit-pinned download, add a
  `revision:` key to `params.yaml` and pass it through — a
  one-line extension.
- Replacing `best.pt` with the ONNX or NCNN export. The temporal
  model's Ultralytics wrapper expects the PyTorch weights.
- Removing the `data/01_raw/models/` entry from `.gitignore`. The
  directory stays git-ignored; DVC manages its contents via the
  `prepare` stage's output cache.
