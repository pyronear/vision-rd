# YOLO weights from HuggingFace — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Source the bbox-tube-temporal YOLO companion-detector weights from `pyronear/yolo11s_nimble-narwhal_v6.0.0` on HuggingFace Hub via a new DVC `prepare` stage, replacing the implicit dependency on a hand-placed `data/01_raw/models/best.pt`.

**Architecture:** New `scripts/prepare.py` wraps `huggingface_hub.hf_hub_download` and writes the file into `data/01_raw/models/`. A new `prepare` DVC stage runs that script, parameterized by `prepare.model_repo` + `prepare.model_filename` in `params.yaml`. The existing `package` stage transparently chains after `prepare` via the shared file path — its dep hash is unchanged because the file currently on disk is byte-identical to the HF artifact.

**Tech Stack:** Python 3.11, uv, DVC, `huggingface_hub` (already used in two sibling experiments).

**Spec:** `docs/specs/2026-04-16-yolo-hf-prepare-stage-design.md`.

**Reference precedent:** `experiments/temporal-models/tracking-fsm-baseline/scripts/prepare.py` and its `prepare` stage in `dvc.yaml` — same HF repo, same filename. Mirror it.

---

## Conventions (apply to every task)

- **Working directory:** `experiments/temporal-models/bbox-tube-temporal/`. All `uv run` and path references are relative to it unless otherwise stated.
- **Imports:** top of module only. No function-local imports. No `# noqa` to silence them.
- **Commits:** always `git add <explicit paths>` (no `git add -A`, no wildcards). No Claude/Anthropic co-author trailer.
- **Commit prefix:** match repo style — `feat(bbox-tube-temporal): ...`, `chore(bbox-tube-temporal): ...`.
- **Dependencies:** `uv add <pkg>` (never `uv pip install`).
- **Network:** `prepare.py` is the only stage that reaches the network. The currently-on-disk `best.pt` already matches the HF artifact, so the first `dvc repro prepare` is a no-op (script's `if dst_model.exists(): skip`).

---

## File structure

**Create:**
- `scripts/prepare.py` — thin CLI wrapper around `hf_hub_download`. ~40 LoC, mirror of `tracking-fsm-baseline/scripts/prepare.py`.

**Modify:**
- `pyproject.toml` — add `huggingface-hub>=0.24` to `[project].dependencies`.
- `params.yaml` — add a `prepare:` block with `model_repo` and `model_filename`.
- `dvc.yaml` — add a `prepare` stage as the new first stage (before `truncate`).

**No changes:**
- `scripts/package_model.py` — its `--yolo-weights-path` default of `data/01_raw/models/best.pt` keeps working unchanged.
- `.gitignore` — `data/01_raw/models/` stays git-ignored; DVC manages it via the `prepare` output cache.

---

## Task 1: Add the `huggingface-hub` dependency

**Files:**
- Modify: `experiments/temporal-models/bbox-tube-temporal/pyproject.toml`

**Rationale:** `huggingface_hub.hf_hub_download` is needed by the new `prepare.py`. The PyPI distribution name is `huggingface-hub` (hyphen); the import name is `huggingface_hub` (underscore). Match the version pin used in the sibling experiment `tracking-fsm-baseline` (`>=0.24`).

- [ ] **Step 1: Add the dependency via uv**

From `experiments/temporal-models/bbox-tube-temporal/`:

```bash
uv add 'huggingface-hub>=0.24'
```

Expected: command updates `pyproject.toml` and `uv.lock`, syncs the venv. The new line in `pyproject.toml` should appear inside `[project].dependencies`, alphabetically near `lightning`.

- [ ] **Step 2: Verify the import works**

```bash
uv run python -c "from huggingface_hub import hf_hub_download; print('ok')"
```

Expected output: `ok`

- [ ] **Step 3: Verify lint and existing tests still pass**

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
```

Expected: all green. (No code changes yet, so this is a sanity check that adding the dep didn't break the env.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(bbox-tube-temporal): add huggingface-hub dependency"
```

---

## Task 2: Add `scripts/prepare.py`

**Files:**
- Create: `experiments/temporal-models/bbox-tube-temporal/scripts/prepare.py`

**Rationale:** Self-contained CLI script that downloads one file from one HF repo into a target directory, skipping if the file already exists. Mirrors `tracking-fsm-baseline/scripts/prepare.py` verbatim — using a different shape (e.g. building it into `package.py`) would diverge from precedent without benefit. No tests: the sibling experiments don't test their `prepare.py` either, and the script is purely a thin wrapper around a third-party call that's exercised end-to-end by `dvc repro`.

- [ ] **Step 1: Create the script**

Write to `scripts/prepare.py`:

```python
"""Download YOLO model weights from HuggingFace Hub.

Fetches a single model file from a HuggingFace repository and saves it
to the local data directory. Skips the download if the file already exists.

Usage:
    uv run python scripts/prepare.py \
        --model-repo pyronear/yolo11s_nimble-narwhal_v6.0.0 \
        --model-filename best.pt \
        --output-model-dir data/01_raw/models
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLO model.")
    parser.add_argument(
        "--model-repo",
        type=str,
        required=True,
        help="HuggingFace model repo ID.",
    )
    parser.add_argument(
        "--model-filename",
        type=str,
        required=True,
        help="Model filename in the HuggingFace repo.",
    )
    parser.add_argument(
        "--output-model-dir",
        type=Path,
        required=True,
        help="Output directory for model weights (data/01_raw/models).",
    )
    args = parser.parse_args()

    args.output_model_dir.mkdir(parents=True, exist_ok=True)
    dst_model = args.output_model_dir / args.model_filename
    if dst_model.exists():
        logger.info("Model already exists at %s, skipping.", dst_model)
    else:
        logger.info("Downloading %s from %s...", args.model_filename, args.model_repo)
        hf_hub_download(
            repo_id=args.model_repo,
            filename=args.model_filename,
            local_dir=args.output_model_dir,
        )
        logger.info("Model saved to %s", dst_model)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script against the existing on-disk file**

```bash
uv run python scripts/prepare.py \
    --model-repo pyronear/yolo11s_nimble-narwhal_v6.0.0 \
    --model-filename best.pt \
    --output-model-dir data/01_raw/models
```

Expected output (one line):
```
INFO: Model already exists at data/01_raw/models/best.pt, skipping.
```

(No network I/O. If you instead see a `Downloading ...` line, the local file was missing or differently named — stop and investigate before continuing.)

- [ ] **Step 3: Verify lint passes**

```bash
uv run ruff check scripts/prepare.py
uv run ruff format --check scripts/prepare.py
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare.py
git commit -m "feat(bbox-tube-temporal): add prepare.py to fetch YOLO weights from HF Hub"
```

---

## Task 3: Add the `prepare` block to `params.yaml`

**Files:**
- Modify: `experiments/temporal-models/bbox-tube-temporal/params.yaml`

**Rationale:** Pin the HF source in params so DVC stage hashing reflects model-version changes. Place the block near the top, before `truncate`, to mirror the layout in `tracking-fsm-baseline/params.yaml` and to read top-down in pipeline order.

- [ ] **Step 1: Add the block to params.yaml**

Insert these lines immediately above the existing `truncate:` block (currently at the top of the file under the explanatory comment header):

```yaml
prepare:
  model_repo: "pyronear/yolo11s_nimble-narwhal_v6.0.0"
  model_filename: "best.pt"

```

(Leave one blank line between `prepare:` and `truncate:`.)

- [ ] **Step 2: Verify YAML still parses**

```bash
uv run python -c "import yaml; print(yaml.safe_load(open('params.yaml'))['prepare'])"
```

Expected output:
```
{'model_repo': 'pyronear/yolo11s_nimble-narwhal_v6.0.0', 'model_filename': 'best.pt'}
```

- [ ] **Step 3: Commit**

```bash
git add params.yaml
git commit -m "feat(bbox-tube-temporal): pin YOLO weights to pyronear/yolo11s_nimble-narwhal_v6.0.0"
```

---

## Task 4: Add the `prepare` stage to `dvc.yaml`

**Files:**
- Modify: `experiments/temporal-models/bbox-tube-temporal/dvc.yaml`

**Rationale:** The new stage must come before `truncate` so it appears first in the DAG. Its single output `data/01_raw/models` becomes a transitive dep of `package` (which already lists `data/01_raw/models/best.pt`); DVC chains them via the shared path with no further wiring.

- [ ] **Step 1: Insert the stage**

In `dvc.yaml`, immediately after `stages:` (line 1) and before the existing `truncate:` block, add:

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

(Two-space indent for the stage name, four-space indent for keys, blank line after the stage.)

- [ ] **Step 2: Validate DVC sees the new stage**

```bash
uv run dvc stage list 2>&1 | head -5
```

Expected: the first line includes `prepare ...`, before the `truncate@*` entries.

- [ ] **Step 3: Confirm only `prepare` is stale (no downstream re-run)**

```bash
uv run dvc status
```

Expected: `prepare` shows as a new stage (e.g. `changed deps:` or `changed outs:`). The `package@*` stages must NOT appear as needing re-run. If `package@*` shows up, stop — investigate before continuing.

```bash
uv run dvc repro --dry prepare
```

Expected: prints the `uv run python scripts/prepare.py ...` command and indicates only that stage will run.

- [ ] **Step 4: Run the stage to populate `dvc.lock`**

```bash
uv run dvc repro prepare
```

Expected: the script prints `INFO: Model already exists at data/01_raw/models/best.pt, skipping.`, then DVC reports the stage as completed and writes a `prepare` entry to `dvc.lock`. The `data/01_raw/models/` directory is now copied into the DVC cache (`.dvc/cache/`); this is a one-time operation.

- [ ] **Step 5: Reconfirm downstream stages did not invalidate**

```bash
uv run dvc status
```

Expected: clean for every stage downstream of `prepare`. The only change since pre-Task-1 should be the new `prepare` lock entry. If `package@*` or any other stage shows as stale, stop and investigate the dep hashes.

- [ ] **Step 6: Lint check (DVC YAML is parsed by ruff-irrelevant tooling, but we still run the project lint sweep)**

```bash
uv run ruff check .
uv run ruff format --check .
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add dvc.yaml dvc.lock
git commit -m "feat(bbox-tube-temporal): add prepare stage to fetch YOLO weights via DVC"
```

---

## Task 5: Final verification

**Files:** none modified.

**Rationale:** End-to-end check that the pipeline is healthy and the change is contained. No changes here — this is the gate before merging.

- [ ] **Step 1: Run the test suite**

```bash
uv run pytest tests/ -v
```

Expected: all green.

- [ ] **Step 2: Confirm pipeline is in sync**

```bash
uv run dvc status
```

Expected output: `Data and pipelines are up to date.` (or equivalent — no stages listed as stale).

- [ ] **Step 3: Sanity-check the recorded HF source**

```bash
uv run dvc params diff
```

Expected: shows `prepare.model_repo = pyronear/yolo11s_nimble-narwhal_v6.0.0` and `prepare.model_filename = best.pt` as added (versus the previous workspace state).

- [ ] **Step 4: Skim the diff against `main`**

```bash
git diff --stat main..HEAD
```

Expected files (6 from this plan, plus the spec + plan docs themselves):
- `pyproject.toml`
- `uv.lock`
- `scripts/prepare.py` (new)
- `params.yaml`
- `dvc.yaml`
- `dvc.lock`
- `docs/specs/2026-04-16-yolo-hf-prepare-stage-design.md` (new, from the spec stage)
- `docs/plans/2026-04-16-yolo-hf-prepare-stage-plan.md` (new, this file)

No source files outside `experiments/temporal-models/bbox-tube-temporal/` should appear in the diff.

If anything outside `experiments/temporal-models/bbox-tube-temporal/` appears in the diff, stop and investigate.
