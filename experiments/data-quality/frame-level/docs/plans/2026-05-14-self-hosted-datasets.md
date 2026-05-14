# Self-hosted raw datasets — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the six frozen DVC imports under `data/01_raw/datasets/` with three self-hosted `dvc add` files (one per split) and a `make refresh-datasets` flow that re-imports from pyro-dataset on demand.

**Architecture:** Three `<split>.dvc` files at `data/01_raw/datasets/`, each tracking the full split directory (images + labels). A `SOURCE.json` next to them records the upstream pyro-dataset version and commit. A Python script orchestrates `git clone` + `dvc pull` of pyro-dataset, file sync into this experiment, and `dvc add` + `dvc push`. The audit-app's version badge is repointed to read `SOURCE.json` instead of the now-removed `repo.rev` field in the old import `.dvc` files.

**Tech Stack:** Python 3.11, DVC, uv, pytest, ruff. Standard library only for the refresh script (subprocess, shutil, json, tempfile, argparse, pathlib).

**Spec:** `docs/specs/2026-05-14-self-hosted-datasets-design.md`

**Working directory for every command:** `experiments/data-quality/frame-level/` (the experiment root). All paths in this plan are relative to it unless prefixed with `/`.

**Sequencing note:** Tasks 1–2 are pure code changes and can land first. Task 3 (migration) depends on a colleague's in-flight push to pyro-dataset's remotes — do not start Task 3 until that push is confirmed complete and `dvc pull` of pyro-dataset at v4.0.0 succeeds end-to-end.

---

## Task 1: Repoint `read_dataset_version` to `SOURCE.json`

**Files:**
- Modify: `src/data_quality_frame_level/audit_app/dataset_version.py`
- Modify: `tests/test_audit_app_dataset_version.py`

The current implementation walks `<split>/{images,labels}.dvc` and reads `deps[0].repo.rev`. Under the new design those `.dvc` files go away. The replacement reads `<datasets_root>/SOURCE.json` and returns its `pyro_dataset_version` field. Signature stays `read_dataset_version(datasets_root: Path) -> str | None`. The `"mixed: <r1>, <r2>"` case disappears (single source of truth).

Callers in `src/data_quality_frame_level/audit_app/main.py:78` and the frontend (`static/app.js`) require no changes.

- [ ] **Step 1.1: Rewrite the test file**

Replace the entire contents of `tests/test_audit_app_dataset_version.py` with:

```python
import json
from pathlib import Path

from data_quality_frame_level.audit_app.dataset_version import read_dataset_version


def test_returns_none_when_root_missing(tmp_path: Path):
    assert read_dataset_version(tmp_path / "nope") is None


def test_returns_none_when_source_json_missing(tmp_path: Path):
    assert read_dataset_version(tmp_path) is None


def test_returns_version_from_source_json(tmp_path: Path):
    (tmp_path / "SOURCE.json").write_text(
        json.dumps(
            {
                "pyro_dataset_version": "v4.0.0",
                "pyro_dataset_commit": "4e16c464edda7400b0ac738c4f45f8d8e50fa735",
                "refreshed_at": "2026-05-14T12:30:00+02:00",
            }
        )
    )
    assert read_dataset_version(tmp_path) == "v4.0.0"


def test_returns_none_when_source_json_malformed(tmp_path: Path):
    (tmp_path / "SOURCE.json").write_text("{not valid json")
    assert read_dataset_version(tmp_path) is None


def test_returns_none_when_version_field_absent(tmp_path: Path):
    (tmp_path / "SOURCE.json").write_text(json.dumps({"other_field": "x"}))
    assert read_dataset_version(tmp_path) is None
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
uv run pytest tests/test_audit_app_dataset_version.py -v
```

Expected: all five tests fail (the old implementation reads `.dvc` files, not `SOURCE.json`).

- [ ] **Step 1.3: Rewrite the implementation**

Replace the entire contents of `src/data_quality_frame_level/audit_app/dataset_version.py` with:

```python
"""Parse the pyro-dataset revision from ``SOURCE.json``.

``data/01_raw/datasets/SOURCE.json`` is written by
``scripts/refresh_datasets.py`` and records which pyro-dataset version
the raw splits were last imported from. The audit-app header surfaces
that version so reviewers know which dataset they are looking at.
"""

import json
from pathlib import Path


def read_dataset_version(datasets_root: Path) -> str | None:
    """Return the pyro-dataset version recorded in ``SOURCE.json``.

    Returns ``None`` when ``datasets_root`` is missing, ``SOURCE.json``
    is missing, the JSON is malformed, or the ``pyro_dataset_version``
    field is absent. Otherwise returns the version string verbatim.
    """
    if not datasets_root.is_dir():
        return None
    source_path = datasets_root / "SOURCE.json"
    if not source_path.is_file():
        return None
    try:
        payload = json.loads(source_path.read_text())
    except json.JSONDecodeError:
        return None
    version = payload.get("pyro_dataset_version")
    if not isinstance(version, str):
        return None
    return version
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
uv run pytest tests/test_audit_app_dataset_version.py -v
```

Expected: all five tests pass.

- [ ] **Step 1.5: Run lint**

```bash
uv run ruff check src/data_quality_frame_level/audit_app/dataset_version.py tests/test_audit_app_dataset_version.py
uv run ruff format --check src/data_quality_frame_level/audit_app/dataset_version.py tests/test_audit_app_dataset_version.py
```

Expected: both pass.

- [ ] **Step 1.6: Commit**

```bash
git add src/data_quality_frame_level/audit_app/dataset_version.py tests/test_audit_app_dataset_version.py
git commit -m "refactor(data-quality/frame-level): read pyro-dataset version from SOURCE.json

Decouple read_dataset_version from .dvc repo.rev parsing in
preparation for replacing frozen imports with self-hosted dvc add."
```

---

## Task 2: Add `scripts/refresh_datasets.py` and `make refresh-datasets`

**Files:**
- Create: `scripts/refresh_datasets.py`
- Create: `tests/test_refresh_datasets.py`
- Modify: `Makefile`

The script orchestrates: clone pyro-dataset shallow at a tag → `dvc pull` inside that clone → copy split directories into this experiment → `dvc add` + `dvc push` → write `SOURCE.json`. The orchestration logic is hard to unit-test (network + subprocess + DVC), so factor the path-mapping rules into a pure function `plan_copies(pyro_root, dest_root)` and unit-test that.

The orchestration is exercised by Task 3's migration run. No mocking of subprocess in tests.

Confirmed source layout in pyro-dataset (from `git show v4.0.0:dvc.yaml`):

- `data/processed/yolo_test/` (output of `merge_yolo_dataset` stage) lives on the `awspyronear-private` remote.
- `data/processed/yolo_train_val/` lives on the default `awspyronear` remote.

`dvc pull` inside pyro-dataset's clone reads `dvc.yaml`'s per-out `remote:` field, so it'll route automatically — the operator just needs AWS creds for both buckets. No remote selection logic needed in our script.

Pyro-dataset's `yolo_test` and `yolo_train_val` directories each contain `images/` and `labels/` subdirectories with per-split sub-folders. The exact layout to verify and copy is:

- `pyro-dataset/data/processed/yolo_test/images/test/*` → `data/01_raw/datasets/test/images/`
- `pyro-dataset/data/processed/yolo_test/labels/test/*` → `data/01_raw/datasets/test/labels/`
- `pyro-dataset/data/processed/yolo_train_val/images/{train,val}/*` → `data/01_raw/datasets/{train,val}/images/`
- `pyro-dataset/data/processed/yolo_train_val/labels/{train,val}/*` → `data/01_raw/datasets/{train,val}/labels/`

- [ ] **Step 2.1: Write the failing path-mapping test**

Create `tests/test_refresh_datasets.py`:

```python
from pathlib import Path

from scripts.refresh_datasets import plan_copies


def test_plan_copies_emits_six_source_dest_pairs(tmp_path: Path):
    pyro = tmp_path / "pyro-dataset"
    dest = tmp_path / "datasets"
    pairs = plan_copies(pyro, dest)

    expected = {
        (
            pyro / "data" / "processed" / "yolo_test" / "images" / "test",
            dest / "test" / "images",
        ),
        (
            pyro / "data" / "processed" / "yolo_test" / "labels" / "test",
            dest / "test" / "labels",
        ),
        (
            pyro / "data" / "processed" / "yolo_train_val" / "images" / "train",
            dest / "train" / "images",
        ),
        (
            pyro / "data" / "processed" / "yolo_train_val" / "labels" / "train",
            dest / "train" / "labels",
        ),
        (
            pyro / "data" / "processed" / "yolo_train_val" / "images" / "val",
            dest / "val" / "images",
        ),
        (
            pyro / "data" / "processed" / "yolo_train_val" / "labels" / "val",
            dest / "val" / "labels",
        ),
    }
    assert set(pairs) == expected
```

- [ ] **Step 2.2: Run test to verify it fails (module doesn't exist)**

```bash
uv run pytest tests/test_refresh_datasets.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.refresh_datasets'` (or `ImportError`).

Note: the existing `scripts/` directory has no `__init__.py`. If the import fails for that reason, add an empty `scripts/__init__.py` in this step and re-run; otherwise leave it alone.

- [ ] **Step 2.3: Implement the script**

Create `scripts/refresh_datasets.py`:

```python
"""Refresh raw datasets from pyro-dataset at a specific version.

Clones pyro-dataset shallow at the given tag, dvc-pulls inside it,
copies the YOLO split directories into this experiment, runs
``dvc add`` + ``dvc push``, and writes ``SOURCE.json``.

Run from the experiment root:

    PYRO_DATASET_VERSION=v4.0.0 make refresh-datasets

or directly:

    uv run python scripts/refresh_datasets.py --version v4.0.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PYRO_DATASET_REPO = "https://github.com/pyronear/pyro-dataset.git"
SPLITS = ("train", "val", "test")
MODALITIES = ("images", "labels")


def plan_copies(pyro_root: Path, dest_root: Path) -> list[tuple[Path, Path]]:
    """Return the (src, dst) directory pairs to copy.

    Pure function — no I/O. Captures the source-path layout inside
    pyro-dataset at v4.0.0 (verified against ``git show v4.0.0:dvc.yaml``).
    """
    pairs: list[tuple[Path, Path]] = []
    for split in SPLITS:
        upstream_dir = "yolo_test" if split == "test" else "yolo_train_val"
        for modality in MODALITIES:
            src = (
                pyro_root
                / "data"
                / "processed"
                / upstream_dir
                / modality
                / split
            )
            dst = dest_root / split / modality
            pairs.append((src, dst))
    return pairs


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess command, streaming output, and abort on failure."""
    print(f"$ {' '.join(cmd)}" + (f"  (cwd={cwd})" if cwd else ""))
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_and_pull(version: str, work_dir: Path) -> Path:
    """Shallow-clone pyro-dataset at ``version`` and dvc-pull inside it."""
    clone_dir = work_dir / "pyro-dataset"
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            version,
            PYRO_DATASET_REPO,
            str(clone_dir),
        ]
    )
    run(["dvc", "pull"], cwd=clone_dir)
    return clone_dir


def resolve_commit(clone_dir: Path) -> str:
    """Return the resolved commit hash of HEAD in the clone."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clone_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def copy_splits(pyro_root: Path, dest_root: Path) -> None:
    """Replace the contents of each ``dest_root/<split>/<modality>``."""
    for src, dst in plan_copies(pyro_root, dest_root):
        if not src.is_dir():
            raise SystemExit(f"missing expected source dir: {src}")
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        print(f"copied {src} -> {dst}")


def dvc_add_and_push(experiment_root: Path) -> None:
    """``dvc add`` each split dir and push to the configured remote."""
    targets = [f"data/01_raw/datasets/{s}" for s in SPLITS]
    run(["uv", "run", "dvc", "add", *targets], cwd=experiment_root)
    run(["uv", "run", "dvc", "push", *[f"{t}.dvc" for t in targets]], cwd=experiment_root)


def write_source_json(
    dest_root: Path,
    version: str,
    commit: str,
    note: str | None,
) -> None:
    """Write ``SOURCE.json`` recording the upstream provenance."""
    payload: dict[str, str] = {
        "pyro_dataset_version": version,
        "pyro_dataset_commit": commit,
        "refreshed_at": datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds"),
    }
    if note:
        payload["note"] = note
    source_path = dest_root / "SOURCE.json"
    source_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {source_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="pyro-dataset git tag or branch (e.g. v4.0.0)",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="optional free-form note to record in SOURCE.json",
    )
    args = parser.parse_args()

    experiment_root = Path(__file__).resolve().parent.parent
    dest_root = experiment_root / "data" / "01_raw" / "datasets"

    with tempfile.TemporaryDirectory(prefix="refresh-datasets-") as tmp:
        work_dir = Path(tmp)
        clone_dir = clone_and_pull(args.version, work_dir)
        commit = resolve_commit(clone_dir)
        copy_splits(clone_dir, dest_root)
        dvc_add_and_push(experiment_root)
        write_source_json(dest_root, args.version, commit, args.note)

    print()
    print("Refresh complete. Review and commit:")
    print("  git status data/01_raw/datasets/")
    print("  git add data/01_raw/datasets/*.dvc data/01_raw/datasets/SOURCE.json \\")
    print("          data/01_raw/datasets/.gitignore")
    print("  uv run dvc commit  # if dvc.lock needs updating")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2.4: Run the unit test to verify it passes**

```bash
uv run pytest tests/test_refresh_datasets.py -v
```

Expected: PASS.

- [ ] **Step 2.5: Add Makefile target**

Open `Makefile`. Add `refresh-datasets` to the `.PHONY` line and append the target at the bottom of the file.

In `Makefile` line 1, change:

```makefile
.PHONY: install lint format test help notebook audit-app audit-export audit-publish audit-summary export-viewer
```

to:

```makefile
.PHONY: install lint format test help notebook audit-app audit-export audit-publish audit-summary export-viewer refresh-datasets
```

Append at the bottom of `Makefile`:

```makefile
refresh-datasets: ## Refresh raw datasets from pyro-dataset. Requires PYRO_DATASET_VERSION=<tag>.
	@test -n "$(PYRO_DATASET_VERSION)" || { echo "PYRO_DATASET_VERSION is required (e.g. make refresh-datasets PYRO_DATASET_VERSION=v4.0.0)"; exit 2; }
	uv run python scripts/refresh_datasets.py --version "$(PYRO_DATASET_VERSION)"
```

- [ ] **Step 2.6: Smoke-test the Makefile guard**

```bash
make refresh-datasets 2>&1 | head -5
```

Expected: prints the `PYRO_DATASET_VERSION is required` error and exits non-zero. Confirms the guard works without running the actual refresh.

- [ ] **Step 2.7: Lint**

```bash
uv run ruff check scripts/refresh_datasets.py tests/test_refresh_datasets.py
uv run ruff format --check scripts/refresh_datasets.py tests/test_refresh_datasets.py
```

Expected: both pass. If `ruff format --check` fails, run `uv run ruff format scripts/refresh_datasets.py tests/test_refresh_datasets.py` and re-check.

- [ ] **Step 2.8: Commit**

```bash
git add scripts/refresh_datasets.py tests/test_refresh_datasets.py Makefile
git commit -m "feat(data-quality/frame-level): add refresh-datasets script + make target

Clone pyro-dataset shallow at \$PYRO_DATASET_VERSION, dvc-pull,
copy split directories into data/01_raw/datasets/, then dvc add +
dvc push under this experiment's remote. Writes SOURCE.json with
the resolved upstream commit. Path-mapping logic factored out as
plan_copies() and unit-tested."
```

---

## Task 3: Migrate from frozen imports to self-hosted datasets

**Files:**
- Delete: `data/01_raw/datasets/{train,val,test}/{images,labels}.dvc` (six files)
- Create (via `dvc add`): `data/01_raw/datasets/{train,val,test}.dvc` (three files)
- Create (via script): `data/01_raw/datasets/SOURCE.json`
- Possibly modify: `dvc.lock` (if dataset dir hashes change)

**Prerequisite:** confirm that pyro-dataset's missing v4.0.0 data is pushed and reachable. Quick check:

```bash
aws s3 ls s3://pyro-test-datasets/dvc/files/md5/3f/5c9c37a177a7992fba42525a388a2d.dir
aws s3 ls s3://pyro-dataset-dvc-v2/dvc/files/md5/cf/be8fd818f3d41a608c411b98832611.dir
```

Both should print one line with a non-zero size. If either errors or returns nothing, stop and resolve upstream before continuing.

- [ ] **Step 3.1: Delete the six old frozen-import `.dvc` files**

```bash
git rm data/01_raw/datasets/train/images.dvc \
       data/01_raw/datasets/train/labels.dvc \
       data/01_raw/datasets/val/images.dvc \
       data/01_raw/datasets/val/labels.dvc \
       data/01_raw/datasets/test/images.dvc \
       data/01_raw/datasets/test/labels.dvc
```

Do not commit yet — Step 3.4 commits the migration as one atomic change.

- [ ] **Step 3.2: Run the refresh script**

```bash
make refresh-datasets PYRO_DATASET_VERSION=v4.0.0
```

Expected: clones pyro-dataset, dvc-pulls (hitting both `s3://pyro-dataset-dvc-v2/dvc/` and `s3://pyro-test-datasets/dvc/`), copies six directories, runs `dvc add` on three split dirs, pushes to `s3://pyro-vision-rd/dvc/experiments/data-quality/frame-level/`, writes `SOURCE.json`.

If `dvc pull` warns `No file hash info found for <path>`, **stop**. That means upstream is still incomplete. Do not continue.

- [ ] **Step 3.3: Verify post-refresh state**

```bash
ls data/01_raw/datasets/
cat data/01_raw/datasets/SOURCE.json
uv run dvc status data/01_raw/datasets/train.dvc data/01_raw/datasets/val.dvc data/01_raw/datasets/test.dvc
uv run dvc status -c data/01_raw/datasets/train.dvc data/01_raw/datasets/val.dvc data/01_raw/datasets/test.dvc
```

Expected:
- `ls` shows `SOURCE.json`, `train.dvc`, `val.dvc`, `test.dvc`, and the split directories (no `images.dvc`/`labels.dvc` per-split files).
- `SOURCE.json` contains the version, the resolved commit `4e16c464edda7400b0ac738c4f45f8d8e50fa735`, and a `refreshed_at` timestamp.
- `dvc status` (workspace vs cache): "Data and pipelines are up to date."
- `dvc status -c` (cache vs remote): "Cache and remote 's3remote' are in sync."

- [ ] **Step 3.4: Reconcile `dvc.lock` dataset deps**

The predict stages in `dvc.yaml` declare `data/01_raw/datasets/{train,val,test}` as directory deps. `dvc.lock` currently pins their dir hashes (e.g. `data/01_raw/datasets/test` md5 `3b8457851c760a78b80b90e5d8bd813f.dir`, nfiles 5283). After the refresh, the dir hashes may differ if the new tree differs from the old by even one file.

Run:

```bash
uv run dvc status
```

If the predict stages are reported as out-of-date solely because of dataset dep hash drift (look for `changed deps: data/01_raw/datasets/<split>`), update `dvc.lock` without recomputing:

```bash
uv run dvc commit dvc.yaml
```

Expected: prompts for confirmation, then updates `dvc.lock` dep hashes to the new dir hashes.

If `dvc status` reports actual stage outputs as out-of-date (i.e. predict outputs need rebuilding), stop and consult before forcing — that indicates the dataset content meaningfully changed and the model predictions need rerunning, which is out of scope for this plan.

- [ ] **Step 3.5: Verify the experiment's CI-relevant checks still pass**

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format --check .
```

Expected: all pass. The audit-app `read_dataset_version` test now reads `SOURCE.json` (Task 1), and `SOURCE.json` now exists, so the audit-app continues to function.

- [ ] **Step 3.6: Smoke-test the audit-app version badge**

Launch the audit app briefly:

```bash
make audit-app &
APP_PID=$!
sleep 5
curl -s http://localhost:8000/api/contexts | python -c "import json,sys; d=json.load(sys.stdin); print('dataset_version=', d.get('dataset_version'))"
kill $APP_PID
```

Expected: prints `dataset_version= v4.0.0`. Confirms the rewritten `read_dataset_version` reads the new `SOURCE.json` correctly through the live app.

- [ ] **Step 3.7: Commit the migration**

```bash
git status data/01_raw/datasets/
```

Expected status: six `.dvc` files deleted, three new `.dvc` files added, `SOURCE.json` added, `.gitignore` files updated (DVC may have updated `data/01_raw/datasets/.gitignore` to include `/train`, `/val`, `/test`, and the per-split `.gitignore` files may be unchanged or empty). `dvc.lock` modified.

Stage explicitly (do **not** use `git add -A`):

```bash
git add data/01_raw/datasets/train.dvc \
        data/01_raw/datasets/val.dvc \
        data/01_raw/datasets/test.dvc \
        data/01_raw/datasets/SOURCE.json \
        data/01_raw/datasets/.gitignore \
        dvc.lock
# the deletes from Step 3.1 are already staged via `git rm`
```

If `data/01_raw/datasets/{train,val,test}/.gitignore` files have content changes (e.g. became empty because `/images` and `/labels` are no longer needed as gitignores under the new layout), include them too:

```bash
git add data/01_raw/datasets/train/.gitignore \
        data/01_raw/datasets/val/.gitignore \
        data/01_raw/datasets/test/.gitignore
```

Then commit:

```bash
git commit -m "feat(data-quality/frame-level): self-host raw datasets via dvc add

Replace six frozen imports of pyro-dataset under
data/01_raw/datasets/<split>/{images,labels}.dvc with three
per-split dvc-add files. Source provenance recorded in
SOURCE.json. Colleagues no longer need pyro-dataset bucket
access to pull this experiment's raw inputs."
```

---

## Self-review checklist (run after writing the plan)

- [ ] **Spec coverage:** every section of `docs/specs/2026-05-14-self-hosted-datasets-design.md` has a task that implements it.
  - §4 Target layout → Task 3 (refresh script produces this layout)
  - §4a Audit-app version badge → Task 1
  - §5 Refresh flow → Task 2
  - §6 Migration → Task 3
  - §7 Testing → Tasks 1.4, 2.4, 3.5, 3.6
  - §8 Risks (dir-hash drift) → Step 3.4
- [ ] **No placeholders:** scan for "TBD", "TODO", "implement later".
- [ ] **Type/name consistency:** `plan_copies`, `read_dataset_version`, `PYRO_DATASET_VERSION` used consistently throughout.
