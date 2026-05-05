# Export flow — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-file export with the four-file layout (`labels/`, `manifest.json`, `pending.json`, `provenance.json`), add a passive DVC stale-state warning surfaced via `/api/contexts` and rendered as a dismissible banner.

**Architecture:** Three pure writer functions in `export.py` (manifest+labels, pending, provenance), each unit-tested with hand-built fixtures. The CLI orchestrates: gathers git/DVC/params metadata, calls each writer in turn. A new persistence helper hashes the local `review.json` and compares with the `.dvc`-tracked md5; the result rides on `/api/contexts`.

**Tech Stack:** Python 3.11, FastAPI, pytest, vanilla JS, Tailwind via CDN. Reuses existing `BBox`, `ReviewState`, `SampleReview`.

**Spec:** [`2026-05-05-export-flow-design.md`](../specs/2026-05-05-export-flow-design.md)

---

## File structure

All paths relative to `experiments/data-quality/frame-level/`.

| File | Responsibility |
|---|---|
| `src/data_quality_frame_level/review_app/export.py` | Split into three writer functions + `ProvenanceInput` dataclass |
| `src/data_quality_frame_level/review_app/persistence.py` | Add `dvc_warning_for_review` helper |
| `src/data_quality_frame_level/review_app/main.py` | Add `dvc_warnings` array to `/api/contexts` response |
| `src/data_quality_frame_level/review_app/static/index.html` | Add `<div id="dvc-banner">` above header |
| `src/data_quality_frame_level/review_app/static/app.js` | Render dismissible banners from contexts response |
| `scripts/export_review_app.py` | Gather git/DVC/params metadata; call all three writers |
| `tests/test_review_app_export.py` | New tests: contributors, pending, provenance |
| `tests/test_review_app_persistence.py` | New tests: dvc_warning_for_review |
| `tests/test_review_app_main.py` | New test: dvc_warnings in /api/contexts |

---

## Task 1: Refactor `export.py` — manifest writer with `contributors` field

**Files:**
- Modify: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/export.py`
- Modify: `experiments/data-quality/frame-level/tests/test_review_app_export.py`

- [ ] **Step 1: Write a failing test for `contributors` field**

Append to `tests/test_review_app_export.py`:

```python
def test_export_manifest_contributors_is_sorted_unique_reviewers(tmp_path: Path):
    review = ReviewState(
        model="m",
        split="val",
        samples={
            "stem_a": SampleReview(
                status="reviewed", bboxes=[_bb(0.6, 0.6)], reviewer="mateo"
            ),
            "stem_b": SampleReview(
                status="reviewed", bboxes=[_bb(0.7, 0.7)], reviewer="arthur"
            ),
            "stem_c": SampleReview(
                status="reviewed", bboxes=[_bb(0.8, 0.8)], reviewer="arthur"
            ),
            "stem_d": SampleReview(status="reviewed", bboxes=[_bb(0.9, 0.9)]),
        },
    )
    originals = {st: [] for st in review.samples}
    out = tmp_path / "10_export" / "m" / "val"
    write_manifest_and_labels(review=review, originals=originals, out_dir=out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["contributors"] == ["arthur", "mateo"]
```

Add `write_manifest_and_labels` to the import line in the test file:

```python
from data_quality_frame_level.review_app.export import (
    DiffCounts,
    compute_diff,
    export_corrections,
    write_manifest_and_labels,
)
```

- [ ] **Step 2: Run, verify it fails**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run pytest tests/test_review_app_export.py::test_export_manifest_contributors_is_sorted_unique_reviewers -v
```

Expected: `ImportError: cannot import name 'write_manifest_and_labels'`.

- [ ] **Step 3: Implement `write_manifest_and_labels`**

In `src/data_quality_frame_level/review_app/export.py`, replace the body of `export_corrections` and add the new function. Final file shape:

```python
"""Export corrected GT to a YOLO-format patch + manifest + provenance.

The export directory contains four siblings:

  labels/<stem>.txt     # corrected YOLO labels (only-changed frames)
  manifest.json         # apply contract — pyro-dataset reads this
  pending.json          # unclear-status frames for second-opinion review
  provenance.json       # audit-side context for reproducibility

Each writer here is a pure function over its inputs; the CLI
(``scripts/export_review_app.py``) gathers the git/DVC/params context
and feeds it in.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from data_quality_frame_level.dataset import BBox
from data_quality_frame_level.review_app.matching import iou
from data_quality_frame_level.review_app.persistence import ReviewState

UNCHANGED_IOU = 0.95


@dataclass(frozen=True)
class DiffCounts:
    added: int
    removed: int
    modified: int

    @property
    def is_change(self) -> bool:
        return self.added + self.removed + self.modified > 0


@dataclass(frozen=True)
class ProvenanceInput:
    audit_repo: str
    audit_commit: str
    audit_branch: str
    experiment: str
    thresholds: dict[str, float]
    predictions_path: str
    predictions_md5: str


def compute_diff(*, original: list[BBox], corrected: list[BBox]) -> DiffCounts:
    matched_orig: set[int] = set()
    matched_corr: set[int] = set()
    modified = 0
    candidates = sorted(
        (
            (i, j, iou(o, c))
            for i, o in enumerate(original)
            for j, c in enumerate(corrected)
        ),
        key=lambda x: x[2],
        reverse=True,
    )
    for oi, cj, score in candidates:
        if score == 0.0:
            break
        if oi in matched_orig or cj in matched_corr:
            continue
        matched_orig.add(oi)
        matched_corr.add(cj)
        if score < UNCHANGED_IOU:
            modified += 1
    removed = len(original) - len(matched_orig)
    added = len(corrected) - len(matched_corr)
    return DiffCounts(added=added, removed=removed, modified=modified)


def _write_yolo_txt(path: Path, bboxes: list[BBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{b.class_id} {b.cx} {b.cy} {b.w} {b.h}" for b in bboxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_manifest_and_labels(
    *,
    review: ReviewState,
    originals: dict[str, list[BBox]],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"
    changed: list[dict] = []
    totals = {"changed": 0, "added": 0, "removed": 0, "modified": 0}
    for stem in sorted(review.samples):
        sample = review.samples[stem]
        if sample.status != "reviewed":
            continue
        original = originals.get(stem, [])
        diff = compute_diff(original=original, corrected=sample.bboxes)
        if not diff.is_change:
            continue
        _write_yolo_txt(labels_dir / f"{stem}.txt", sample.bboxes)
        changed.append(
            {
                "stem": stem,
                "added": diff.added,
                "removed": diff.removed,
                "modified": diff.modified,
                "reviewer": sample.reviewer,
                "note": sample.note,
            }
        )
        totals["changed"] += 1
        totals["added"] += diff.added
        totals["removed"] += diff.removed
        totals["modified"] += diff.modified
    contributors = sorted({c["reviewer"] for c in changed if c["reviewer"]})
    manifest = {
        "version": 1,
        "model": review.model,
        "split": review.split,
        "exported_at": _now_iso(),
        "contributors": contributors,
        "changed": changed,
        "totals": totals,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_pending(*, review: ReviewState, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        {
            "stem": stem,
            "reviewer": sample.reviewer,
            "note": sample.note,
        }
        for stem, sample in sorted(review.samples.items())
        if sample.status == "unclear"
    ]
    payload = {
        "version": 1,
        "model": review.model,
        "split": review.split,
        "exported_at": _now_iso(),
        "pending": pending,
    }
    (out_dir / "pending.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def write_provenance(
    *,
    prov: ProvenanceInput,
    model: str,
    split: str,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "audit_repo": prov.audit_repo,
        "audit_commit": prov.audit_commit,
        "audit_branch": prov.audit_branch,
        "experiment": prov.experiment,
        "model": model,
        "split": split,
        "thresholds": prov.thresholds,
        "predictions_path": prov.predictions_path,
        "predictions_md5": prov.predictions_md5,
        "exported_at": _now_iso(),
    }
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def export_corrections(
    *,
    review: ReviewState,
    originals: dict[str, list[BBox]],
    out_dir: Path,
    provenance: ProvenanceInput | None = None,
) -> dict:
    """Orchestrator: write all four files. Returns the manifest payload."""
    manifest = write_manifest_and_labels(
        review=review, originals=originals, out_dir=out_dir
    )
    write_pending(review=review, out_dir=out_dir)
    if provenance is not None:
        write_provenance(
            prov=provenance, model=review.model, split=review.split, out_dir=out_dir
        )
    return manifest
```

- [ ] **Step 4: Run the new test, verify pass**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run pytest tests/test_review_app_export.py -v
```

Expected: existing tests still pass + new contributors test passes.

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/export.py experiments/data-quality/frame-level/tests/test_review_app_export.py
git commit -m "feat(data-quality/frame-level): split exporter; add contributors field"
```

---

## Task 2: `pending.json` writer

**Files:**
- Modify: `experiments/data-quality/frame-level/tests/test_review_app_export.py`

(`export.py` already has `write_pending` from Task 1; this task just adds tests.)

- [ ] **Step 1: Write failing tests for pending**

Append to `tests/test_review_app_export.py`:

```python
def test_export_pending_includes_only_unclear(tmp_path: Path):
    review = ReviewState(
        model="m",
        split="val",
        samples={
            "stem_a": SampleReview(status="reviewed", bboxes=[_bb(0.5, 0.5)]),
            "stem_b": SampleReview(
                status="unclear", bboxes=[], reviewer="arthur", note="check this"
            ),
            "stem_c": SampleReview(
                status="unclear", bboxes=[], reviewer="mateo"
            ),
        },
    )
    out = tmp_path / "10_export" / "m" / "val"
    write_pending(review=review, out_dir=out)
    pending = json.loads((out / "pending.json").read_text())
    assert pending["version"] == 1
    assert pending["model"] == "m"
    assert pending["split"] == "val"
    assert [p["stem"] for p in pending["pending"]] == ["stem_b", "stem_c"]
    by_stem = {p["stem"]: p for p in pending["pending"]}
    assert by_stem["stem_b"]["reviewer"] == "arthur"
    assert by_stem["stem_b"]["note"] == "check this"
    assert by_stem["stem_c"]["reviewer"] == "mateo"


def test_export_pending_empty_when_no_unclear(tmp_path: Path):
    review = ReviewState(
        model="m",
        split="val",
        samples={
            "stem_a": SampleReview(status="reviewed", bboxes=[_bb(0.5, 0.5)]),
        },
    )
    out = tmp_path / "10_export" / "m" / "val"
    write_pending(review=review, out_dir=out)
    pending = json.loads((out / "pending.json").read_text())
    assert pending["pending"] == []
```

Update the import block in the test file to include `write_pending`:

```python
from data_quality_frame_level.review_app.export import (
    DiffCounts,
    compute_diff,
    export_corrections,
    write_manifest_and_labels,
    write_pending,
)
```

- [ ] **Step 2: Run, verify pass**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run pytest tests/test_review_app_export.py -v
```

Expected: all tests pass (the implementation already exists).

- [ ] **Step 3: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/data-quality/frame-level/tests/test_review_app_export.py
git commit -m "test(data-quality/frame-level): pending.json export"
```

---

## Task 3: `provenance.json` writer tests

**Files:**
- Modify: `experiments/data-quality/frame-level/tests/test_review_app_export.py`

- [ ] **Step 1: Write failing test for provenance**

Append:

```python
def test_export_provenance_writes_all_fields(tmp_path: Path):
    prov = ProvenanceInput(
        audit_repo="pyronear/vision-rd",
        audit_commit="abc1234",
        audit_branch="arthur/feature",
        experiment="experiments/data-quality/frame-level",
        thresholds={"conf": 0.05, "iou": 0.05, "review_conf": 0.35},
        predictions_path="data/07_model_output/m/val/predictions.json",
        predictions_md5="deadbeefcafe",
    )
    out = tmp_path / "10_export" / "m" / "val"
    write_provenance(prov=prov, model="m", split="val", out_dir=out)
    payload = json.loads((out / "provenance.json").read_text())
    assert payload["version"] == 1
    assert payload["audit_repo"] == "pyronear/vision-rd"
    assert payload["audit_commit"] == "abc1234"
    assert payload["audit_branch"] == "arthur/feature"
    assert payload["experiment"] == "experiments/data-quality/frame-level"
    assert payload["model"] == "m"
    assert payload["split"] == "val"
    assert payload["thresholds"] == {"conf": 0.05, "iou": 0.05, "review_conf": 0.35}
    assert payload["predictions_path"] == "data/07_model_output/m/val/predictions.json"
    assert payload["predictions_md5"] == "deadbeefcafe"
    assert "exported_at" in payload
```

Update the import block to include `ProvenanceInput` and `write_provenance`:

```python
from data_quality_frame_level.review_app.export import (
    DiffCounts,
    ProvenanceInput,
    compute_diff,
    export_corrections,
    write_manifest_and_labels,
    write_pending,
    write_provenance,
)
```

- [ ] **Step 2: Run, verify pass**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run pytest tests/test_review_app_export.py -v
```

Expected: all tests pass (implementation in Task 1).

- [ ] **Step 3: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/data-quality/frame-level/tests/test_review_app_export.py
git commit -m "test(data-quality/frame-level): provenance.json export"
```

---

## Task 4: CLI integration — gather git/DVC/params context

**Files:**
- Modify: `experiments/data-quality/frame-level/scripts/export_review_app.py`

- [ ] **Step 1: Rewrite the CLI**

Replace the contents of `scripts/export_review_app.py` with:

```python
"""Build YOLO-format patches from review.json under data/10_export/.

Iterates every (model, split) for which a review.json exists; emits
``labels/<stem>.txt`` + ``manifest.json`` + ``pending.json`` +
``provenance.json`` under ``data/10_export/<model>/<split>/``.
Existing exports are overwritten.
"""

import argparse
import hashlib
import logging
import subprocess
from pathlib import Path

import yaml

from data_quality_frame_level.dataset import iter_frames
from data_quality_frame_level.review_app.export import (
    ProvenanceInput,
    export_corrections,
)
from data_quality_frame_level.review_app.persistence import read_review_state

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _audit_git_state(repo_root: Path) -> tuple[str, str, str, bool]:
    """Return (audit_repo, commit, branch, is_dirty) for the repo at repo_root.

    Walks up from repo_root to find the .git dir so the experiment-level
    cwd still finds the parent vision-rd repo.
    """
    git_root = Path(_git(["rev-parse", "--show-toplevel"], cwd=repo_root))
    commit = _git(["rev-parse", "HEAD"], cwd=git_root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=git_root)
    status = _git(["status", "--porcelain"], cwd=git_root)
    is_dirty = bool(status.strip())
    remote = ""
    try:
        remote = _git(["remote", "get-url", "origin"], cwd=git_root)
    except subprocess.CalledProcessError:
        pass
    audit_repo = _audit_repo_from_remote(remote)
    return audit_repo, commit + ("+dirty" if is_dirty else ""), branch, is_dirty


def _audit_repo_from_remote(remote_url: str) -> str:
    """Convert a git remote URL to ``owner/repo`` form, or return raw URL."""
    if not remote_url:
        return ""
    if remote_url.startswith("git@"):
        # git@github.com:owner/repo.git
        _, rest = remote_url.split(":", 1)
        return rest.removesuffix(".git")
    if remote_url.startswith("http"):
        path = remote_url.split("//", 1)[1].split("/", 1)[1]
        return path.removesuffix(".git")
    return remote_url


def _md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root
    params = yaml.safe_load((repo / "params.yaml").read_text())
    models = list(params["models"].keys())
    datasets_root = repo / "data" / "01_raw" / "datasets"
    splits = sorted(p.name for p in datasets_root.iterdir() if p.is_dir())

    audit_repo, audit_commit, audit_branch, _ = _audit_git_state(repo)
    experiment = "experiments/data-quality/frame-level"

    for model in models:
        model_params = params["models"][model]
        thresholds = {
            "conf": float(model_params["conf_thresh"]),
            "iou": float(model_params["iou_thresh"]),
            "review_conf": float(model_params["review_conf_thresh"]),
        }
        for split in splits:
            review_path = repo / "data" / "09_review" / model / split / "review.json"
            if not review_path.is_file():
                log.info("skip: no review.json at %s", review_path)
                continue
            predictions_path = (
                repo / "data" / "07_model_output" / model / split / "predictions.json"
            )
            if not predictions_path.is_file():
                log.warning(
                    "skip: missing predictions at %s", predictions_path
                )
                continue
            state = read_review_state(review_path, model=model, split=split)
            originals = {
                f.stem: f.gt_bboxes for f in iter_frames(datasets_root / split)
            }
            out_dir = repo / "data" / "10_export" / model / split
            prov = ProvenanceInput(
                audit_repo=audit_repo,
                audit_commit=audit_commit,
                audit_branch=audit_branch,
                experiment=experiment,
                thresholds=thresholds,
                predictions_path=str(
                    predictions_path.relative_to(repo)
                ),
                predictions_md5=_md5_of_file(predictions_path),
            )
            manifest = export_corrections(
                review=state,
                originals=originals,
                out_dir=out_dir,
                provenance=prov,
            )
            log.info(
                "%s/%s: %d changed, %d added, %d removed, %d modified",
                model,
                split,
                manifest["totals"]["changed"],
                manifest["totals"]["added"],
                manifest["totals"]["removed"],
                manifest["totals"]["modified"],
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI against the live tree**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run --group review-app python scripts/export_review_app.py 2>&1 | tail -10
```

Expected: emits `data/10_export/yolo11s-nimble-narwhal/<split>/{labels,manifest.json,pending.json,provenance.json}` for each split that has a `review.json`.

- [ ] **Step 3: Inspect a generated provenance.json**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
ls data/10_export/yolo11s-nimble-narwhal/val/
cat data/10_export/yolo11s-nimble-narwhal/val/provenance.json
```

Expected: Four files in the directory, provenance lists `audit_commit` (40-char SHA, possibly with `+dirty`), `audit_branch`, `predictions_md5` (32-char hex), `thresholds` matching `params.yaml`.

- [ ] **Step 4: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/data-quality/frame-level/scripts/export_review_app.py
git commit -m "feat(data-quality/frame-level): CLI emits manifest + pending + provenance"
```

---

## Task 5: `dvc_warning_for_review` helper

**Files:**
- Modify: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/persistence.py`
- Modify: `experiments/data-quality/frame-level/tests/test_review_app_persistence.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_review_app_persistence.py`:

```python
import hashlib

from data_quality_frame_level.review_app.persistence import (
    dvc_warning_for_review,
)


def _md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def test_dvc_warning_returns_none_when_no_dvc_file(tmp_path: Path):
    review = tmp_path / "review.json"
    review.write_text('{"version": 1, "samples": {}}')
    assert dvc_warning_for_review(review) is None


def test_dvc_warning_stale_local(tmp_path: Path):
    review = tmp_path / "review.json"
    review.write_text("local content")
    dvc_file = tmp_path / "review.json.dvc"
    dvc_file.write_text(
        "outs:\n- md5: deadbeefcafe1234\n  size: 16\n  hash: md5\n  path: review.json\n"
    )
    w = dvc_warning_for_review(review)
    assert w is not None
    assert w["kind"] == "stale_local"
    assert w["tracked_md5"] == "deadbeefcafe1234"
    assert w["local_md5"] == _md5(b"local content")


def test_dvc_warning_in_sync(tmp_path: Path):
    review = tmp_path / "review.json"
    content = b"in sync"
    review.write_bytes(content)
    dvc_file = tmp_path / "review.json.dvc"
    dvc_file.write_text(
        f"outs:\n- md5: {_md5(content)}\n  size: {len(content)}\n  hash: md5\n  path: review.json\n"
    )
    assert dvc_warning_for_review(review) is None


def test_dvc_warning_missing_local(tmp_path: Path):
    review = tmp_path / "review.json"
    dvc_file = tmp_path / "review.json.dvc"
    dvc_file.write_text(
        "outs:\n- md5: aaaa\n  size: 0\n  hash: md5\n  path: review.json\n"
    )
    w = dvc_warning_for_review(review)
    assert w is not None
    assert w["kind"] == "missing_local"
```

- [ ] **Step 2: Run, verify it fails**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run pytest tests/test_review_app_persistence.py -v
```

Expected: ImportError on `dvc_warning_for_review`.

- [ ] **Step 3: Implement the helper**

Append to `src/data_quality_frame_level/review_app/persistence.py`:

```python
import hashlib

import yaml


def _md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tracked_md5(dvc_path: Path, target_filename: str) -> str | None:
    """Read the md5 for ``target_filename`` from a single-file ``.dvc``."""
    payload = yaml.safe_load(dvc_path.read_text())
    for out in payload.get("outs", []):
        if out.get("path") == target_filename and out.get("hash", "md5") == "md5":
            return out.get("md5")
    return None


def dvc_warning_for_review(review_path: Path) -> dict | None:
    """Compare local review.json md5 with the .dvc-tracked md5.

    Returns a warning dict if the local file is stale or missing relative
    to the tracked version. Returns None when:

    - There is no sibling ``.dvc`` file (untracked / first session).
    - The local file matches the tracked md5.
    """
    dvc_path = review_path.with_suffix(review_path.suffix + ".dvc")
    if not dvc_path.is_file():
        return None
    tracked = _tracked_md5(dvc_path, review_path.name)
    if tracked is None:
        return None
    if not review_path.is_file():
        return {
            "kind": "missing_local",
            "tracked_md5": tracked,
            "local_md5": None,
            "message": (
                f"DVC tracks {review_path.name} but the local file is missing. "
                "Run `make review-pull` before reviewing."
            ),
        }
    local = _md5_of_file(review_path)
    if local == tracked:
        return None
    return {
        "kind": "stale_local",
        "tracked_md5": tracked,
        "local_md5": local,
        "message": (
            f"Local {review_path.name} differs from the DVC-tracked version. "
            "Run `make review-pull` before reviewing — your saves may "
            "overwrite a peer's work."
        ),
    }
```

- [ ] **Step 4: Run, verify pass**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run pytest tests/test_review_app_persistence.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/persistence.py experiments/data-quality/frame-level/tests/test_review_app_persistence.py
git commit -m "feat(data-quality/frame-level): dvc_warning_for_review helper"
```

---

## Task 6: Wire `dvc_warnings` into `/api/contexts`

**Files:**
- Modify: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/main.py`
- Modify: `experiments/data-quality/frame-level/tests/test_review_app_main.py`

- [ ] **Step 1: Write a failing test**

Append to `tests/test_review_app_main.py`:

```python
def test_get_contexts_includes_dvc_warnings(tmp_path: Path, app_tree):
    client, paths = app_tree
    # Write a sibling .dvc file with an md5 that won't match anything.
    dvc_path = paths.review_path.with_suffix(paths.review_path.suffix + ".dvc")
    dvc_path.parent.mkdir(parents=True, exist_ok=True)
    dvc_path.write_text(
        "outs:\n- md5: ffeeddccbbaa\n  size: 0\n  hash: md5\n  path: review.json\n"
    )
    # Trigger one save so review.json exists with a known content.
    client.post(
        "/api/sample",
        params={"model": "m", "split": "val"},
        json={
            "stem": "s_2024-01-01T00-00-00",
            "status": "reviewed",
            "bboxes": [],
            "reviewer": "arthur",
        },
    )
    body = client.get("/api/contexts").json()
    assert "dvc_warnings" in body
    assert len(body["dvc_warnings"]) == 1
    w = body["dvc_warnings"][0]
    assert w["model"] == "m"
    assert w["split"] == "val"
    assert w["kind"] == "stale_local"
```

- [ ] **Step 2: Run, verify it fails**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run pytest tests/test_review_app_main.py::test_get_contexts_includes_dvc_warnings -v
```

Expected: AssertionError on `"dvc_warnings" in body`.

- [ ] **Step 3: Update the route**

In `src/data_quality_frame_level/review_app/main.py`, change the imports and `/api/contexts` handler:

Add the import near the others:

```python
from data_quality_frame_level.review_app.persistence import dvc_warning_for_review
```

Replace the `get_contexts` handler:

```python
    @app.get("/api/contexts")
    def get_contexts() -> dict:
        warnings: list[dict] = []
        for (m, s), paths in contexts.items():
            w = dvc_warning_for_review(paths.review_path)
            if w is not None:
                warnings.append({**w, "model": m, "split": s})
        return {"models": models, "splits": splits, "dvc_warnings": warnings}
```

- [ ] **Step 4: Run, verify pass + existing tests still pass**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run pytest tests/test_review_app_main.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/main.py experiments/data-quality/frame-level/tests/test_review_app_main.py
git commit -m "feat(data-quality/frame-level): expose dvc_warnings on /api/contexts"
```

---

## Task 7: Frontend banner

**Files:**
- Modify: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/index.html`
- Modify: `experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js`

- [ ] **Step 1: Add the banner container above the header**

In `static/index.html`, just inside `<body>` (before the `<header>` element), add:

```html
  <div id="dvc-banner-host" class="flex flex-col"></div>
```

- [ ] **Step 2: Add Tailwind component styles for the banner**

In `static/index.html`, inside the existing `<style type="text/tailwindcss">` block, append a new component rule (right after the `#help-pane em` rule, before the closing `}` of `@layer components`):

```css
      .dvc-banner { @apply flex items-center gap-3 border-b border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-900; }
      .dvc-banner .msg { @apply flex-1; }
      .dvc-banner .ctx { @apply font-mono text-xs text-amber-700; }
      .dvc-banner button { @apply rounded border border-amber-300 bg-white px-2 py-0.5 text-xs text-amber-700 transition hover:bg-amber-100; }
```

- [ ] **Step 3: Render banners from `/api/contexts` response**

In `static/app.js`, find the `init()` function and the line `const ctxs = await api.contexts();` (around line 91). Right after that line, insert a call to render banners:

```javascript
  renderDvcBanners(ctxs.dvc_warnings || []);
```

Then at module scope (e.g., near `escapeHtml`), add:

```javascript
function renderDvcBanners(warnings) {
  const host = document.getElementById('dvc-banner-host');
  host.innerHTML = '';
  const dismissed = JSON.parse(sessionStorage.getItem('dvc_dismissed') || '[]');
  warnings.forEach(w => {
    const key = `${w.model}/${w.split}/${w.kind}`;
    if (dismissed.includes(key)) return;
    const div = document.createElement('div');
    div.className = 'dvc-banner';
    div.innerHTML = `
      <span class="msg">⚠️ ${escapeHtml(w.message)}</span>
      <span class="ctx">${escapeHtml(w.model)} / ${escapeHtml(w.split)}</span>
      <button type="button">Dismiss</button>`;
    div.querySelector('button').addEventListener('click', () => {
      dismissed.push(key);
      sessionStorage.setItem('dvc_dismissed', JSON.stringify(dismissed));
      div.remove();
    });
    host.appendChild(div);
  });
}
```

- [ ] **Step 4: Smoke-test in browser**

Restart the app:

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
# Force a stale state for testing:
echo "outs:\n- md5: deadbeef\n  size: 0\n  hash: md5\n  path: review.json" > data/09_review/yolo11s-nimble-narwhal/val/review.json.dvc
uv run --group review-app python scripts/run_review_app.py --port 8765 &
disown
sleep 3
```

Open `http://localhost:8765`. Expected: a yellow banner above the header reading "⚠️ Local review.json differs from the DVC-tracked version..." with model/split context and a Dismiss button. Click Dismiss → banner disappears. Reload page → banner stays gone (sessionStorage). New tab → banner returns.

Clean up the fake `.dvc` file after testing:

```bash
rm /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level/data/09_review/yolo11s-nimble-narwhal/val/review.json.dvc
```

Stop the server:

```bash
pkill -f run_review_app.py
```

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/index.html experiments/data-quality/frame-level/src/data_quality_frame_level/review_app/static/app.js
git commit -m "feat(data-quality/frame-level): DVC stale-state banner"
```

---

## Task 8: Full pytest + lint pass

**Files:** none new

- [ ] **Step 1: Run all tests**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run pytest tests/ -v
```

Expected: every test passes — including the existing 61 plus the new ones added in Tasks 1, 2, 3, 5, 6 (≈ 8 new tests, total ≈ 69).

- [ ] **Step 2: Lint + format check**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run ruff check . && uv run ruff format --check .
```

If anything reports issues:

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
uv run ruff check . --fix && uv run ruff format .
```

Then re-run pytest to confirm formatting didn't break anything.

- [ ] **Step 3: End-to-end smoke**

Run the full export pipeline against live data:

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd/experiments/data-quality/frame-level
make review-export
ls data/10_export/yolo11s-nimble-narwhal/val/
```

Expected: `labels/`, `manifest.json`, `pending.json`, `provenance.json` present. Inspect each:

```bash
cat data/10_export/yolo11s-nimble-narwhal/val/manifest.json | python3 -m json.tool | head -30
cat data/10_export/yolo11s-nimble-narwhal/val/pending.json | python3 -m json.tool | head -10
cat data/10_export/yolo11s-nimble-narwhal/val/provenance.json | python3 -m json.tool
```

Verify `manifest.json` has a `contributors` field (sorted, unique reviewer handles), `pending.json` lists only unclear-status stems, `provenance.json` has the full git/threshold/predictions metadata.

- [ ] **Step 4: Final commit if anything was tweaked during smoke**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git status --short
# only commit if status reports modifications
git add -p experiments/data-quality/frame-level/
git commit -m "fix(data-quality/frame-level): smoke-test polish for export flow"
```

If `git status` reports nothing dirty, skip Step 4.

---

## Self-review notes

Spec coverage:

- §4 Output layout (4 files) → Tasks 1, 2, 3 (writers); Task 4 (CLI calling all three).
- §5.1 `manifest.json` w/ `contributors` → Task 1.
- §5.2 `pending.json` → Tasks 1 (impl) + 2 (tests).
- §5.3 `provenance.json` → Tasks 1 (impl) + 3 (tests) + 4 (CLI fills it in).
- §6 Apply contract (informative) → no implementation needed.
- §7 Stale-state safety check → Tasks 5 (helper), 6 (route), 7 (banner).
- §8 Implementation changes → Tasks 1, 4, 5, 6, 7.
- §9 Open questions → none blocking.

Type consistency:

- `ProvenanceInput` defined in Task 1, used identically in Tasks 3, 4.
- `write_manifest_and_labels`, `write_pending`, `write_provenance` signatures used consistently in Tasks 1, 2, 3, 4.
- `dvc_warning_for_review(review_path: Path) -> dict | None` consistent in Tasks 5, 6.
- DVC warning dict shape (`kind`, `tracked_md5`, `local_md5`, `message`) consistent across Task 5 implementation, Task 6 route, Task 7 banner.

No placeholders or unfilled steps.
