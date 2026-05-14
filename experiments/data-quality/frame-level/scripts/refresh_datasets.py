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
from datetime import UTC, datetime
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
            src = pyro_root / "data" / "processed" / upstream_dir / modality / split
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
    run(
        [
            "dvc",
            "pull",
            "data/processed/yolo_train_val",
            "data/processed/yolo_test",
        ],
        cwd=clone_dir,
    )
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


def precheck_no_tracked_files(experiment_root: Path) -> None:
    """Abort early if any target split has git-tracked files.

    ``dvc add`` refuses to track paths git already tracks. Detect this
    before the slow clone+pull so the operator gets a clear instruction
    immediately rather than after several minutes of wasted work.
    """
    targets = [f"data/01_raw/datasets/{s}" for s in SPLITS]
    result = subprocess.run(
        ["git", "ls-files", "--", *targets],
        cwd=experiment_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    if tracked:
        lines = "\n".join(f"  {t}" for t in tracked)
        raise SystemExit(
            "Cannot refresh: the following files are git-tracked inside the\n"
            "dataset directories. Run `git rm` to untrack them before retrying:\n"
            f"{lines}"
        )


def dvc_add_and_push(experiment_root: Path) -> None:
    """``dvc add`` each split dir and push to the configured remote."""
    targets = [f"data/01_raw/datasets/{s}" for s in SPLITS]
    run(["uv", "run", "dvc", "add", *targets], cwd=experiment_root)
    run(
        ["uv", "run", "dvc", "push", *[f"{t}.dvc" for t in targets]],
        cwd=experiment_root,
    )


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
        "refreshed_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
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

    precheck_no_tracked_files(experiment_root)

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
