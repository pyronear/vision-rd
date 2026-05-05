"""Build YOLO-format patches from review.json under data/10_export/.

Iterates every (model, split) for which a review.json exists; emits
``labels/<stem>.txt`` + ``manifest.json`` + ``pending.json`` +
``provenance.json`` under ``data/10_export/<model>/<split>/``.
Existing exports are overwritten.
"""

import argparse
import contextlib
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


def _audit_repo_from_remote(remote_url: str) -> str:
    """Convert a git remote URL to ``owner/repo`` form, or return raw URL."""
    if not remote_url:
        return ""
    if remote_url.startswith("git@"):
        _, rest = remote_url.split(":", 1)
        return rest.removesuffix(".git")
    if remote_url.startswith("http"):
        path = remote_url.split("//", 1)[1].split("/", 1)[1]
        return path.removesuffix(".git")
    return remote_url


def _audit_git_state(repo_root: Path) -> tuple[str, str, str]:
    """Return (audit_repo, commit_with_dirty_marker, branch)."""
    git_root = Path(_git(["rev-parse", "--show-toplevel"], cwd=repo_root))
    commit = _git(["rev-parse", "HEAD"], cwd=git_root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=git_root)
    status = _git(["status", "--porcelain"], cwd=git_root)
    is_dirty = bool(status.strip())
    remote = ""
    with contextlib.suppress(subprocess.CalledProcessError):
        remote = _git(["remote", "get-url", "origin"], cwd=git_root)
    audit_repo = _audit_repo_from_remote(remote)
    return audit_repo, commit + ("+dirty" if is_dirty else ""), branch


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

    audit_repo, audit_commit, audit_branch = _audit_git_state(repo)
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
                log.warning("skip: missing predictions at %s", predictions_path)
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
                predictions_path=str(predictions_path.relative_to(repo)),
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
