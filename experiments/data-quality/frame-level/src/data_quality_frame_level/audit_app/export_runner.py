"""Per-(model, split) export orchestration shared by CLI and API.

Reads review.json + predictions.json + GT for a single ``(model, split)``,
gathers git/params metadata, and writes the four export files. Both
``scripts/export_audit_app.py`` (the CLI) and the FastAPI ``/api/export``
route call :func:`export_one`.
"""

import contextlib
import hashlib
import subprocess
from pathlib import Path

from data_quality_frame_level.audit_app.export import (
    ProvenanceInput,
    export_corrections,
)
from data_quality_frame_level.audit_app.persistence import read_review_state
from data_quality_frame_level.dataset import iter_frames

EXPERIMENT_DIR = "experiments/data-quality/frame-level"


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _audit_repo_from_remote(remote_url: str) -> str:
    if not remote_url:
        return ""
    if remote_url.startswith("git@"):
        _, rest = remote_url.split(":", 1)
        return rest.removesuffix(".git")
    if remote_url.startswith("http"):
        path = remote_url.split("//", 1)[1].split("/", 1)[1]
        return path.removesuffix(".git")
    return remote_url


def audit_git_state(repo_root: Path) -> tuple[str, str, str]:
    """Return (audit_repo, commit_with_dirty_marker, branch).

    Returns empty strings when ``repo_root`` isn't inside a git repo —
    keeps the export flow usable in test fixtures and from non-git
    directories.
    """
    try:
        git_root = Path(_git(["rev-parse", "--show-toplevel"], cwd=repo_root))
    except subprocess.CalledProcessError:
        return "", "", ""
    commit = _git(["rev-parse", "HEAD"], cwd=git_root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=git_root)
    status = _git(["status", "--porcelain"], cwd=git_root)
    is_dirty = bool(status.strip())
    remote = ""
    with contextlib.suppress(subprocess.CalledProcessError):
        remote = _git(["remote", "get-url", "origin"], cwd=git_root)
    return (
        _audit_repo_from_remote(remote),
        commit + ("+dirty" if is_dirty else ""),
        branch,
    )


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_one(
    *,
    repo_root: Path,
    model: str,
    split: str,
    conf: float,
    iou: float,
    review_conf: float,
) -> dict | None:
    """Export one (model, split) to ``data/10_export/<model>/<split>/``.

    ``conf``/``iou``/``review_conf`` are the review-time thresholds the
    reviewer was using; they're recorded in provenance, not used to filter.

    Returns the manifest payload, or ``None`` if there's no review.json
    or no predictions.json for the context.
    """
    review_path = repo_root / "data" / "09_review" / model / split / "review.json"
    if not review_path.is_file():
        return None
    predictions_path = (
        repo_root / "data" / "07_model_output" / model / split / "predictions.json"
    )
    if not predictions_path.is_file():
        return None
    split_dir = repo_root / "data" / "01_raw" / "datasets" / split
    thresholds = {"conf": conf, "iou": iou, "review_conf": review_conf}
    audit_repo, audit_commit, audit_branch = audit_git_state(repo_root)
    state = read_review_state(review_path, model=model, split=split)
    originals = {f.stem: f.gt_bboxes for f in iter_frames(split_dir)}
    out_dir = repo_root / "data" / "10_export" / model / split
    prov = ProvenanceInput(
        audit_repo=audit_repo,
        audit_commit=audit_commit,
        audit_branch=audit_branch,
        experiment=EXPERIMENT_DIR,
        thresholds=thresholds,
        predictions_path=str(predictions_path.relative_to(repo_root)),
        predictions_md5=md5_of_file(predictions_path),
    )
    return export_corrections(
        review=state,
        originals=originals,
        out_dir=out_dir,
        provenance=prov,
    )
