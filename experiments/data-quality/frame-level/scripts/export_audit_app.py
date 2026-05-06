"""Build YOLO-format patches from review.json under data/10_export/.

Iterates every (model, split) for which a review.json exists; emits
``labels/<stem>.txt`` + ``manifest.json`` + ``pending.json`` +
``provenance.json`` under ``data/10_export/<model>/<split>/``.
Existing exports are overwritten.
"""

import argparse
import logging
from pathlib import Path

import yaml

from data_quality_frame_level.audit_app.export_runner import export_one

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.05)
    parser.add_argument("--review-conf", type=float, default=0.35)
    args = parser.parse_args()
    repo = args.repo_root
    params = yaml.safe_load((repo / "params.yaml").read_text())
    models = list(params["models"].keys())
    datasets_root = repo / "data" / "01_raw" / "datasets"
    splits = sorted(p.name for p in datasets_root.iterdir() if p.is_dir())

    for model in models:
        for split in splits:
            manifest = export_one(
                repo_root=repo,
                model=model,
                split=split,
                conf=args.conf,
                iou=args.iou,
                review_conf=args.review_conf,
            )
            if manifest is None:
                log.info("skip: no review.json or predictions for %s/%s", model, split)
                continue
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
