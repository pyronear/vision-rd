"""Build YOLO-format patches from review.json under data/10_export/.

Iterates every (model, split) for which a review.json exists; emits
``labels/<stem>.txt`` + ``manifest.json`` under
``data/10_export/<model>/<split>/``. Existing exports are overwritten.
"""

import argparse
import logging
from pathlib import Path

import yaml

from data_quality_frame_level.dataset import iter_frames
from data_quality_frame_level.review_app.export import export_corrections
from data_quality_frame_level.review_app.persistence import read_review_state

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root
    params = yaml.safe_load((repo / "params.yaml").read_text())
    models = list(params["models"].keys())
    datasets_root = repo / "data" / "01_raw" / "datasets"
    splits = sorted(p.name for p in datasets_root.iterdir() if p.is_dir())
    for model in models:
        for split in splits:
            review_path = repo / "data" / "09_review" / model / split / "review.json"
            if not review_path.is_file():
                log.info("skip: no review.json at %s", review_path)
                continue
            state = read_review_state(review_path, model=model, split=split)
            originals = {
                f.stem: f.gt_bboxes for f in iter_frames(datasets_root / split)
            }
            out_dir = repo / "data" / "10_export" / model / split
            manifest = export_corrections(
                review=state, originals=originals, out_dir=out_dir
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
