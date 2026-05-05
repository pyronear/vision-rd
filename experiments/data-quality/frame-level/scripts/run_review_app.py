"""Launch the frame-level review app via uvicorn.

Discovers contexts from ``params.yaml`` (models) and the
``data/01_raw/datasets/`` tree (splits). Lazy-loads each context on
first request.

Usage::

    uv run --group review-app python scripts/run_review_app.py
"""

import argparse
from pathlib import Path

import uvicorn
import yaml

from data_quality_frame_level.review_app.main import create_app
from data_quality_frame_level.review_app.state import Paths


def _discover_paths(
    repo_root: Path,
) -> tuple[dict[tuple[str, str], Paths], list[str], list[str]]:
    params = yaml.safe_load((repo_root / "params.yaml").read_text())
    models = list(params["models"].keys())
    datasets_root = repo_root / "data" / "01_raw" / "datasets"
    splits = sorted(p.name for p in datasets_root.iterdir() if p.is_dir())
    contexts: dict[tuple[str, str], Paths] = {}
    for model in models:
        for split in splits:
            split_dir = datasets_root / split
            pred_path = (
                repo_root
                / "data"
                / "07_model_output"
                / model
                / split
                / "predictions.json"
            )
            review_path = (
                repo_root / "data" / "09_review" / model / split / "review.json"
            )
            if pred_path.is_file() and split_dir.is_dir():
                contexts[(model, split)] = Paths(
                    split_dir=split_dir,
                    predictions_path=pred_path,
                    review_path=review_path,
                )
    return contexts, models, splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    contexts, models, splits = _discover_paths(args.repo_root)
    app = create_app(contexts=contexts, models=models, splits=splits)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
