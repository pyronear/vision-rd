"""CLI: run the model's bundled YOLO over the working set, cache detections.

Loads a model package to reuse its EXACT YOLO weights + detection params, so
cached detections match BboxTubeTemporalModel.predict. Also writes the lab
pipeline config so the app stays model-free.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from bbox_tube_temporal.inference import run_yolo_on_frames
from bbox_tube_temporal.package import load_model_package

from tube_builder_lab.cache import cache_one
from tube_builder_lab.pipeline import extract_pipeline_config, write_pipeline_config
from tube_builder_lab.store import build_frames, read_meta, seq_dir_for_key
from tube_builder_lab.working_set import load_working_set

logging.basicConfig(level=logging.INFO)


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    store_dir = Path(params["store"])
    out_dir = Path(params["detections_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    model_zip = Path(params["models_dir"]) / params["model_name"] / "model.zip"

    pkg = load_model_package(model_zip)
    cfg = extract_pipeline_config(pkg.config)
    write_pipeline_config(Path(params["pipeline_config"]), cfg)

    def run_yolo(frames):
        return run_yolo_on_frames(
            pkg.yolo_model,
            frames,
            confidence_threshold=cfg.confidence_threshold,
            iou_nms=cfg.iou_nms,
            image_size=cfg.image_size,
        )

    ws = load_working_set(Path("working_set.yaml"))
    for item in ws.all():
        seq_dir = seq_dir_for_key(store_dir, item.key)
        if seq_dir is None:
            logging.warning(
                "no sequence dir for %s; run import_sequences first", item.key
            )
            continue
        frames = build_frames(seq_dir, read_meta(seq_dir))
        cache_one(out_dir=out_dir, key=item.key, frames=frames, run_yolo=run_yolo)


if __name__ == "__main__":
    main()
