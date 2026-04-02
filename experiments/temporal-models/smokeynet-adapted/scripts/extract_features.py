"""Extract RoI features and build tubes for all sequences in a split.

Runs the YOLO model on each sequence, extracts RoI-aligned backbone
features, builds smoke tubes, and saves one ``.pt`` + ``.json`` pair
per sequence for training.

Usage:
    uv run python scripts/extract_features.py \
        --data-dir data/01_raw/datasets/train \
        --model-path data/01_raw/models/yolo11s_nimble-narwhal_v6.0.0.pt \
        --output-dir data/03_primary/train \
        --confidence-threshold 0.01 \
        --iou-nms 0.2 \
        --image-size 1024 \
        --roi-size 7 \
        --context-factor 1.2 \
        --max-detections-per-frame 10 \
        --iou-threshold 0.2 \
        --max-misses 2
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from smokeynet_adapted.backbone import YoloRoiExtractor
from smokeynet_adapted.data import (
    get_sorted_frames,
    is_wf_sequence,
    list_sequences,
    parse_timestamp,
)
from smokeynet_adapted.detector import load_model, run_yolo_on_frame
from smokeynet_adapted.tubes import build_tubes
from smokeynet_adapted.types import FrameDetections

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract RoI features and build tubes."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, required=True)
    parser.add_argument("--iou-nms", type=float, required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--roi-size", type=int, default=7)
    parser.add_argument("--context-factor", type=float, default=1.2)
    parser.add_argument("--max-detections-per-frame", type=int, default=10)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--max-misses", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading YOLO model from %s", args.model_path)
    yolo_model = load_model(args.model_path)

    logger.info("Building RoI extractor (d_model=%d)", args.d_model)
    roi_extractor = YoloRoiExtractor(
        yolo_model=yolo_model,
        d_model=args.d_model,
        roi_size=args.roi_size,
        context_factor=args.context_factor,
    )
    roi_extractor.to(device)
    roi_extractor.eval()

    sequences = list_sequences(args.data_dir)
    logger.info("Found %d sequences in %s", len(sequences), args.data_dir)

    for seq_dir in tqdm(sequences, desc="Extracting features"):
        seq_id = seq_dir.name
        pt_path = args.output_dir / f"{seq_id}.pt"
        json_path = args.output_dir / f"{seq_id}.json"

        if pt_path.exists() and json_path.exists():
            continue

        frame_paths = get_sorted_frames(seq_dir)
        if not frame_paths:
            continue

        gt = is_wf_sequence(seq_dir)

        # Run YOLO on each frame
        frame_dets: list[FrameDetections] = []
        for idx, fpath in enumerate(frame_paths):
            dets = run_yolo_on_frame(
                yolo_model,
                fpath,
                conf=args.confidence_threshold,
                iou_nms=args.iou_nms,
                img_size=args.image_size,
            )
            if len(dets) > args.max_detections_per_frame:
                dets.sort(key=lambda d: d.confidence, reverse=True)
                dets = dets[: args.max_detections_per_frame]
            frame_dets.append(
                FrameDetections(
                    frame_idx=idx,
                    frame_id=fpath.stem,
                    timestamp=parse_timestamp(fpath.stem),
                    detections=dets,
                )
            )

        # Build tubes
        tubes = build_tubes(
            frame_dets,
            iou_threshold=args.iou_threshold,
            max_misses=args.max_misses,
        )

        # Extract RoI features per frame
        all_features = []
        all_frame_indices = []
        all_bbox_coords = []
        all_det_labels = []

        for fd in frame_dets:
            if not fd.detections:
                continue

            img = cv2.imread(str(frame_paths[fd.frame_idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img = img.shape[:2]

            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            bboxes = torch.tensor([[d.cx, d.cy, d.w, d.h] for d in fd.detections])

            with torch.no_grad():
                feats = roi_extractor(img_t, bboxes, image_size=(h_img, w_img))

            all_features.append(feats.cpu())
            all_frame_indices.extend([fd.frame_idx] * len(fd.detections))
            all_bbox_coords.append(bboxes)

            # Label: all detections labelled 1.0 for WF, 0.0 for FP
            label = 1.0 if gt else 0.0
            all_det_labels.extend([label] * len(fd.detections))

        # Save .pt
        if all_features:
            roi_features = torch.cat(all_features, dim=0)
            bbox_coords = torch.cat(all_bbox_coords, dim=0)
        else:
            roi_features = torch.zeros(0, args.d_model)
            bbox_coords = torch.zeros(0, 4)

        pt_data = {
            "roi_features": roi_features,
            "frame_indices": torch.tensor(all_frame_indices, dtype=torch.long),
            "bbox_coords": bbox_coords,
            "detection_labels": torch.tensor(all_det_labels),
            "sequence_label": torch.tensor(1.0 if gt else 0.0),
        }
        torch.save(pt_data, pt_path)

        # Save .json tube metadata
        tube_dicts = []
        for tube in tubes:
            entries = []
            for entry in tube.entries:
                ed = {"frame_idx": entry.frame_idx, "detection": None}
                if entry.detection is not None:
                    ed["detection"] = dataclasses.asdict(entry.detection)
                entries.append(ed)
            tube_dicts.append(
                {
                    "tube_id": tube.tube_id,
                    "start_frame": tube.start_frame,
                    "end_frame": tube.end_frame,
                    "entries": entries,
                }
            )

        metadata = {
            "sequence_id": seq_id,
            "is_positive": gt,
            "num_frames": len(frame_paths),
            "num_detections": int(roi_features.shape[0]),
            "num_tubes": len(tubes),
            "tubes": tube_dicts,
        }
        json_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Feature extraction complete. Output in %s", args.output_dir)


if __name__ == "__main__":
    main()
