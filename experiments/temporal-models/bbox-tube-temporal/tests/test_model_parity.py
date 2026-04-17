"""Parity test: offline training path logits == predict() logits on same input.

Methodology:
- Build tubes offline from GT labels via load_frame_detections + build_tubes
  + select_longest_tube + interpolate_gaps.
- Crop patches via the exact same primitives as TubePatchDataset
  (expand_bbox / norm_bbox_to_pixel_square / crop_and_resize / to_tensor /
  ImageNet normalization), batched in the same shape.
- Forward through the classifier -> reference logit.
- Run BboxTubeTemporalModel.predict() with a fake YOLO that returns the
  same GT detections per frame.
- Assert predict()'s winning-tube logit == reference logit to 1e-5.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from pyrocore.types import Frame
from torchvision.transforms.functional import to_tensor

from bbox_tube_temporal.data import load_frame_detections
from bbox_tube_temporal.model import BboxTubeTemporalModel
from bbox_tube_temporal.model_input import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
from bbox_tube_temporal.temporal_classifier import TemporalSmokeClassifier
from bbox_tube_temporal.tubes import (
    build_tubes,
    interpolate_gaps,
    select_longest_tube,
)

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "wildfire" / "seq_synth01"


CFG: dict = {
    "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 128},
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "min_tube_length": 2,
        "infer_min_tube_length": 2,
        "min_detected_entries": 2,
        "interpolate_gaps": True,
    },
    "model_input": {
        "context_factor": 1.5,
        "patch_size": 32,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "resnet18",
        "arch": "gru",
        "hidden_dim": 32,
        "num_layers": 1,
        "bidirectional": False,
        "max_frames": 5,
        "pretrained": False,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


def _offline_logit(classifier: TemporalSmokeClassifier) -> float:
    """Run the offline training path and return the classifier logit."""
    fdets = load_frame_detections(FIXTURE)
    tubes = build_tubes(fdets, iou_threshold=0.2, max_misses=2)
    tube = select_longest_tube(tubes)
    assert tube is not None
    interpolate_gaps(tube)

    mi = CFG["model_input"]
    t_max = CFG["classifier"]["max_frames"]
    patches = torch.zeros(t_max, 3, mi["patch_size"], mi["patch_size"])
    mask = torch.zeros(t_max, dtype=torch.bool)
    frame_paths = sorted((FIXTURE / "images").glob("*.jpg"))
    mean = torch.tensor(mi["normalization"]["mean"]).view(3, 1, 1)
    std = torch.tensor(mi["normalization"]["std"]).view(3, 1, 1)
    for slot, entry in enumerate(tube.entries[:t_max]):
        det = entry.detection
        assert det is not None
        img = np.array(Image.open(frame_paths[entry.frame_idx]).convert("RGB"))
        h_img, w_img, _ = img.shape
        cx, cy, w, h = expand_bbox(det.cx, det.cy, det.w, det.h, mi["context_factor"])
        box = norm_bbox_to_pixel_square(cx, cy, w, h, w_img, h_img)
        p = crop_and_resize(img, box, mi["patch_size"])
        pt = to_tensor(Image.fromarray(p))
        patches[slot] = (pt - mean) / std
        mask[slot] = True

    with torch.no_grad():
        logit = classifier(patches.unsqueeze(0), mask.unsqueeze(0))
    return float(logit.item())


def _fake_yolo_from_gt(fixture: Path) -> MagicMock:
    """Build a fake YOLO that returns GT detections per frame."""
    fdets = load_frame_detections(fixture)
    by_path = {}
    for fd in fdets:
        img_path = fixture / "images" / f"{fd.frame_id}.jpg"
        by_path[str(img_path)] = fd.detections

    def predict(paths, **_kwargs):
        results = []
        for p in paths:
            dets = by_path[p]
            r = MagicMock()
            if not dets:
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self: 0
                r.boxes.xywhn = torch.zeros(0, 4)
                r.boxes.conf = torch.zeros(0)
                r.boxes.cls = torch.zeros(0)
            else:
                n_dets = len(dets)
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self, n=n_dets: n
                r.boxes.xywhn = torch.tensor([[d.cx, d.cy, d.w, d.h] for d in dets])
                r.boxes.conf = torch.tensor([d.confidence for d in dets])
                r.boxes.cls = torch.tensor([d.class_id for d in dets]).float()
            results.append(r)
        return results

    m = MagicMock()
    m.predict.side_effect = predict
    return m


@pytest.fixture(scope="module")
def classifier() -> TemporalSmokeClassifier:
    torch.manual_seed(0)
    model = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        pretrained=False,
    )
    model.eval()
    return model


def test_parity_logit_matches(classifier: TemporalSmokeClassifier) -> None:
    offline = _offline_logit(classifier)

    frames = [
        Frame(frame_id=p.stem, image_path=p, timestamp=None)
        for p in sorted((FIXTURE / "images").glob("*.jpg"))
    ]
    yolo = _fake_yolo_from_gt(FIXTURE)
    # Pin device to CPU so offline and online paths share numerics on any host.
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=classifier, config=CFG, device="cpu"
    )
    out = model.predict(frames=frames)

    kept = out.details["tubes"]["kept"]
    assert len(kept) >= 1
    online = max(t["logit"] for t in kept)

    assert online == pytest.approx(offline, abs=1e-5)


CFG_TRANSFORMER: dict = {
    "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 224},
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "min_tube_length": 2,
        "infer_min_tube_length": 2,
        "min_detected_entries": 2,
        "interpolate_gaps": True,
    },
    "model_input": {
        "context_factor": 1.5,
        "patch_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "vit_small_patch16_224",
        "arch": "transformer",
        "hidden_dim": 32,
        "max_frames": 5,
        "pretrained": False,
        "global_pool": "token",
        "transformer_num_layers": 1,
        "transformer_num_heads": 2,
        "transformer_ffn_dim": 64,
        "transformer_dropout": 0.0,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


@pytest.fixture(scope="module")
def transformer_classifier() -> TemporalSmokeClassifier:
    torch.manual_seed(0)
    model = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        arch="transformer",
        hidden_dim=32,
        pretrained=False,
        global_pool="token",
        transformer_num_layers=1,
        transformer_num_heads=2,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
        max_frames=5,
    )
    model.eval()
    return model


def _offline_logit_with_cfg(classifier: TemporalSmokeClassifier, cfg: dict) -> float:
    """Variant of _offline_logit that reads patch_size/normalization from cfg."""
    fdets = load_frame_detections(FIXTURE)
    tubes = build_tubes(fdets, iou_threshold=0.2, max_misses=2)
    tube = select_longest_tube(tubes)
    assert tube is not None
    interpolate_gaps(tube)

    mi = cfg["model_input"]
    t_max = cfg["classifier"]["max_frames"]
    patches = torch.zeros(t_max, 3, mi["patch_size"], mi["patch_size"])
    mask = torch.zeros(t_max, dtype=torch.bool)
    frame_paths = sorted((FIXTURE / "images").glob("*.jpg"))
    mean = torch.tensor(mi["normalization"]["mean"]).view(3, 1, 1)
    std = torch.tensor(mi["normalization"]["std"]).view(3, 1, 1)
    for slot, entry in enumerate(tube.entries[:t_max]):
        det = entry.detection
        assert det is not None
        img = np.array(Image.open(frame_paths[entry.frame_idx]).convert("RGB"))
        h_img, w_img, _ = img.shape
        cx, cy, w, h = expand_bbox(det.cx, det.cy, det.w, det.h, mi["context_factor"])
        box = norm_bbox_to_pixel_square(cx, cy, w, h, w_img, h_img)
        p = crop_and_resize(img, box, mi["patch_size"])
        pt = to_tensor(Image.fromarray(p))
        patches[slot] = (pt - mean) / std
        mask[slot] = True

    with torch.no_grad():
        logit = classifier(patches.unsqueeze(0), mask.unsqueeze(0))
    return float(logit.item())


def test_parity_logit_matches_transformer(
    transformer_classifier: TemporalSmokeClassifier,
) -> None:
    offline = _offline_logit_with_cfg(transformer_classifier, CFG_TRANSFORMER)

    frames = [
        Frame(frame_id=p.stem, image_path=p, timestamp=None)
        for p in sorted((FIXTURE / "images").glob("*.jpg"))
    ]
    yolo = _fake_yolo_from_gt(FIXTURE)
    # Pin device to CPU so offline (CPU-only) and online paths share numerics;
    # ViT has enough float32 drift between CPU/GPU to break 1e-5 parity.
    model = BboxTubeTemporalModel(
        yolo_model=yolo,
        classifier=transformer_classifier,
        config=CFG_TRANSFORMER,
        device="cpu",
    )
    out = model.predict(frames=frames)

    kept = out.details["tubes"]["kept"]
    assert len(kept) >= 1
    online = max(t["logit"] for t in kept)

    assert online == pytest.approx(offline, abs=1e-5)
