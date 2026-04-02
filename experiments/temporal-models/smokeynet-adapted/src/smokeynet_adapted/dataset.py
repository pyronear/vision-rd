"""PyTorch Dataset for training SmokeyNetAdapted from precomputed features."""

import json
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .types import Detection, Tube, TubeEntry


class SmokeyNetDataset(Dataset):
    """Dataset of sequences with precomputed RoI features and tube metadata.

    Each sequence is stored as:
    - ``<sequence_id>.pt``: dict with ``roi_features``, ``frame_indices``,
      ``bbox_coords``, ``detection_labels``, ``sequence_label``.
    - ``<sequence_id>.json``: tube metadata for reconstruction.

    Args:
        data_dir: Path to the directory containing ``.pt`` and ``.json`` files.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.sequence_ids = sorted(p.stem for p in self.data_dir.glob("*.pt"))

    def __len__(self) -> int:
        return len(self.sequence_ids)

    def __getitem__(self, idx: int) -> dict[str, Tensor | list[Tube] | str]:
        seq_id = self.sequence_ids[idx]
        pt_path = self.data_dir / f"{seq_id}.pt"
        json_path = self.data_dir / f"{seq_id}.json"

        data = torch.load(pt_path, weights_only=True)
        tubes = _load_tubes_from_json(json_path)

        return {
            "sequence_id": seq_id,
            "roi_features": data["roi_features"],
            "frame_indices": data["frame_indices"],
            "bbox_coords": data["bbox_coords"],
            "detection_labels": data["detection_labels"],
            "sequence_label": data["sequence_label"],
            "tubes": tubes,
        }


def _load_tubes_from_json(json_path: Path) -> list[Tube]:
    """Load tube metadata from a JSON file."""
    with open(json_path) as f:
        tube_dicts = json.load(f)["tubes"]

    tubes = []
    for td in tube_dicts:
        entries = []
        for ed in td["entries"]:
            det = None
            if ed.get("detection") is not None:
                d = ed["detection"]
                det = Detection(
                    class_id=d["class_id"],
                    cx=d["cx"],
                    cy=d["cy"],
                    w=d["w"],
                    h=d["h"],
                    confidence=d["confidence"],
                )
            entries.append(TubeEntry(frame_idx=ed["frame_idx"], detection=det))
        tubes.append(
            Tube(
                tube_id=td["tube_id"],
                entries=entries,
                start_frame=td["start_frame"],
                end_frame=td["end_frame"],
            )
        )
    return tubes
