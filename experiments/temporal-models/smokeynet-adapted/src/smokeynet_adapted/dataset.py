"""PyTorch Dataset for the basic temporal smoke classifier.

Reads cropped PNG patches produced by ``scripts/build_model_input.py``
and returns per-tube tensors padded to a fixed length with a mask.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class TubePatchDataset(Dataset):
    """Dataset of cropped tube patches stored as PNG folders.

    Each item:

    .. code-block:: python

        {
            "patches": Tensor[max_frames, 3, 224, 224],  # float32, ImageNet-normalized
            "mask":    Tensor[max_frames] bool,           # True = real frame
            "label":   Tensor[] float32,                  # 0.0 fp, 1.0 smoke
            "sequence_id": str,
        }

    Args:
        split_dir: Directory containing ``_index.json`` and one
            sub-directory per tube.
        max_frames: Pad/truncate length.
    """

    def __init__(self, split_dir: Path, max_frames: int) -> None:
        self.split_dir = Path(split_dir)
        self.max_frames = max_frames
        index = json.loads((self.split_dir / "_index.json").read_text())
        self.index: list[dict] = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        record = self.index[idx]
        seq_id: str = record["sequence_id"]
        label_int: int = record["label_int"]
        seq_dir = self.split_dir / seq_id

        meta = json.loads((seq_dir / "meta.json").read_text())
        frame_files = [seq_dir / f["filename"] for f in meta["frames"]]
        n = min(len(frame_files), self.max_frames)

        patches = torch.zeros(self.max_frames, 3, 224, 224, dtype=torch.float32)
        mask = torch.zeros(self.max_frames, dtype=torch.bool)
        for i in range(n):
            img = Image.open(frame_files[i]).convert("RGB")
            tensor = to_tensor(img)  # CHW float32 in [0, 1]
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
            patches[i] = tensor
            mask[i] = True

        return {
            "patches": patches,
            "mask": mask,
            "label": torch.tensor(float(label_int), dtype=torch.float32),
            "sequence_id": seq_id,
        }
