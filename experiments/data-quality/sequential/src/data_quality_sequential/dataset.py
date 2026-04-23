"""Sequence discovery + folder-based ground truth.

Scans a pyro-dataset sequential split (``<split>/{wildfire,fp}/<seq>/images/*.jpg``)
and emits :class:`SequenceRef` records with ground truth inferred from the parent
directory (``wildfire/`` = positive, ``fp/`` = negative).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SequenceRef:
    """One sequence ready to be fed to a :class:`pyrocore.TemporalModel`.

    Attributes:
        name: Sequence directory name (unique within a split).
        split: ``"train"`` | ``"val"`` | ``"test"``.
        ground_truth: ``True`` iff the sequence lives under ``wildfire/``.
        frame_paths: Frame image paths, sorted by filename.
    """

    name: str
    split: str
    ground_truth: bool
    frame_paths: list[Path]


def _collect(group_dir: Path, split: str, ground_truth: bool) -> list[SequenceRef]:
    if not group_dir.is_dir():
        return []
    refs: list[SequenceRef] = []
    for seq_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        images_dir = seq_dir / "images"
        if not images_dir.is_dir():
            continue
        frames = sorted(images_dir.glob("*.jpg"))
        if not frames:
            continue
        refs.append(
            SequenceRef(
                name=seq_dir.name,
                split=split,
                ground_truth=ground_truth,
                frame_paths=frames,
            )
        )
    return refs


def iter_sequences(split_dir: Path, split: str) -> Iterator[SequenceRef]:
    """Yield every sequence in ``split_dir`` with its ground-truth label.

    Sequences with no ``images/`` directory or no ``*.jpg`` files are skipped.

    Args:
        split_dir: Root of a single split (e.g., ``data/01_raw/datasets/train``).
        split: Label attached to each emitted :class:`SequenceRef` (typically
            the directory name).

    Yields:
        :class:`SequenceRef` for each non-empty sequence under ``wildfire/`` and
        then ``fp/``.
    """
    yield from _collect(split_dir / "wildfire", split=split, ground_truth=True)
    yield from _collect(split_dir / "fp", split=split, ground_truth=False)
