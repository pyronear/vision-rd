"""Assemble flattened, temporally-ordered sample sequences for the playground.

Source: the temporal-model-explorer sequence store, where each ``seq_<id>/`` has
a ``meta.json`` (with a ``label`` and an ordered ``frames`` list) and an
``images/`` dir. We copy each chosen sequence's images into a single flat dir,
renamed ``NNN_<original>.jpg`` in meta order so a plain filename sort reproduces
the temporal order the playground CLI relies on.

Usage:
    uv run python scripts/build_sample_sequences.py \
        --store ../temporal-model-explorer/data/03_primary/sequences \
        --out data/01_raw/sample_sequences \
        --per-label 2
"""

import argparse
import json
import shutil
from collections.abc import Iterator
from pathlib import Path


def _iter_meta(store: Path) -> Iterator[tuple[Path, dict]]:
    for meta_path in sorted(store.rglob("meta.json")):
        yield meta_path.parent, json.loads(meta_path.read_text())


def build(store: Path, out: Path, per_label: int) -> None:
    wanted = {"smoke": per_label, "fp": per_label}
    counts = {"smoke": 0, "fp": 0}

    for seq_dir, meta in _iter_meta(store):
        label = meta.get("label")
        if label not in wanted or counts[label] >= wanted[label]:
            continue
        frames = meta.get("frames", [])
        if not frames:
            continue

        dest = out / f"{label}-{seq_dir.name}"
        dest.mkdir(parents=True, exist_ok=True)
        for i, ref in enumerate(frames):
            src = seq_dir / ref["file"]
            if not src.is_file():
                continue
            shutil.copyfile(src, dest / f"{i:03d}_{Path(ref['file']).name}")
        counts[label] += 1
        print(f"wrote {dest} ({len(frames)} frames)")

        if all(counts[k] >= wanted[k] for k in wanted):
            break

    missing = {k: wanted[k] - counts[k] for k in wanted if counts[k] < wanted[k]}
    if missing:
        print(f"WARNING: could not fill all labels, short by {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=2)
    args = parser.parse_args()
    build(args.store, args.out, args.per_label)


if __name__ == "__main__":
    main()
