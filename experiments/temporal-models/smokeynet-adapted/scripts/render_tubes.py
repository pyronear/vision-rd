"""Render one PNG per built smoke tube to a label-nested directory tree.

For each tube JSON in ``--tubes-dir``, look up the source sequence
under ``--raw-dir``, render ``plot_tube_summary``, and save the PNG to
``--output-dir/{smoke,fp}/<sequence_id>.png``.

The output directory is wiped at the start of each run so stale PNGs
from earlier runs (for sequences that have since been filtered out)
don't linger.
"""

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from smokeynet_adapted.data import (
    find_sequence_dir,
    get_sorted_frames,
    load_tube_record,
)
from smokeynet_adapted.tubes import plot_tube_summary, tube_from_record


def _render_one(
    record_path: Path,
    *,
    raw_dir: Path,
    output_dir: Path,
    dpi: int,
) -> tuple[str | None, str | None]:
    """Render a single tube to PNG.

    Returns ``(label, None)`` on success, ``(None, reason)`` on skip.
    """
    record = load_tube_record(record_path)
    sequence_id = record["sequence_id"]
    label = record["label"]
    seq_dir = find_sequence_dir(raw_dir, sequence_id)
    if seq_dir is None:
        return None, f"source dir not found for {sequence_id}"

    frame_paths = get_sorted_frames(seq_dir)
    if not frame_paths:
        return None, f"no frames for {sequence_id}"

    tube = tube_from_record(record)

    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    out_path = label_dir / f"{sequence_id}.png"

    fig = plot_tube_summary(
        frame_paths,
        [tube],
        num_frames=record["num_frames"],
        tube_labels=[label == "smoke"],
        title=f"{sequence_id} [{label}]",
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return label, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tubes-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    split = args.tubes_dir.name

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record_paths = sorted(
        p for p in args.tubes_dir.glob("*.json") if p.name != "_summary.json"
    )

    if not record_paths:
        print(f"[{split}] no tube records found in {args.tubes_dir}; nothing to render")
        return

    by_label: dict[str, int] = {"smoke": 0, "fp": 0}
    skipped: list[str] = []

    for path in record_paths:
        label, reason = _render_one(
            path,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
        if reason is not None:
            skipped.append(reason)
            continue
        assert label is not None
        by_label[label] += 1

    rendered = sum(by_label.values())
    print(
        f"[{split}] rendered {rendered}/{len(record_paths)} tubes "
        f"(smoke={by_label['smoke']}, fp={by_label['fp']}, skipped={len(skipped)})"
    )
    for reason in skipped[:5]:
        print(f"  skipped: {reason}")
    if len(skipped) > 5:
        print(f"  ... and {len(skipped) - 5} more")


if __name__ == "__main__":
    main()
