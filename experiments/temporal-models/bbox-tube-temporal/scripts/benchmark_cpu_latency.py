"""CPU (and any-device) latency benchmark for the packaged temporal model.

Thin CLI wrapper around :mod:`bbox_tube_temporal.benchmark_latency`. Loads a
packaged model, runs ``predict()`` over the given sequences directory (with
warmup discard), writes a JSON to ``--output``, and prints a summary table.

See ``docs/specs/2026-04-17-cpu-latency-benchmark-design.md``.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from bbox_tube_temporal.benchmark_latency import (
    print_summary,
    run_benchmark_on_model,
)
from bbox_tube_temporal.data import list_sequences
from bbox_tube_temporal.model import BboxTubeTemporalModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zip", type=Path, required=True)
    parser.add_argument("--sequences-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Torch device. Defaults to cpu for benchmark reproducibility.",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Cap on number of sequences (first N in sorted order). None = all.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of leading sequences retained in records but excluded "
        "from summary aggregates.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model = BboxTubeTemporalModel.from_archive(args.model_zip, device=args.device)

    sequences = list_sequences(args.sequences_dir)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]

    # Pass the tqdm iterator directly so the progress bar tracks actual
    # prediction work, not list materialization.
    result = run_benchmark_on_model(
        model,
        tqdm(sequences, desc=f"bench {args.model_zip.name}", unit="seq"),
        warmup=args.warmup,
    )
    # Self-describing JSON: record which model + device produced the numbers.
    result["summary"]["model_zip"] = str(args.model_zip)
    result["summary"]["device"] = args.device

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print_summary(result)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
