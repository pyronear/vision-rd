"""Command-line entry point for the temporal model playground.

Usage:
    uv run playground run --model bbox-tube-vit-dinov2 path/to/sequence_dir/
    uv run playground run --model-package model.zip f1.jpg f2.jpg --json
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from bbox_tube_temporal.model import BboxTubeTemporalModel

from .core import format_summary, resolve_frames, resolve_model_package

DEFAULT_MODELS_DIR = Path("data/01_raw/models")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="playground", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a model on a directory or list of frames.")
    group = run.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model", help="Model name under data/01_raw/models/<name>/model.zip"
    )
    group.add_argument("--model-package", help="Path to an explicit model .zip")
    run.add_argument(
        "--device", default=None, help="torch device (default: auto cuda/mps/cpu)"
    )
    run.add_argument(
        "--json", action="store_true", help="Emit full TemporalModelOutput as JSON"
    )
    run.add_argument(
        "inputs",
        nargs="+",
        help="A directory of images, or image file paths in order",
    )
    run.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory of named model packages (default: {DEFAULT_MODELS_DIR})",
    )
    return parser


def _run(args: argparse.Namespace) -> None:
    frame_paths = resolve_frames(args.inputs)
    package = resolve_model_package(
        model=args.model,
        model_package=Path(args.model_package) if args.model_package else None,
        models_dir=args.models_dir,
    )

    model = BboxTubeTemporalModel.from_package(package, device=args.device)
    start = time.perf_counter()
    out = model.predict_sequence(frame_paths)
    runtime_ms = (time.perf_counter() - start) * 1000.0

    if args.json:
        print(json.dumps(dataclasses.asdict(out), indent=2, default=str))
    else:
        print(format_summary(out, frame_paths, runtime_ms))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        _run(args)


if __name__ == "__main__":
    main()
