"""CLI: import the annotated zip into the sequence store."""

import argparse
from pathlib import Path

from temporal_model_explorer.import_local_zip import import_zip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument(
        "--out", type=Path, default=Path("data/03_primary/sequences/local_zip")
    )
    args = ap.parse_args()
    n = import_zip(args.zip, args.out)
    print(f"imported {n} sequences into {args.out}")


if __name__ == "__main__":
    main()
