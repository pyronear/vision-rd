"""CLI: run the models in the DVC-tracked models dir over the sequence store."""

import argparse
from pathlib import Path

import torch

from temporal_model_explorer.run_models import load_models, run_over_store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("data/03_primary/sequences"))
    ap.add_argument("--models-dir", type=Path, default=Path("data/06_models"))
    ap.add_argument("--out", type=Path, default=Path("data/07_model_output"))
    ap.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    models = load_models(args.models_dir, device=device)
    if not models:
        raise SystemExit(
            f"No models under {args.models_dir} (run prepare_models first)."
        )
    df = run_over_store(
        args.store, models, args.out / "results.parquet", args.out / "details"
    )
    n_seq = df["key"].nunique()
    out_file = args.out / "results.parquet"
    print(
        f"ran {len(models)} model(s) on {device} over {n_seq} sequences -> {out_file}"
    )


if __name__ == "__main__":
    main()
