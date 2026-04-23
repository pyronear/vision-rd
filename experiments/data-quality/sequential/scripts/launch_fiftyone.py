"""Launch the FiftyOne app to browse the data-quality FP/FN review datasets.

Opens the FiftyOne web UI on port 5151 with one of the persisted
``dq-seq_*`` datasets loaded. Switch between FP and FN sets, and across
splits, via the dataset selector in the FiftyOne sidebar.

Usage::

    uv run --group explore python scripts/launch_fiftyone.py
"""

import signal
import sys

import fiftyone as fo


def main() -> None:
    datasets = fo.list_datasets(glob_patt="dq-seq_*")
    if not datasets:
        raise SystemExit(
            "No dq-seq_* datasets found. "
            "Run 'uv run dvc repro' (or the build_fiftyone stages) first."
        )

    # Default to the first FP dataset when available — those are usually
    # the shortest and highest-signal to start review with.
    dataset_name = next(
        (d for d in sorted(datasets) if d.endswith("_fp")),
        datasets[0],
    )
    dataset = fo.load_dataset(dataset_name)
    print(f"Launching FiftyOne with dataset '{dataset_name}' ({len(dataset)} samples)")
    print(f"Available datasets: {', '.join(sorted(datasets))}")
    print("Switch datasets in the FiftyOne UI via the dataset selector dropdown.")

    session = fo.launch_app(dataset, port=5151)

    def _shutdown(signum, _frame):
        print("\nShutting down FiftyOne...")
        session.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    session.wait()


if __name__ == "__main__":
    main()
