"""Launch the FiftyOne app to browse leaderboard pyro-annotator datasets.

Loads persisted ``leaderboard-pyro-annotator-*`` datasets and opens the
FiftyOne web UI. When multiple datasets are present (one per model),
picks one with "errors" in the name if available, else the first match.

Usage::

    uv run --group explore python scripts/launch_fiftyone.py
"""

import signal
import sys

import fiftyone as fo


def main() -> None:
    datasets = fo.list_datasets(glob_patt="leaderboard-pyro-annotator-*")
    if not datasets:
        raise SystemExit(
            "No leaderboard-pyro-annotator-* datasets found. "
            "Run 'python scripts/build_fiftyone_errors.py' first."
        )

    # Prefer an 'errors' dataset if present, else the first match.
    dataset_name = next(
        (d for d in sorted(datasets) if "errors" in d),
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
