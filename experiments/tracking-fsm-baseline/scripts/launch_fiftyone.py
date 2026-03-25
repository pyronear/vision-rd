"""Launch the FiftyOne app to browse tracking datasets.

Loads persisted tracking-fsm-* datasets and opens the FiftyOne web UI.

Usage:
    uv run --group explore python scripts/launch_fiftyone.py
"""

import fiftyone as fo


def main() -> None:
    datasets = fo.list_datasets(glob_patt="tracking-fsm-*")
    if not datasets:
        raise SystemExit(
            "No tracking-fsm-* datasets found. Run 'make fiftyone-build' first."
        )

    # Load val dataset by default if available, otherwise first match
    dataset_name = next(
        (d for d in sorted(datasets) if "val" in d),
        datasets[0],
    )
    dataset = fo.load_dataset(dataset_name)
    print(f"Launching FiftyOne with dataset '{dataset_name}' ({len(dataset)} samples)")
    print(f"Available datasets: {', '.join(sorted(datasets))}")
    print("Switch datasets in the FiftyOne UI via the dataset selector dropdown.")

    session = fo.launch_app(dataset, port=5151)
    session.wait()


if __name__ == "__main__":
    main()
