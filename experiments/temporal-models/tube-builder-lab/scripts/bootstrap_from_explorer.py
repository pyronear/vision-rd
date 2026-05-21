"""CLI: bootstrap the lab store from the local temporal-model-explorer store.

A no-creds convenience alternative to scripts/import_sequences.py for when the
working-set sequences already exist in the explorer's local data. Copies them
into the lab's flat store; track the result with DVC afterwards.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from tube_builder_lab.import_sequences import bootstrap_from_explorer
from tube_builder_lab.working_set import load_working_set

logging.basicConfig(level=logging.INFO)

DEFAULT_EXPLORER_STORE = Path("../temporal-model-explorer/data/03_primary/sequences")


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    lab_store = Path(params["store"])
    ws = load_working_set(Path("working_set.yaml"))
    keys = [i.key for i in ws.all()]
    copied, missing = bootstrap_from_explorer(
        lab_store=lab_store, explorer_store=DEFAULT_EXPLORER_STORE, keys=keys
    )
    print(f"copied {copied} sequences into {lab_store}")
    if missing:
        print(f"missing from explorer ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main()
