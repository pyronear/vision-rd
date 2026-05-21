"""CLI: import the working-set sequences by id into the lab store.

Creds via env: PLATFORM_API_ENDPOINT, PLATFORM_LOGIN, PLATFORM_PASSWORD.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from tube_builder_lab import platform_api
from tube_builder_lab.import_sequences import import_keys
from tube_builder_lab.working_set import load_working_set

logging.basicConfig(level=logging.INFO)


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text())
    store_dir = Path(params["store"])
    detections_limit = int(params["detections_limit"])
    ws = load_working_set(Path("working_set.yaml"))
    keys = [i.key for i in ws.all()]

    endpoint = os.environ["PLATFORM_API_ENDPOINT"]
    token = platform_api.get_access_token(
        endpoint, os.environ["PLATFORM_LOGIN"], os.environ["PLATFORM_PASSWORD"]
    )
    n = import_keys(
        store_dir=store_dir,
        keys=keys,
        api_endpoint=endpoint,
        token=token,
        detections_limit=detections_limit,
    )
    print(f"imported {n} sequences into {store_dir}")


if __name__ == "__main__":
    main()
