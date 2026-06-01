"""CLI: import pyro-annotator sequences (extracted zip) into the store.

Reads creds from env: PLATFORM_API_ENDPOINT, PLATFORM_ADMIN_LOGIN,
PLATFORM_ADMIN_PASSWORD. Admin creds are required: the regular login is
org-scoped and returns 403 for these sequences.
"""

import argparse
import os
from pathlib import Path

from temporal_model_explorer import platform_api
from temporal_model_explorer.import_platform import build_camera_index, build_org_index
from temporal_model_explorer.import_pyro_annotator import import_pyro_annotator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="extracted zip root (the seq_annotation_done_by_label dir)",
    )
    ap.add_argument("--out", type=Path, default=Path("data/03_primary/sequences"))
    args = ap.parse_args()

    endpoint = os.environ["PLATFORM_API_ENDPOINT"]
    admin_login = os.environ.get("PLATFORM_ADMIN_LOGIN")
    admin_password = os.environ.get("PLATFORM_ADMIN_PASSWORD")
    if not (admin_login and admin_password):
        raise SystemExit(
            "PLATFORM_ADMIN_LOGIN/PLATFORM_ADMIN_PASSWORD are required: "
            "pyro-annotator sequences are only readable with admin creds."
        )

    token = platform_api.get_access_token(endpoint, admin_login, admin_password)
    camera_index = build_camera_index(endpoint, token)
    org_index = build_org_index(endpoint, token)

    n = import_pyro_annotator(
        args.src,
        args.out,
        endpoint,
        token,
        camera_index=camera_index,
        org_index=org_index,
    )
    print(f"imported {n} pyro-annotator sequences into {args.out}")


if __name__ == "__main__":
    main()
