"""CLI: import platform sequences for a date range into the store.

Reads creds from env: PLATFORM_API_ENDPOINT, PLATFORM_LOGIN, PLATFORM_PASSWORD.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import yaml

from temporal_model_explorer import platform_api
from temporal_model_explorer.import_platform import build_org_index, import_platform


def _date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/03_primary/sequences"))
    ap.add_argument("--date-from", type=_date, required=True)
    ap.add_argument("--date-to", type=_date, required=True)
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = ap.parse_args()

    params = yaml.safe_load(args.params.read_text())
    mapping = params["label_mapping"]
    camera_ids = set(params["platform"].get("camera_ids") or [])

    endpoint = os.environ["PLATFORM_API_ENDPOINT"]
    token = platform_api.get_access_token(
        endpoint, os.environ["PLATFORM_LOGIN"], os.environ["PLATFORM_PASSWORD"]
    )

    # Org-name enrichment (optional): static params map, overlaid with live names
    # from the admin /organizations endpoint when admin creds are present + valid.
    org_index: dict[int, str] = {
        int(k): v for k, v in (params.get("org_names") or {}).items()
    }
    admin_login = os.environ.get("PLATFORM_ADMIN_LOGIN")
    admin_password = os.environ.get("PLATFORM_ADMIN_PASSWORD")
    if admin_login and admin_password:
        try:
            admin_token = platform_api.get_access_token(
                endpoint, admin_login, admin_password
            )
            org_index.update(build_org_index(endpoint, admin_token))
        except Exception as exc:  # noqa: BLE001 - admin is optional; log and continue
            print(f"admin org enrichment skipped: {exc}")

    n = import_platform(
        endpoint,
        token,
        args.out,
        args.date_from,
        args.date_to,
        detections_limit=params["platform"]["detections_limit"],
        smoke_values=mapping["smoke_values"],
        fp_values=mapping["fp_values"],
        camera_ids=camera_ids or None,
        org_index=org_index or None,
    )
    print(f"imported {n} platform sequences into {args.out}")


if __name__ == "__main__":
    main()
