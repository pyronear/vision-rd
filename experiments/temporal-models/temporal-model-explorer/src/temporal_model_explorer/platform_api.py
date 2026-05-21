"""Lean Pyronear platform API client (mirror of pyro-dataset's platform/api.py).

Only the read endpoints the explorer needs; auth via /login/creds, bearer header.
Admin/organizations is optional and lives in ``list_organizations``.
"""

from __future__ import annotations

from datetime import date
from urllib.parse import urlencode

import requests


def get_access_token(api_endpoint: str, username: str, password: str) -> str:
    resp = requests.post(
        f"{api_endpoint}/api/v1/login/creds",
        data={"username": username, "password": password},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _get(route: str, token: str) -> object:
    resp = requests.get(route, headers=_headers(token), timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_cameras(api_endpoint: str, token: str) -> list[dict]:
    return _get(f"{api_endpoint}/api/v1/cameras/?include_non_trustable=true", token)


def list_organizations(api_endpoint: str, token: str) -> list[dict]:
    """Admin-only; call only when admin creds are available."""
    return _get(f"{api_endpoint}/api/v1/organizations/", token)


def list_sequences_for_date(
    api_endpoint: str, token: str, day: date, limit: int, offset: int
) -> list[dict]:
    query = urlencode(
        {"from_date": f"{day:%Y-%m-%d}", "limit": limit, "offset": offset}
    )
    return _get(f"{api_endpoint}/api/v1/sequences/all/fromdate?{query}", token)


def list_sequence_detections(
    api_endpoint: str, token: str, sequence_id: int, limit: int = 30, desc: bool = False
) -> list[dict]:
    desc_str = "true" if desc else "false"
    qs = f"limit={limit}&desc={desc_str}"
    route = f"{api_endpoint}/api/v1/sequences/{sequence_id}/detections?{qs}"
    return _get(route, token)
