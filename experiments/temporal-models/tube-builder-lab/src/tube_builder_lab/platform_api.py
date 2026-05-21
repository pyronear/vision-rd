"""Lean Pyronear platform API client — only the read endpoints this lab needs.

Duplicated (not shared) so the experiment stays fully isolated. Auth via
/login/creds, bearer header.
"""

from __future__ import annotations

import requests


def get_access_token(api_endpoint: str, username: str, password: str) -> str:
    resp = requests.post(
        f"{api_endpoint}/api/v1/login/creds",
        data={"username": username, "password": password},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def list_sequence_detections(
    api_endpoint: str, token: str, sequence_id: int, limit: int = 30
) -> list[dict]:
    route = (
        f"{api_endpoint}/api/v1/sequences/{sequence_id}/detections"
        f"?limit={limit}&desc=false"
    )
    resp = requests.get(route, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_image(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content
