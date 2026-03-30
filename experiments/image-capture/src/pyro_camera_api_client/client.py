# Copyright (C) 2022-2026, Pyronear.
# Source: https://github.com/pyronear/pyro-engine/blob/develop/pyro_camera_api/client/pyro_camera_api_client/client.py

from __future__ import annotations

import io
from typing import Any

import requests
from PIL import Image


class PyroCameraAPIClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        stream: bool = False,
        timeout: float | None = None,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        effective_timeout = self.timeout if timeout is None else timeout
        resp = requests.request(
            method=method,
            url=url,
            params=params,
            json=json,
            timeout=effective_timeout,
            stream=stream,
        )
        resp.raise_for_status()
        return resp

    def health(self) -> dict[str, Any]:
        resp = self._request("GET", "/health")
        return resp.json()

    def list_cameras(self) -> list[str]:
        resp = self._request("GET", "/cameras/cameras_list")
        return resp.json()

    def get_camera_infos(self) -> list[dict[str, Any]]:
        resp = self._request("GET", "/cameras/camera_infos")
        return resp.json()

    def get_latest_image(
        self,
        camera_ip: str,
        pose: int,
        quality: int | None = None,
    ) -> Image.Image | None:
        params: dict[str, Any] = {"camera_ip": camera_ip, "pose": pose}
        if quality is not None:
            params["quality"] = quality

        resp = self._request("GET", "/cameras/latest_image", params=params, stream=True)

        if resp.status_code == 204 or not resp.content:
            return None

        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def list_presets(self, camera_ip: str) -> dict[str, Any]:
        params = {"camera_ip": camera_ip}
        resp = self._request("GET", "/control/preset/list", params=params)
        return resp.json()
