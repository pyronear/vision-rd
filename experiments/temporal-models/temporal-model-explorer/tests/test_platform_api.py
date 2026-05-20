# tests/test_platform_api.py
from datetime import date

import temporal_model_explorer.platform_api as api


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_get_access_token_posts_creds(monkeypatch):
    seen = {}

    def fake_post(url, data, timeout):
        seen["url"], seen["data"] = url, data
        return _Resp({"access_token": "tok123"})

    monkeypatch.setattr(api.requests, "post", fake_post)
    tok = api.get_access_token("https://x", "u", "p")
    assert tok == "tok123"
    assert seen["url"] == "https://x/api/v1/login/creds"
    assert seen["data"] == {"username": "u", "password": "p"}


def test_list_sequences_for_date_builds_url(monkeypatch):
    seen = {}

    def fake_get(url, headers, timeout):
        seen["url"], seen["headers"] = url, headers
        return _Resp([{"id": 1}])

    monkeypatch.setattr(api.requests, "get", fake_get)
    out = api.list_sequences_for_date("https://x", "tok", date(2026, 5, 19), 100, 0)
    assert out == [{"id": 1}]
    assert "from_date=2026-05-19" in seen["url"]
    assert seen["headers"]["Authorization"] == "Bearer tok"


def test_list_sequence_detections_url(monkeypatch):
    monkeypatch.setattr(
        api.requests, "get", lambda url, headers, timeout: _Resp([{"id": 9}])
    )
    out = api.list_sequence_detections("https://x", "tok", 7, limit=5, desc=False)
    assert out == [{"id": 9}]
