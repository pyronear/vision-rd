# Temporal Model Explorer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local tool that imports Pyronear camera-event sequences (platform API + a local annotated zip), runs one or more packaged temporal models on them, and shows in Streamlit which sequences each model would **KEEP** vs **DISCARD**, with filters.

**Architecture:** Three-stage pipeline (`import → run → view`) writing intermediate files under Kedro-style `data/` layers, plus a thin Streamlit viewer. The model runner reuses the Phase-1 lib `bbox-tube-temporal-core` (`BboxTubeTemporalModel.from_package`). Results are written frontend-agnostic (`results.parquet` + per-sequence `details` JSON) so the viewer is swappable.

**Tech Stack:** Python 3.11, uv, pandas/pyarrow, requests, pyyaml, pillow, streamlit, pytest, ruff; depends on `lib/bbox-tube-temporal/` (+ `pyrocore`).

**Spec:** `docs/specs/2026-05-20-temporal-model-explorer-design.md`.

---

## File structure

```
experiments/temporal-models/temporal-model-explorer/
  pyproject.toml      # deps + uv.sources to the lib & pyrocore
  Makefile            # install / lint / format / test / app
  params.yaml         # model registry, label mapping, platform defaults
  dvc.yaml            # import_local_zip + run_models stages
  .gitignore          # .venv/ data/ __pycache__/
  src/temporal_model_explorer/
    __init__.py
    platform_api.py     # lean platform client (mirror of pyro-dataset client)
    store.py            # SequenceMeta/FrameRef types, meta.json IO, label + Frame helpers
    outcomes.py         # decision/outcome/probability + results filtering (pure)
    import_local_zip.py # annotated zip  -> store
    import_platform.py  # platform API   -> store
    run_models.py       # store -> results.parquet (+ details JSON)
    app.py              # Streamlit viewer (not unit-tested)
  scripts/
    prepare_models.py   # copy registry model.zips -> DVC-tracked data/06_models/
    import_local_zip.py # argparse wrapper -> import_local_zip.import_zip
    import_platform.py  # argparse wrapper -> import_platform.import_platform
    run_models.py       # argparse wrapper -> run_models.run_over_store
  tests/
    __init__.py
    fixtures/
  .dvc/                 # this experiment's own DVC project (dvc init --subdir)
```

**Common store layout** (produced by both importers, scanned by the runner):
`data/03_primary/sequences/<source>/<key>/{images/…, meta.json}` where
`<source>` is `local_zip` or `platform` and `<key>` is `zip_<id>` / `platform_<id>`.

**DVC-tracked artifacts:** `data/06_models/<name>/model.zip` (from `prepare_models`),
`data/03_primary/sequences/local_zip` (from `import_local_zip`), and
`data/07_model_output/{results.parquet,details}` (from `run_models`).

---

### Task 1: Scaffold the experiment project

**Files:**
- Create: `experiments/temporal-models/temporal-model-explorer/pyproject.toml`
- Create: `…/Makefile`, `…/.gitignore`, `…/params.yaml`
- Create: `…/src/temporal_model_explorer/__init__.py` (empty), `…/tests/__init__.py` (empty)

- [ ] **Step 1: `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/temporal_model_explorer"]

[project]
name = "temporal-model-explorer"
version = "0.1.0"
description = "Local tool to run & compare temporal smoke models (keep/discard) on platform + annotated sequences"
requires-python = ">=3.11"
dependencies = [
    "bbox-tube-temporal-core",
    "pyrocore",
    "requests>=2.31",
    "pandas>=2.0",
    "pyarrow>=15",
    "pyyaml>=6.0",
    "pillow>=10.0",
    "streamlit>=1.40",
]

[tool.uv.sources]
bbox-tube-temporal-core = { path = "../../../lib/bbox-tube-temporal" }
pyrocore = { path = "../../../lib/pyrocore" }

[dependency-groups]
dev = ["pytest>=8.0", "ruff>=0.9", "dvc[s3]>=3.56"]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "PLC0415"]

[tool.ruff.format]
quote-style = "double"
```

- [ ] **Step 2: `Makefile`**

```makefile
.PHONY: install lint format test app help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

lint: ## Run ruff linter
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

test: ## Run tests
	uv run pytest tests/ -v

app: ## Launch the Streamlit viewer
	uv run streamlit run src/temporal_model_explorer/app.py
```

- [ ] **Step 3: `.gitignore`**

```gitignore
.venv/
data/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 4: `params.yaml`**

```yaml
platform:
  detections_limit: 30
  date_from: "2026-05-19"
  date_to: "2026-05-19"
  camera_ids: []          # empty = all cameras

label_mapping:
  smoke_values: [wildfire, other_smoke, industrial, smoke]
  fp_values: [other, low_cloud, high_cloud, cloud, tree, water_body, lens_droplet, light, antenna, building]

org_names: {}             # optional org_id -> name (fallback when admin unavailable)

models:                   # name -> SOURCE model.zip path (copied into the DVC-tracked
                          # data/06_models/<name>/model.zip by the prepare_models stage)
  vit_dinov2_finetune: ../bbox-tube-temporal/data/06_models/vit_dinov2_finetune/model.zip
  gru_convnext_finetune: ../bbox-tube-temporal/data/06_models/gru_convnext_finetune/model.zip

local_zip: ../data/seq_annotation_done_by_label.zip
```

- [ ] **Step 5: empty package files**

```bash
cd experiments/temporal-models/temporal-model-explorer
mkdir -p src/temporal_model_explorer tests scripts
: > src/temporal_model_explorer/__init__.py
: > tests/__init__.py
```

- [ ] **Step 6: install**

Run: `cd experiments/temporal-models/temporal-model-explorer && uv sync`
Expected: resolves `bbox-tube-temporal-core` + `pyrocore` from the local paths and installs streamlit/pandas/dvc/etc. with no error.

- [ ] **Step 7: init this experiment's own DVC project**

DVC is per-experiment here (each experiment has its own `.dvc/`; there is no
repo-root DVC project). Initialize one for the explorer:

Run: `uv run dvc init --subdir`
Expected: creates `.dvc/` (config, `.gitignore`) and `.dvcignore`, staged for git.

- [ ] **Step 8: commit**

```bash
E=experiments/temporal-models/temporal-model-explorer
git add "$E/pyproject.toml" "$E/Makefile" "$E/.gitignore" "$E/params.yaml" \
        "$E/src/temporal_model_explorer/__init__.py" "$E/tests/__init__.py" \
        "$E/uv.lock" "$E/.dvc/config" "$E/.dvc/.gitignore" "$E/.dvcignore"
git commit -m "feat(explorer): scaffold temporal-model-explorer experiment (+ dvc init)"
```

---

### Task 2: `outcomes.py` — decision / outcome / probability / filtering (pure logic)

**Files:**
- Create: `src/temporal_model_explorer/outcomes.py`
- Test: `tests/test_outcomes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_outcomes.py
import pandas as pd
from temporal_model_explorer.outcomes import (
    decision_from_output, max_probability, compute_outcome, filter_results,
)


def test_decision_from_output():
    assert decision_from_output(True) == "keep"
    assert decision_from_output(False) == "discard"


def test_max_probability_picks_max_over_kept_tubes():
    details = {"tubes": {"kept": [{"probability": 0.2}, {"probability": 0.8}, {"probability": None}]}}
    assert max_probability(details) == 0.8


def test_max_probability_none_when_no_probs():
    assert max_probability({"tubes": {"kept": [{"probability": None}]}}) is None
    assert max_probability({}) is None


def test_compute_outcome_all_branches():
    assert compute_outcome("keep", "smoke") == "kept-smoke"
    assert compute_outcome("discard", "smoke") == "discarded-smoke"
    assert compute_outcome("keep", "fp") == "kept-fp"
    assert compute_outcome("discard", "fp") == "discarded-fp"
    assert compute_outcome("keep", "unknown") == "n/a"


def test_filter_results_by_decision_and_label():
    df = pd.DataFrame([
        {"decision": "keep", "label": "smoke", "outcome": "kept-smoke"},
        {"decision": "discard", "label": "fp", "outcome": "discarded-fp"},
    ])
    out = filter_results(df, decision="discard")
    assert list(out["label"]) == ["fp"]
    assert len(filter_results(df, label="smoke")) == 1
    assert len(filter_results(df, errors_only=True)) == 0
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest tests/test_outcomes.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `outcomes.py`**

```python
"""Pure decision/outcome helpers + results filtering (no I/O, no Streamlit)."""

from __future__ import annotations

from typing import Any

import pandas as pd

ERROR_OUTCOMES = {"discarded-smoke", "kept-fp"}


def decision_from_output(is_positive: bool) -> str:
    """Map a model's is_positive verdict to keep/discard."""
    return "keep" if is_positive else "discard"


def max_probability(details: dict[str, Any] | None) -> float | None:
    """Largest calibrated probability across kept tubes, or None."""
    kept = (details or {}).get("tubes", {}).get("kept", [])
    probs = [t.get("probability") for t in kept if t.get("probability") is not None]
    return max(probs) if probs else None


def compute_outcome(decision: str, label: str) -> str:
    """Outcome of a decision vs the ground-truth label."""
    if label == "smoke":
        return "kept-smoke" if decision == "keep" else "discarded-smoke"
    if label == "fp":
        return "kept-fp" if decision == "keep" else "discarded-fp"
    return "n/a"


def filter_results(
    df: pd.DataFrame,
    *,
    model: str | None = None,
    decision: str | None = None,
    label: str | None = None,
    outcome: str | None = None,
    source: str | None = None,
    camera_name: str | None = None,
    organization_name: str | None = None,
    errors_only: bool = False,
) -> pd.DataFrame:
    """Apply the Streamlit sidebar filters to the results frame."""
    out = df
    for col, val in (
        ("model", model), ("decision", decision), ("label", label),
        ("outcome", outcome), ("source", source), ("camera_name", camera_name),
        ("organization_name", organization_name),
    ):
        if val is not None:
            out = out[out[col] == val]
    if errors_only:
        out = out[out["outcome"].isin(ERROR_OUTCOMES)]
    return out
```

- [ ] **Step 4: Run → pass**

Run: `uv run pytest tests/test_outcomes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/temporal_model_explorer/outcomes.py tests/test_outcomes.py
git commit -m "feat(explorer): decision/outcome/probability + results filtering"
```

---

### Task 3: `store.py` — sequence store types, meta.json IO, label + Frame helpers

**Files:**
- Create: `src/temporal_model_explorer/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_store.py
from pathlib import Path

from pyrocore import Frame
from temporal_model_explorer.store import (
    FrameRef, SequenceMeta, write_meta, read_meta, iter_sequence_dirs,
    normalize_label, build_frames,
)

SMOKE = ["wildfire", "other_smoke"]
FP = ["other", "low_cloud"]


def _meta(key="zip_1"):
    return SequenceMeta(
        key=key, sequence_id="1", source="local_zip", label="smoke",
        label_detail="wildfire", label_source="zip_folder",
        frames=[FrameRef(file="images/detection_5.jpg", detection_id=5, created_at=None)],
    )


def test_meta_roundtrip(tmp_path):
    d = tmp_path / "zip_1"
    write_meta(d, _meta())
    got = read_meta(d)
    assert got == _meta()
    assert (d / "meta.json").exists()


def test_iter_sequence_dirs_finds_meta_recursively(tmp_path):
    write_meta(tmp_path / "local_zip" / "zip_1", _meta("zip_1"))
    write_meta(tmp_path / "platform" / "platform_2", _meta("platform_2"))
    found = {p.name for p in iter_sequence_dirs(tmp_path)}
    assert found == {"zip_1", "platform_2"}


def test_normalize_label():
    assert normalize_label("wildfire", SMOKE, FP) == "smoke"
    assert normalize_label("other_smoke", SMOKE, FP) == "smoke"
    assert normalize_label("other", SMOKE, FP) == "fp"
    assert normalize_label("low_cloud", SMOKE, FP) == "fp"
    assert normalize_label(None, SMOKE, FP) == "unknown"
    assert normalize_label("mystery", SMOKE, FP) == "unknown"


def test_build_frames_orders_and_resolves_paths(tmp_path):
    d = tmp_path / "zip_1"
    (d / "images").mkdir(parents=True)
    (d / "images" / "detection_5.jpg").write_bytes(b"x")
    frames = build_frames(d, _meta())
    assert isinstance(frames[0], Frame)
    assert frames[0].image_path == d / "images" / "detection_5.jpg"
    assert frames[0].frame_id == "detection_5"
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `store.py`**

```python
"""Common local sequence store: types, meta.json IO, label + Frame helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from pyrocore import Frame

META_FILENAME = "meta.json"


@dataclass
class FrameRef:
    file: str                       # path relative to the sequence dir, e.g. "images/detection_5.jpg"
    detection_id: int | None = None
    created_at: str | None = None   # ISO timestamp (platform) or None (zip)


@dataclass
class SequenceMeta:
    key: str
    sequence_id: str
    source: str                     # "platform" | "local_zip"
    label: str                      # "smoke" | "fp" | "unknown"
    label_detail: str | None
    label_source: str               # "platform_is_wildfire" | "zip_folder"
    frames: list[FrameRef] = field(default_factory=list)
    camera_id: int | None = None
    camera_name: str | None = None
    organization_id: int | None = None
    organization_name: str | None = None
    started_at: str | None = None


def write_meta(seq_dir: Path, meta: SequenceMeta) -> None:
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / META_FILENAME).write_text(json.dumps(asdict(meta), indent=2))


def read_meta(seq_dir: Path) -> SequenceMeta:
    payload = json.loads((seq_dir / META_FILENAME).read_text())
    frames = [FrameRef(**f) for f in payload.pop("frames", [])]
    return SequenceMeta(frames=frames, **payload)


def iter_sequence_dirs(store_dir: Path) -> Iterator[Path]:
    """Yield every directory under ``store_dir`` containing a meta.json (recursive)."""
    if not store_dir.exists():
        return
    for meta_path in sorted(store_dir.rglob(META_FILENAME)):
        yield meta_path.parent


def normalize_label(raw: str | None, smoke_values: list[str], fp_values: list[str]) -> str:
    """Normalize a raw category to the tri-state keep/discard label."""
    if not raw:
        return "unknown"
    v = raw.lower()
    if v in {s.lower() for s in smoke_values} or "smoke" in v or v == "wildfire":
        return "smoke"
    if v in {f.lower() for f in fp_values}:
        return "fp"
    return "unknown"


def build_frames(seq_dir: Path, meta: SequenceMeta) -> list[Frame]:
    """Build the ordered pyrocore Frame list the model consumes (meta order = time axis)."""
    frames: list[Frame] = []
    for ref in meta.frames:
        ts = None
        if ref.created_at:
            try:
                ts = datetime.fromisoformat(ref.created_at.replace("Z", "+00:00"))
            except ValueError:
                ts = None
        frames.append(
            Frame(frame_id=Path(ref.file).stem, image_path=seq_dir / ref.file, timestamp=ts)
        )
    return frames
```

- [ ] **Step 4: Run → pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/temporal_model_explorer/store.py tests/test_store.py
git commit -m "feat(explorer): sequence store types + meta IO + label/Frame helpers"
```

---

### Task 4: `platform_api.py` — lean platform client (mirror)

**Files:**
- Create: `src/temporal_model_explorer/platform_api.py`
- Test: `tests/test_platform_api.py`

- [ ] **Step 1: Write failing tests**

```python
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
    monkeypatch.setattr(api.requests, "get", lambda url, headers, timeout: _Resp([{"id": 9}]))
    out = api.list_sequence_detections("https://x", "tok", 7, limit=5, desc=False)
    assert out == [{"id": 9}]
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest tests/test_platform_api.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `platform_api.py`**

```python
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
    query = urlencode({"from_date": f"{day:%Y-%m-%d}", "limit": limit, "offset": offset})
    return _get(f"{api_endpoint}/api/v1/sequences/all/fromdate?{query}", token)


def list_sequence_detections(
    api_endpoint: str, token: str, sequence_id: int, limit: int = 30, desc: bool = False
) -> list[dict]:
    desc_str = "true" if desc else "false"
    route = f"{api_endpoint}/api/v1/sequences/{sequence_id}/detections?limit={limit}&desc={desc_str}"
    return _get(route, token)
```

- [ ] **Step 4: Run → pass**

Run: `uv run pytest tests/test_platform_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/temporal_model_explorer/platform_api.py tests/test_platform_api.py
git commit -m "feat(explorer): lean platform API client"
```

---

### Task 5: `import_local_zip.py` — annotated zip → store

**Files:**
- Create: `src/temporal_model_explorer/import_local_zip.py`
- Test: `tests/test_import_local_zip.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_import_local_zip.py
import zipfile
from pathlib import Path

from temporal_model_explorer.import_local_zip import label_from_parts, import_zip
from temporal_model_explorer.store import read_meta


def test_label_from_parts():
    assert label_from_parts(("smoke", "wildfire")) == ("smoke", "wildfire")
    assert label_from_parts(("fp", "tree")) == ("fp", "tree")
    assert label_from_parts(("unlabeled",)) == ("unknown", None)
    assert label_from_parts(("smoke",)) == ("smoke", None)


def _make_zip(path: Path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("root/smoke/wildfire/seq_10/images/detection_2.jpg", b"b")
        zf.writestr("root/smoke/wildfire/seq_10/images/detection_1.jpg", b"a")
        zf.writestr("root/smoke/wildfire/seq_10/labels/detection_1.txt", b"0 0 0 0 0")
        zf.writestr("root/fp/tree/seq_20/images/detection_3.jpg", b"c")
        zf.writestr("__MACOSX/root/._x", b"junk")
        zf.writestr("root/.DS_Store", b"junk")


def test_import_zip_writes_store(tmp_path):
    z = tmp_path / "data.zip"
    _make_zip(z)
    store = tmp_path / "store"
    n = import_zip(z, store)
    assert n == 2

    smoke = read_meta(store / "zip_10")
    assert smoke.label == "smoke" and smoke.label_detail == "wildfire"
    assert smoke.source == "local_zip" and smoke.sequence_id == "10"
    # images copied, ordered by filename (detection_1 before detection_2)
    assert [f.file for f in smoke.frames] == ["images/detection_1.jpg", "images/detection_2.jpg"]
    assert (store / "zip_10" / "images" / "detection_1.jpg").read_bytes() == b"a"

    fp = read_meta(store / "zip_20")
    assert fp.label == "fp" and fp.label_detail == "tree"
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest tests/test_import_local_zip.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `import_local_zip.py`**

```python
"""Import the annotated `seq_annotation_done_by_label` zip into the sequence store.

Zip layout: <root>/<category>[/<detail>]/seq_<id>/images/detection_<id>.jpg
  category: "smoke" | "fp" | "unlabeled"; detail: subfolder (wildfire, tree, …).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .store import FrameRef, SequenceMeta, write_meta

_DET_RE = re.compile(r"detection_(\d+)")


def label_from_parts(parts: tuple[str, ...]) -> tuple[str, str | None]:
    """Map the category path components (between zip root and seq dir) to (label, detail)."""
    if not parts:
        return "unknown", None
    top = parts[0]
    detail = parts[1] if len(parts) > 1 else None
    if top == "smoke":
        return "smoke", detail
    if top == "fp":
        return "fp", detail
    return "unknown", None


def _detection_id(file_name: str) -> int | None:
    m = _DET_RE.search(file_name)
    return int(m.group(1)) if m else None


def import_zip(zip_path: Path, store_dir: Path) -> int:
    """Extract image frames + write meta.json per sequence. Returns #sequences imported."""
    store_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict] = {}

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            if "__MACOSX" in name or name.endswith("/") or Path(name).name == ".DS_Store":
                continue
            parts = Path(name).parts
            if "images" not in parts:
                continue
            idx = parts.index("images")
            if idx == 0 or not parts[idx - 1].startswith("seq_"):
                continue
            seq_dirname = parts[idx - 1]
            category_parts = parts[1:idx - 1]  # drop zip root at parts[0]
            label, detail = label_from_parts(tuple(category_parts))
            entry = grouped.setdefault(
                seq_dirname, {"label": label, "detail": detail, "files": []}
            )
            entry["files"].append((name, parts[-1]))

        count = 0
        for seq_dirname, entry in grouped.items():
            seq_id = seq_dirname.removeprefix("seq_")
            key = f"zip_{seq_id}"
            out_dir = store_dir / key
            (out_dir / "images").mkdir(parents=True, exist_ok=True)
            frames: list[FrameRef] = []
            for src_name, file_name in sorted(entry["files"], key=lambda t: t[1]):
                (out_dir / "images" / file_name).write_bytes(zf.read(src_name))
                frames.append(
                    FrameRef(file=f"images/{file_name}", detection_id=_detection_id(file_name))
                )
            write_meta(
                out_dir,
                SequenceMeta(
                    key=key, sequence_id=seq_id, source="local_zip",
                    label=entry["label"], label_detail=entry["detail"],
                    label_source="zip_folder", frames=frames,
                ),
            )
            count += 1
    return count
```

- [ ] **Step 4: Run → pass**

Run: `uv run pytest tests/test_import_local_zip.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/temporal_model_explorer/import_local_zip.py tests/test_import_local_zip.py
git commit -m "feat(explorer): import annotated zip into the sequence store"
```

---

### Task 6: `import_platform.py` — platform API → store

**Files:**
- Create: `src/temporal_model_explorer/import_platform.py`
- Test: `tests/test_import_platform.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_import_platform.py
from datetime import date

import temporal_model_explorer.import_platform as ip
from temporal_model_explorer.store import read_meta

SMOKE = ["wildfire", "other_smoke"]
FP = ["other"]


def test_import_platform_writes_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ip.platform_api, "list_sequences_for_date",
        lambda ep, tok, day, limit, offset: [
            {"id": 43392, "camera_id": 7, "is_wildfire": "other_smoke", "started_at": "2026-05-19T14:10:01Z"}
        ],
    )
    monkeypatch.setattr(
        ip.platform_api, "list_sequence_detections",
        lambda ep, tok, sid, limit, desc: [
            {"id": 2, "url": "http://img/2", "created_at": "2026-05-19T14:10:31Z"},
            {"id": 1, "url": "http://img/1", "created_at": "2026-05-19T14:10:01Z"},
        ],
    )
    camera_index = {7: {"id": 7, "name": "cam-7", "organization_id": 3}}
    n = ip.import_platform(
        "https://x", "tok", tmp_path, date(2026, 5, 19), date(2026, 5, 19),
        detections_limit=5, smoke_values=SMOKE, fp_values=FP,
        camera_index=camera_index, download=lambda url: f"BYTES:{url}".encode(),
    )
    assert n == 1
    meta = read_meta(tmp_path / "platform_43392")
    assert meta.label == "smoke" and meta.label_detail == "other_smoke"
    assert meta.camera_name == "cam-7" and meta.organization_id == 3
    # frames ordered by created_at ascending (detection 1 then 2)
    assert [f.detection_id for f in meta.frames] == [1, 2]
    assert (tmp_path / "platform_43392" / "images" / "detection_1.jpg").read_bytes() == b"BYTES:http://img/1"


def test_camera_filter_excludes_other_cameras(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ip.platform_api, "list_sequences_for_date",
        lambda ep, tok, day, limit, offset: [{"id": 1, "camera_id": 99, "is_wildfire": "other"}],
    )
    n = ip.import_platform(
        "https://x", "tok", tmp_path, date(2026, 5, 19), date(2026, 5, 19),
        detections_limit=5, smoke_values=SMOKE, fp_values=FP,
        camera_ids={7}, camera_index={}, download=lambda url: b"x",
    )
    assert n == 0
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest tests/test_import_platform.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `import_platform.py`**

```python
"""Import platform sequences (date range, optional camera filter) into the store.

Each detection's full-frame image is downloaded via its presigned ``url``. Frames
are ordered by detection ``created_at``. Requires only regular platform creds.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import requests

from . import platform_api
from .store import FrameRef, SequenceMeta, normalize_label, write_meta

log = logging.getLogger(__name__)


def download_image(url: str) -> bytes:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def build_camera_index(api_endpoint: str, token: str) -> dict[int, dict]:
    return {c["id"]: c for c in platform_api.list_cameras(api_endpoint, token)}


def _import_one(
    api_endpoint, token, store_dir, seq, camera_index, detections_limit,
    smoke_values, fp_values, download,
) -> int:
    sid = seq["id"]
    raw_label = seq.get("is_wildfire")
    cam = camera_index.get(seq.get("camera_id"), {})
    dets = platform_api.list_sequence_detections(
        api_endpoint, token, sid, limit=detections_limit, desc=False
    )
    dets = sorted(dets, key=lambda d: d.get("created_at") or "")
    out_dir = store_dir / f"platform_{sid}"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    frames: list[FrameRef] = []
    for det in dets:
        url = det.get("url")
        if not url:
            log.warning("detection %s of seq %s has no url; skipping", det.get("id"), sid)
            continue
        try:
            data = download(url)
        except Exception as exc:  # noqa: BLE001 - log + skip a bad frame, keep going
            log.warning("download failed for detection %s: %s", det.get("id"), exc)
            continue
        fname = f"detection_{det['id']}.jpg"
        (out_dir / "images" / fname).write_bytes(data)
        frames.append(
            FrameRef(file=f"images/{fname}", detection_id=det["id"], created_at=det.get("created_at"))
        )
    write_meta(
        out_dir,
        SequenceMeta(
            key=f"platform_{sid}", sequence_id=str(sid), source="platform",
            label=normalize_label(raw_label, smoke_values, fp_values),
            label_detail=raw_label, label_source="platform_is_wildfire",
            frames=frames, camera_id=seq.get("camera_id"),
            camera_name=cam.get("name"), organization_id=cam.get("organization_id"),
            started_at=seq.get("started_at"),
        ),
    )
    return 1


def import_platform(
    api_endpoint: str, token: str, store_dir: Path, day_from: date, day_to: date, *,
    detections_limit: int, smoke_values: list[str], fp_values: list[str],
    camera_ids: set[int] | None = None, camera_index: dict | None = None,
    download=download_image,
) -> int:
    """Import all sequences in [day_from, day_to]. Returns #sequences imported."""
    store_dir.mkdir(parents=True, exist_ok=True)
    if camera_index is None:
        camera_index = build_camera_index(api_endpoint, token)
    count = 0
    day = day_from
    while day <= day_to:
        for seq in platform_api.list_sequences_for_date(api_endpoint, token, day, 100, 0):
            if camera_ids and seq.get("camera_id") not in camera_ids:
                continue
            count += _import_one(
                api_endpoint, token, store_dir, seq, camera_index,
                detections_limit, smoke_values, fp_values, download,
            )
        day += timedelta(days=1)
    return count
```

- [ ] **Step 4: Run → pass**

Run: `uv run pytest tests/test_import_platform.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/temporal_model_explorer/import_platform.py tests/test_import_platform.py
git commit -m "feat(explorer): import platform sequences into the store"
```

---

### Task 7: `run_models.py` — store → results.parquet (+ details JSON)

**Files:**
- Create: `src/temporal_model_explorer/run_models.py`
- Test: `tests/test_run_models.py`

- [ ] **Step 1: Write failing tests** (uses a fake model — no real `model.zip`)

```python
# tests/test_run_models.py
from pathlib import Path

from temporal_model_explorer.run_models import run_over_store
from temporal_model_explorer.store import FrameRef, SequenceMeta, write_meta


class _Output:
    def __init__(self, is_positive, trigger, details):
        self.is_positive = is_positive
        self.trigger_frame_index = trigger
        self.details = details


class _FakeModel:
    """Keeps a sequence iff it has >= 3 frames; triggers on frame index 1."""

    def predict(self, frames):
        keep = len(frames) >= 3
        details = {"tubes": {"kept": [{"probability": 0.9}]}} if keep else {"tubes": {"kept": []}}
        return _Output(keep, 1 if keep else None, details)


def _seq(store, key, label, n):
    d = store / "local_zip" / key
    (d / "images").mkdir(parents=True)
    frames = []
    for i in range(n):
        (d / "images" / f"detection_{i}.jpg").write_bytes(b"x")
        frames.append(FrameRef(file=f"images/detection_{i}.jpg", detection_id=i))
    write_meta(d, SequenceMeta(key=key, sequence_id=key.split("_")[-1], source="local_zip",
                               label=label, label_detail=None, label_source="zip_folder", frames=frames))


def test_run_over_store_writes_results(tmp_path):
    store = tmp_path / "sequences"
    _seq(store, "zip_1", "smoke", 4)   # kept  -> kept-smoke
    _seq(store, "zip_2", "fp", 2)      # discarded -> discarded-fp
    results = tmp_path / "out" / "results.parquet"
    details = tmp_path / "out" / "details"

    df = run_over_store(store, {"fake": _FakeModel()}, results, details)

    assert results.exists()
    rows = {r["key"]: r for r in df.to_dict("records")}
    assert rows["zip_1"]["decision"] == "keep"
    assert rows["zip_1"]["outcome"] == "kept-smoke"
    assert rows["zip_1"]["trigger_frame_file"] == "images/detection_1.jpg"
    assert rows["zip_1"]["probability"] == 0.9
    assert rows["zip_2"]["decision"] == "discard"
    assert rows["zip_2"]["outcome"] == "discarded-fp"
    assert rows["zip_2"]["trigger_frame_file"] is None
    assert (details / "fake" / "zip_1.json").exists()


def test_load_models_scans_dir(tmp_path, monkeypatch):
    import temporal_model_explorer.run_models as rm
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "model.zip").write_bytes(b"x")
    (tmp_path / "empty").mkdir()  # no model.zip -> skipped
    monkeypatch.setattr(
        rm.BboxTubeTemporalModel, "from_package",
        staticmethod(lambda p, device="cpu": f"loaded:{p}"),
    )
    models = rm.load_models(tmp_path)
    assert set(models) == {"m1"}


def test_load_models_missing_dir(tmp_path):
    from temporal_model_explorer.run_models import load_models
    assert load_models(tmp_path / "nope") == {}
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest tests/test_run_models.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `run_models.py`**

```python
"""Run configured temporal models over the sequence store and write results."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
from bbox_tube_temporal.model import BboxTubeTemporalModel

from .outcomes import compute_outcome, decision_from_output, max_probability
from .store import build_frames, iter_sequence_dirs, read_meta

log = logging.getLogger(__name__)


def load_models(models_dir: Path, device: str = "cpu") -> dict:
    """Load every ``<name>/model.zip`` under the DVC-tracked ``models_dir``.

    Names are the subdirectory names; a subdir without a ``model.zip`` is skipped.
    """
    models: dict[str, object] = {}
    if not models_dir.exists():
        return models
    for sub in sorted(models_dir.iterdir()):
        pkg = sub / "model.zip"
        if not pkg.exists():
            log.warning("no model.zip under %s; skipping", sub)
            continue
        models[sub.name] = BboxTubeTemporalModel.from_package(pkg, device=device)
    return models


def run_over_store(
    store_dir: Path, models: dict, results_path: Path, details_dir: Path
) -> pd.DataFrame:
    """Run every model over every stored sequence; write parquet + per-seq details JSON."""
    rows: list[dict] = []
    for seq_dir in iter_sequence_dirs(store_dir):
        meta = read_meta(seq_dir)
        frames = build_frames(seq_dir, meta)
        for name, model in models.items():
            t0 = time.perf_counter()
            out = model.predict(frames)
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            decision = decision_from_output(out.is_positive)
            tfile = None
            if out.trigger_frame_index is not None and 0 <= out.trigger_frame_index < len(meta.frames):
                tfile = meta.frames[out.trigger_frame_index].file
            rows.append({
                "key": meta.key, "source": meta.source, "sequence_id": meta.sequence_id,
                "camera_id": meta.camera_id, "camera_name": meta.camera_name,
                "organization_id": meta.organization_id, "organization_name": meta.organization_name,
                "label": meta.label, "label_detail": meta.label_detail, "n_frames": len(frames),
                "model": name, "decision": decision,
                "trigger_frame_index": out.trigger_frame_index, "trigger_frame_file": tfile,
                "probability": max_probability(out.details),
                "outcome": compute_outcome(decision, meta.label), "runtime_ms": runtime_ms,
            })
            model_details = details_dir / name
            model_details.mkdir(parents=True, exist_ok=True)
            (model_details / f"{meta.key}.json").write_text(json.dumps(out.details, indent=2, default=str))
    df = pd.DataFrame(rows)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(results_path)
    return df
```

- [ ] **Step 4: Run → pass**

Run: `uv run pytest tests/test_run_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/temporal_model_explorer/run_models.py tests/test_run_models.py
git commit -m "feat(explorer): model runner -> results.parquet + details"
```

---

### Task 8: CLI wrappers + DVC pipeline

**Files:**
- Create: `scripts/import_local_zip.py`, `scripts/import_platform.py`, `scripts/run_models.py`
- Create: `dvc.yaml`

- [ ] **Step 1: `scripts/import_local_zip.py`**

```python
"""CLI: import the annotated zip into the sequence store."""

import argparse
from pathlib import Path

from temporal_model_explorer.import_local_zip import import_zip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/03_primary/sequences/local_zip"))
    args = ap.parse_args()
    n = import_zip(args.zip, args.out)
    print(f"imported {n} sequences into {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `scripts/import_platform.py`**

```python
"""CLI: import platform sequences for a date range into the store.

Reads creds from env: PLATFORM_API_ENDPOINT, PLATFORM_LOGIN, PLATFORM_PASSWORD.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import yaml

from temporal_model_explorer import platform_api
from temporal_model_explorer.import_platform import import_platform


def _date(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/03_primary/sequences/platform"))
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
    n = import_platform(
        endpoint, token, args.out, args.date_from, args.date_to,
        detections_limit=params["platform"]["detections_limit"],
        smoke_values=mapping["smoke_values"], fp_values=mapping["fp_values"],
        camera_ids=camera_ids or None,
    )
    print(f"imported {n} platform sequences into {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: `scripts/run_models.py`**

```python
"""CLI: run the models in the DVC-tracked models dir over the sequence store."""

import argparse
from pathlib import Path

from temporal_model_explorer.run_models import load_models, run_over_store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("data/03_primary/sequences"))
    ap.add_argument("--models-dir", type=Path, default=Path("data/06_models"))
    ap.add_argument("--out", type=Path, default=Path("data/07_model_output"))
    args = ap.parse_args()

    models = load_models(args.models_dir)
    if not models:
        raise SystemExit(f"No models under {args.models_dir} (run prepare_models first).")
    df = run_over_store(args.store, models, args.out / "results.parquet", args.out / "details")
    print(f"ran {len(models)} model(s) over {df['key'].nunique()} sequences -> {args.out}/results.parquet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: `scripts/prepare_models.py`** (copies source packages into the DVC-tracked models dir)

```python
"""CLI: copy the configured model packages into the DVC-tracked models dir.

The source paths (in params.yaml `models`) live outside this experiment's DVC
root, so they are read by the cmd rather than declared as DVC deps. The output
``data/06_models`` is a DVC out, so the models become DVC-tracked here.
"""

import argparse
import shutil
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/06_models"))
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = ap.parse_args()

    params = yaml.safe_load(args.params.read_text())
    count = 0
    for name, src in params["models"].items():
        src_path = Path(src)
        if not src_path.exists():
            print(f"skip {name}: source not found at {src_path}")
            continue
        dest = args.out / name / "model.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest)
        count += 1
        print(f"copied {name} -> {dest}")
    print(f"prepared {count} model(s) in {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: `dvc.yaml`**

```yaml
stages:
  prepare_models:
    cmd: uv run python scripts/prepare_models.py --out data/06_models
    deps:
      - scripts/prepare_models.py
    params:
      - models
    outs:
      - data/06_models

  import_local_zip:
    cmd: uv run python scripts/import_local_zip.py --zip ../data/seq_annotation_done_by_label.zip --out data/03_primary/sequences/local_zip
    deps:
      - src/temporal_model_explorer/import_local_zip.py
      - src/temporal_model_explorer/store.py
      - scripts/import_local_zip.py
    outs:
      - data/03_primary/sequences/local_zip

  run_models:
    cmd: uv run python scripts/run_models.py --store data/03_primary/sequences --models-dir data/06_models --out data/07_model_output
    deps:
      - data/03_primary/sequences
      - data/06_models
      - src/temporal_model_explorer/run_models.py
      - src/temporal_model_explorer/store.py
      - src/temporal_model_explorer/outcomes.py
      - scripts/run_models.py
    outs:
      - data/07_model_output/results.parquet
      - data/07_model_output/details
```

> **External inputs are referenced in `cmd`, not `deps`.** DVC (a per-experiment
> project rooted at this dir) can't take regular deps outside its root, so the
> source `model.zip`s (`prepare_models`) and the annotated zip (`import_local_zip`)
> are read by the command and the stages produce **local DVC-tracked outs**
> (`data/06_models`, `data/03_primary/sequences/local_zip`). `run_models` then
> deps on those local outs → DVC chains `prepare_models` + `import_local_zip` →
> `run_models`, and the models are DVC-tracked.
>
> `import_platform` is intentionally **not** a DVC stage (live API, not
> reproducible). Run it on demand: `uv run python scripts/import_platform.py
> --date-from … --date-to …`; its output lands in
> `data/03_primary/sequences/platform/`, which `run_models` (depending on the
> parent `data/03_primary/sequences`) picks up on its next run.

- [ ] **Step 6: Verify the CLIs run (lint + help)**

Run: `uv run ruff check . && uv run python scripts/run_models.py --help && uv run python scripts/prepare_models.py --help`
Expected: ruff clean; argparse help prints (`--store/--models-dir/--out`; `--out/--params`).

- [ ] **Step 7: Commit**

```bash
git add scripts/prepare_models.py scripts/import_local_zip.py scripts/import_platform.py scripts/run_models.py dvc.yaml
git commit -m "feat(explorer): CLI wrappers + DVC pipeline (prepare_models, import_local_zip, run_models)"
```

---

### Task 9: `app.py` — Streamlit viewer

**Files:**
- Create: `src/temporal_model_explorer/app.py`
- Test: `tests/test_app_helpers.py` (pure helper only; UI not unit-tested)

- [ ] **Step 1: Write failing test for the one pure helper**

```python
# tests/test_app_helpers.py
import pandas as pd

from temporal_model_explorer.app import pivot_decisions


def test_pivot_decisions_one_row_per_sequence():
    df = pd.DataFrame([
        {"key": "zip_1", "label": "smoke", "model": "a", "decision": "keep"},
        {"key": "zip_1", "label": "smoke", "model": "b", "decision": "discard"},
    ])
    wide = pivot_decisions(df)
    row = wide[wide["key"] == "zip_1"].iloc[0]
    assert row["a"] == "keep" and row["b"] == "discard"
    assert row["label"] == "smoke"
```

- [ ] **Step 2: Run → fail**

Run: `uv run pytest tests/test_app_helpers.py -v`
Expected: FAIL (module/function not found).

- [ ] **Step 3: Implement `app.py`** (helper is import-safe; Streamlit code guarded under `main()`)

```python
"""Streamlit viewer over the explorer results (frontend-agnostic data layer).

Reads only data/07_model_output/results.parquet + data/03_primary/sequences/**;
never runs models or fetches. Run with: `streamlit run app.py` (or `make app`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .outcomes import filter_results
from .store import read_meta

RESULTS = Path("data/07_model_output/results.parquet")
STORE = Path("data/03_primary/sequences")


def pivot_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per sequence, one column of decision per model (+ carried meta)."""
    meta_cols = ["key", "source", "label", "label_detail", "camera_name", "organization_name"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    wide = df.pivot_table(
        index="key", columns="model", values="decision", aggfunc="first"
    ).reset_index()
    meta = df[meta_cols].drop_duplicates("key")
    return wide.merge(meta, on="key", how="left")


def _find_seq_dir(key: str) -> Path | None:
    matches = list(STORE.rglob(f"{key}/meta.json"))
    return matches[0].parent if matches else None


def main() -> None:  # pragma: no cover - Streamlit UI
    import streamlit as st

    st.set_page_config(page_title="Temporal Model Explorer", layout="wide")
    st.title("Temporal Model Explorer — keep vs discard")

    if not RESULTS.exists():
        st.warning("No results yet. Run `uv run dvc repro run_models` (or the run_models CLI).")
        return
    df = pd.read_parquet(RESULTS)

    st.sidebar.header("Filters")
    def _opt(col):
        return [None, *sorted(x for x in df[col].dropna().unique())]
    model = st.sidebar.selectbox("model", _opt("model"))
    decision = st.sidebar.selectbox("decision", [None, "keep", "discard"])
    label = st.sidebar.selectbox("label", [None, "smoke", "fp", "unknown"])
    outcome = st.sidebar.selectbox("outcome", _opt("outcome"))
    source = st.sidebar.selectbox("source", _opt("source"))
    camera = st.sidebar.selectbox("camera", _opt("camera_name"))
    org = st.sidebar.selectbox("organization", _opt("organization_name"))
    errors_only = st.sidebar.checkbox("errors only (smoke discarded / fp kept)")

    view = filter_results(
        df, model=model, decision=decision, label=label, outcome=outcome,
        source=source, camera_name=camera, organization_name=org, errors_only=errors_only,
    )
    st.subheader(f"{view['key'].nunique()} sequences")
    st.dataframe(view, use_container_width=True)

    keys = sorted(view["key"].unique())
    if keys:
        key = st.selectbox("Inspect a sequence", keys)
        seq_dir = _find_seq_dir(key)
        if seq_dir:
            meta = read_meta(seq_dir)
            st.write({"label": meta.label, "label_detail": meta.label_detail,
                      "camera": meta.camera_name, "started_at": meta.started_at})
            st.dataframe(view[view["key"] == key][["model", "decision", "trigger_frame_index", "probability"]])
            imgs = [str(seq_dir / f.file) for f in meta.frames]
            st.image(imgs, width=180, caption=[Path(p).name for p in imgs])


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run → pass**

Run: `uv run pytest tests/test_app_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Full suite + lint + format**

Run: `uv run pytest tests/ -v && uv run ruff check . && uv run ruff format --check .`
Expected: all tests pass; ruff clean; format clean.

- [ ] **Step 6: Commit**

```bash
git add src/temporal_model_explorer/app.py tests/test_app_helpers.py
git commit -m "feat(explorer): Streamlit viewer (keep/discard table + drill-down)"
```

---

### Task 10: End-to-end smoke run (real models, manual — not CI)

**Files:** none (verification only). Requires DVC-synced `model.zip`s + the annotated zip.

- [ ] **Step 1: Build the deterministic pipeline via DVC** (prepare_models → import_local_zip → run_models)

Run: `uv run dvc repro`
Expected: `prepare_models` copies both `model.zip`s into (DVC-tracked) `data/06_models/`;
`import_local_zip` populates `data/03_primary/sequences/local_zip/`; `run_models`
writes `data/07_model_output/results.parquet`. `uv run dvc status` then shows
"pipelines up to date". (Equivalent manual run: `scripts/prepare_models.py`, then
`scripts/import_local_zip.py …`, then `scripts/run_models.py --store
data/03_primary/sequences --models-dir data/06_models --out data/07_model_output`.)

- [ ] **Step 2: (optional) Import a day of platform sequences**

Run (creds in env): `uv run python scripts/import_platform.py --date-from 2026-05-19 --date-to 2026-05-19`
Expected: `imported M platform sequences …` into `data/03_primary/sequences/platform/`. Re-run `uv run dvc repro run_models` to include them.

- [ ] **Step 3: Launch the viewer**

Run: `make app`
Expected: Streamlit opens; the table shows per-model KEEP/DISCARD + outcome; filters and drill-down work.

---

## Self-Review

**1. Spec coverage:**
- 3-stage pipeline + Streamlit → Tasks 5–9. ✓
- Lean platform client (no admin; optional `list_organizations`) → Task 4. ✓
- Two importers → common store (`<source>/<key>/{images,meta.json}`) → Tasks 5, 6. ✓
- `meta.json` shape incl. org fields + provenance (platform fills all; zip nulls camera/org) → Tasks 3, 5, 6. ✓
- Platform `is_wildfire → label` best-effort/configurable → `normalize_label` (Task 3) + `params.yaml` (Task 1). ✓
- Model runner via lib, skip-missing, results columns + `outcome` → Task 7. ✓
- Frontend-agnostic results; non-UI logic in `outcomes.py`/`store.py` → Tasks 2, 9. ✓
- Streamlit filters/views + drill-down → Task 9. ✓
- DVC pipeline + **DVC-tracked models** (`prepare_models` → `data/06_models`;
  `import_local_zip`; `run_models` deterministic; platform import standalone;
  per-experiment `dvc init --subdir`) → Tasks 1, 8. ✓
- Testing strategy (mocked API, synthetic zip, fake model, pure helpers) → Tasks 2–9. ✓
- **Not covered (deferred per spec):** org-name enrichment via admin `/organizations/` (the client method exists in Task 4 but no importer wiring) and optional zip→API enrichment — both are spec "optional"; leave for follow-up.

**2. Placeholder scan:** No TBD/TODO; every code/ test step has complete content. ✓

**3. Type consistency:** `SequenceMeta`/`FrameRef` fields (Task 3) are used identically in Tasks 5–7 and 9; results columns (Task 7) match `filter_results`/`pivot_decisions` column names (Tasks 2, 9); `decision`/`outcome` vocab is consistent across Tasks 2, 7, 9. `import_platform` kwargs in tests (Task 6) match the implementation signature. ✓
