# Pyro-annotator Source Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pyro-annotator` as a second sequence source in the Temporal Model Explorer — a human-labeled zip export, enriched with real camera/org/timestamps via the admin API, browsable under a new `source` selector.

**Architecture:** A new importer (`import_pyro_annotator`) walks the extracted zip, derives the ground-truth label from the folder path, copies the frames already present in the zip, and calls the admin platform API per sequence to fill camera/org/timestamps. Sequences are written under a dedicated `pyro-annotator/` store subtree (mirroring the existing org/camera layout) so the source-agnostic `run_models` stage scores them with no pipeline change. The Streamlit app gains a `source` selectbox at the top of its sidebar cascade.

**Tech Stack:** Python 3.11+, `uv`, pytest, pandas/Streamlit. Reuses `store.py` (`FrameRef`, `SequenceMeta`, `write_meta`, `read_meta`) and `platform_api.py` (`get_access_token`, `list_sequence_detections`, `build_camera_index`, `build_org_index`).

**Working directory for all commands:** `experiments/temporal-models/temporal-model-explorer` (within the worktree). All `uv`/`pytest`/`git` commands below assume this cwd. Use `--active` with `uv run` is not needed; ignore the `VIRTUAL_ENV` mismatch warning.

**Spec:** `docs/specs/2026-06-01-pyro-annotator-source-design.md`

---

## File Structure

- **Modify** `src/temporal_model_explorer/store.py` — promote the `slug()` helper here (shared store-layout concern); add nothing else.
- **Modify** `src/temporal_model_explorer/import_platform.py` — use `store.slug` instead of the private `_slug` (no behavior change).
- **Create** `src/temporal_model_explorer/import_pyro_annotator.py` — zip-walk, label parsing, enrichment, store write.
- **Create** `scripts/import_pyro_annotator.py` — thin CLI (env creds → admin token → camera/org index → import).
- **Modify** `src/temporal_model_explorer/app.py` — add `source` selectbox + filter (lines ~515–527).
- **Create** `tests/test_import_pyro_annotator.py` — label parsing, zip walk, importer with stubbed API + real file copy, enrichment fallback.
- **Modify** `README.md` — document the new source and import command.

---

## Task 1: Promote `slug()` to `store.py`

DRY prep: the on-disk slug logic currently lives privately in `import_platform.py`. Move the base `slug` to `store.py` so the new importer reuses it without reaching into another module's privates. Behavior is unchanged, so the existing platform test stays green.

**Files:**
- Modify: `src/temporal_model_explorer/store.py`
- Modify: `src/temporal_model_explorer/import_platform.py:37` (`_slug`)
- Test: `tests/test_store.py` (add one), `tests/test_import_platform.py` (existing, must still pass)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_store.py`:

```python
def test_slug_lowercases_and_replaces_spaces_and_slashes():
    from temporal_model_explorer.store import slug

    assert slug("SDIS 77") == "sdis-77"
    assert slug("Champ/Du Feu") == "champ-du-feu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py::test_slug_lowercases_and_replaces_spaces_and_slashes -v`
Expected: FAIL with `ImportError: cannot import name 'slug'`.

- [ ] **Step 3: Add `slug` to `store.py`**

Add near the top of `src/temporal_model_explorer/store.py` (after the imports, before `FrameRef`):

```python
def slug(value: str) -> str:
    """On-disk-safe slug: lowercased, spaces and slashes become dashes."""
    return value.strip().lower().replace(" ", "-").replace("/", "-")
```

- [ ] **Step 4: Rewire `import_platform.py` to use it**

In `src/temporal_model_explorer/import_platform.py`, update the import line `from .store import FrameRef, SequenceMeta, normalize_label, write_meta` to also import `slug`:

```python
from .store import FrameRef, SequenceMeta, normalize_label, slug, write_meta
```

Delete the private `_slug` definition (lines ~37–38):

```python
def _slug(value: str) -> str:
    return value.strip().lower().replace(" ", "-").replace("/", "-")
```

Update its two call sites inside `_org_slug` and `_camera_slug` from `_slug(...)` to `slug(...)`:

```python
def _org_slug(org_id: int | None, org_index: dict[int, str] | None) -> str:
    """On-disk subdir for an org: its name when known, else org_<id>, else 'unknown'."""
    if org_id is None:
        return "unknown"
    name = (org_index or {}).get(org_id)
    return slug(name) if name else f"org_{org_id}"


def _camera_slug(cam: dict, camera_id: int | None) -> str:
    """On-disk subdir for a camera: name when known, else cam_<id>, else 'unknown'."""
    name = cam.get("name")
    if name:
        return slug(name)
    return f"cam_{camera_id}" if camera_id is not None else "unknown"
```

- [ ] **Step 5: Run tests to verify all pass**

Run: `uv run pytest tests/test_store.py tests/test_import_platform.py -v`
Expected: PASS (new slug test + all existing platform import tests).

- [ ] **Step 6: Commit**

```bash
git add src/temporal_model_explorer/store.py src/temporal_model_explorer/import_platform.py tests/test_store.py
git commit -m "refactor(explorer): promote slug() helper to store"
```

---

## Task 2: Label parsing from folder path

**Files:**
- Create: `src/temporal_model_explorer/import_pyro_annotator.py`
- Test: `tests/test_import_pyro_annotator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_import_pyro_annotator.py`:

```python
import temporal_model_explorer.import_pyro_annotator as ipa


def test_parse_label_smoke_keeps_subtype():
    assert ipa.parse_label("smoke", "wildfire") == ("smoke", "wildfire")
    assert ipa.parse_label("smoke", "industrial") == ("smoke", "industrial")


def test_parse_label_fp_keeps_subtype():
    assert ipa.parse_label("fp", "low_cloud") == ("fp", "low_cloud")


def test_parse_label_unlabeled_is_unknown_with_no_detail():
    assert ipa.parse_label("unlabeled", None) == ("unknown", None)


def test_parse_label_rejects_unknown_class():
    import pytest

    with pytest.raises(ValueError):
        ipa.parse_label("bogus", None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_import_pyro_annotator.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: module ... has no attribute 'parse_label'`.

- [ ] **Step 3: Create the module with `parse_label`**

Create `src/temporal_model_explorer/import_pyro_annotator.py`:

```python
"""Import pyro-annotator sequences (human-labeled zip export) into the store.

The label comes from the folder path (``smoke/<subtype>``, ``fp/<subtype>``,
``unlabeled``). Frames already live in the zip, so images are copied (not
downloaded). Camera/org/timestamps are enriched per sequence via the admin
platform API, so these sequences sit in the same org -> camera navigation as the
alert-API source. Enrichment requires admin creds.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from . import platform_api
from .store import FrameRef, SequenceMeta, slug, write_meta

log = logging.getLogger(__name__)


def parse_label(klass: str, subtype: str | None) -> tuple[str, str | None]:
    """Map a (class, subtype) folder pair to the tri-state label + detail."""
    if klass == "smoke":
        return "smoke", subtype
    if klass == "fp":
        return "fp", subtype
    if klass == "unlabeled":
        return "unknown", None
    raise ValueError(f"unknown class folder: {klass!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_import_pyro_annotator.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/temporal_model_explorer/import_pyro_annotator.py tests/test_import_pyro_annotator.py
git commit -m "feat(explorer): parse pyro-annotator labels from folder path"
```

---

## Task 3: Walk the extracted zip tree

`iter_zip_sequences` yields `(klass, subtype, seq_id, seq_dir)` for every `seq_<id>/` that contains an `images/` dir, skipping macOS junk.

**Files:**
- Modify: `src/temporal_model_explorer/import_pyro_annotator.py`
- Test: `tests/test_import_pyro_annotator.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_import_pyro_annotator.py`:

```python
def _make_seq(root, *parts, det_ids):
    """Create <root>/<parts...>/images/detection_<id>.jpg files."""
    seq_dir = root.joinpath(*parts)
    (seq_dir / "images").mkdir(parents=True)
    for d in det_ids:
        (seq_dir / "images" / f"detection_{d}.jpg").write_bytes(b"img")
    return seq_dir


def test_iter_zip_sequences_finds_seqs_with_class_and_subtype(tmp_path):
    src = tmp_path / "seq_annotation_done_by_label"
    _make_seq(src, "smoke", "wildfire", "seq_40972", det_ids=[1, 2])
    _make_seq(src, "fp", "low_cloud", "seq_40720", det_ids=[5])
    _make_seq(src, "unlabeled", "seq_40438", det_ids=[9])
    # macOS junk must be ignored
    (src / "__MACOSX" / "smoke").mkdir(parents=True)
    (src / "__MACOSX" / "smoke" / "._wildfire").write_bytes(b"junk")

    found = sorted(
        (klass, subtype, seq_id) for klass, subtype, seq_id, _ in ipa.iter_zip_sequences(src)
    )
    assert found == [
        ("fp", "low_cloud", 40720),
        ("smoke", "wildfire", 40972),
        ("unlabeled", None, 40438),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_import_pyro_annotator.py::test_iter_zip_sequences_finds_seqs_with_class_and_subtype -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'iter_zip_sequences'`.

- [ ] **Step 3: Implement `iter_zip_sequences`**

Add to `src/temporal_model_explorer/import_pyro_annotator.py`:

```python
def iter_zip_sequences(
    src: Path,
) -> Iterator[tuple[str, str | None, int, Path]]:
    """Yield (class, subtype, seq_id, seq_dir) for each seq_<id>/ with images/.

    Layout: ``<class>/<subtype>/seq_<id>`` (smoke, fp) or ``<class>/seq_<id>``
    (unlabeled). macOS ``__MACOSX`` entries are skipped.
    """
    for images_dir in sorted(src.rglob("images")):
        seq_dir = images_dir.parent
        rel = seq_dir.relative_to(src).parts
        if "__MACOSX" in rel or not seq_dir.name.startswith("seq_"):
            continue
        klass = rel[0]
        subtype = rel[1] if len(rel) == 3 else None
        seq_id = int(seq_dir.name[len("seq_") :])
        yield klass, subtype, seq_id, seq_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_import_pyro_annotator.py::test_iter_zip_sequences_finds_seqs_with_class_and_subtype -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/temporal_model_explorer/import_pyro_annotator.py tests/test_import_pyro_annotator.py
git commit -m "feat(explorer): walk pyro-annotator zip tree"
```

---

## Task 4: Import + enrich one sequence into the store

`import_pyro_annotator` copies frames from the zip, enriches camera/org/timestamps via the injected detections function + camera/org indexes, orders frames by `created_at`, and writes `meta.json`. The detections function and indexes are injectable (no network in tests), mirroring `test_import_platform.py`.

**Files:**
- Modify: `src/temporal_model_explorer/import_pyro_annotator.py`
- Test: `tests/test_import_pyro_annotator.py`

- [ ] **Step 1: Write the failing test (happy path: enriched)**

Add to `tests/test_import_pyro_annotator.py`:

```python
from temporal_model_explorer.store import read_meta


def test_import_enriches_and_writes_store(tmp_path):
    src = tmp_path / "seq_annotation_done_by_label"
    _make_seq(src, "smoke", "wildfire", "seq_40972", det_ids=[2, 1])  # unsorted on disk
    out = tmp_path / "store"

    def fake_detections(ep, tok, sid, limit, desc):
        assert sid == 40972
        return [
            {"id": 1, "camera_id": 65, "created_at": "2026-05-09T15:03:49"},
            {"id": 2, "camera_id": 65, "created_at": "2026-05-09T15:04:50"},
        ]

    camera_index = {65: {"id": 65, "name": "nemours-02", "organization_id": 7}}
    n = ipa.import_pyro_annotator(
        src,
        out,
        "https://x",
        "admintok",
        camera_index=camera_index,
        org_index={7: "sdis-77"},
        list_detections=fake_detections,
    )

    assert n == 1
    seq_dir = out / "pyro-annotator" / "sdis-77" / "nemours-02" / "seq_40972"
    meta = read_meta(seq_dir)
    assert meta.key == "pyro_annotator_40972"
    assert meta.source == "pyro-annotator"
    assert meta.label == "smoke" and meta.label_detail == "wildfire"
    assert meta.label_source == "pyro_annotator_folder"
    assert meta.camera_id == 65 and meta.camera_name == "nemours-02"
    assert meta.organization_id == 7 and meta.organization_name == "sdis-77"
    # frames ordered by created_at ascending -> detection 1 then 2
    assert [f.detection_id for f in meta.frames] == [1, 2]
    assert meta.frames[0].created_at == "2026-05-09T15:03:49"
    assert meta.started_at == "2026-05-09T15:03:49"
    # image copied from the zip
    assert (seq_dir / "images" / "detection_1.jpg").read_bytes() == b"img"
```

- [ ] **Step 2: Write the failing test (fallback: no enrichment)**

Add to `tests/test_import_pyro_annotator.py`:

```python
def test_import_falls_back_to_unknown_when_no_detections(tmp_path):
    src = tmp_path / "seq_annotation_done_by_label"
    _make_seq(src, "fp", "low_cloud", "seq_99999", det_ids=[3])
    out = tmp_path / "store"

    n = ipa.import_pyro_annotator(
        src,
        out,
        "https://x",
        "admintok",
        camera_index={},
        org_index={},
        list_detections=lambda ep, tok, sid, limit, desc: [],
    )

    assert n == 1
    seq_dir = out / "pyro-annotator" / "unknown" / "unknown" / "seq_99999"
    meta = read_meta(seq_dir)
    assert meta.label == "fp" and meta.label_detail == "low_cloud"
    assert meta.camera_name == "unknown" and meta.organization_name == "unknown"
    assert meta.camera_id is None and meta.organization_id is None
    assert meta.frames[0].created_at is None
    assert meta.started_at is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_import_pyro_annotator.py -k import -v`
Expected: FAIL with `AttributeError: ... has no attribute 'import_pyro_annotator'`.

- [ ] **Step 4: Implement `_import_one` and `import_pyro_annotator`**

Add to `src/temporal_model_explorer/import_pyro_annotator.py`:

```python
def _import_one(
    api_endpoint: str,
    token: str,
    out: Path,
    klass: str,
    subtype: str | None,
    seq_id: int,
    seq_dir: Path,
    camera_index: dict,
    org_index: dict[int, str] | None,
    detections_limit: int,
    list_detections,
) -> int:
    label, label_detail = parse_label(klass, subtype)

    # Enrich: detection timestamps + camera id (constant per sequence).
    ts_by_id: dict[int, str | None] = {}
    camera_id: int | None = None
    try:
        dets = list_detections(
            api_endpoint, token, seq_id, limit=detections_limit, desc=False
        )
        for d in dets:
            ts_by_id[d["id"]] = d.get("created_at")
        if dets:
            camera_id = dets[0].get("camera_id")
    except Exception as exc:  # noqa: BLE001 - enrichment is best-effort; log + fall back
        log.warning("enrichment failed for seq %s: %s", seq_id, exc)

    cam = camera_index.get(camera_id, {}) if camera_id is not None else {}
    org_id = cam.get("organization_id")
    camera_name = cam.get("name") or "unknown"
    org_name = (org_index or {}).get(org_id) or "unknown"

    seq_out = out / "pyro-annotator" / slug(org_name) / slug(camera_name) / f"seq_{seq_id}"
    (seq_out / "images").mkdir(parents=True, exist_ok=True)

    frames: list[FrameRef] = []
    for img in sorted((seq_dir / "images").glob("*.jpg")):
        det_id = int(img.stem.split("_")[-1])
        shutil.copyfile(img, seq_out / "images" / img.name)
        frames.append(
            FrameRef(
                file=f"images/{img.name}",
                detection_id=det_id,
                created_at=ts_by_id.get(det_id),
            )
        )
    # Time axis: known timestamps first (ascending), unknowns last by detection id.
    frames.sort(key=lambda f: (f.created_at is None, f.created_at or "", f.detection_id or 0))
    started_at = next((f.created_at for f in frames if f.created_at), None)

    write_meta(
        seq_out,
        SequenceMeta(
            key=f"pyro_annotator_{seq_id}",
            sequence_id=str(seq_id),
            source="pyro-annotator",
            label=label,
            label_detail=label_detail,
            label_source="pyro_annotator_folder",
            frames=frames,
            camera_id=camera_id,
            camera_name=camera_name,
            organization_id=org_id,
            organization_name=org_name,
            started_at=started_at,
        ),
    )
    return 1


def import_pyro_annotator(
    src: Path,
    out: Path,
    api_endpoint: str,
    token: str,
    *,
    detections_limit: int = 200,
    camera_index: dict | None = None,
    org_index: dict[int, str] | None = None,
    list_detections=platform_api.list_sequence_detections,
) -> int:
    """Import every sequence under ``src`` into ``out``. Returns #sequences."""
    camera_index = camera_index or {}
    count = 0
    for klass, subtype, seq_id, seq_dir in iter_zip_sequences(src):
        count += _import_one(
            api_endpoint,
            token,
            out,
            klass,
            subtype,
            seq_id,
            seq_dir,
            camera_index,
            org_index,
            detections_limit,
            list_detections,
        )
    return count
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_import_pyro_annotator.py -v`
Expected: PASS (all tasks 2–4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/temporal_model_explorer/import_pyro_annotator.py tests/test_import_pyro_annotator.py
git commit -m "feat(explorer): import + enrich pyro-annotator sequences into the store"
```

---

## Task 5: CLI script

Thin CLI mirroring `scripts/import_platform.py`: read endpoint + **admin** creds from env, build camera/org indexes with the admin token, run the import.

**Files:**
- Create: `scripts/import_pyro_annotator.py`

- [ ] **Step 1: Create the CLI**

Create `scripts/import_pyro_annotator.py`:

```python
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
```

- [ ] **Step 2: Verify it parses + errors clearly without creds**

Run: `uv run python scripts/import_pyro_annotator.py --help`
Expected: usage text listing `--src` and `--out`.

Run (no creds): `env -u PLATFORM_API_ENDPOINT uv run python scripts/import_pyro_annotator.py --src /tmp/nope`
Expected: a `KeyError: 'PLATFORM_API_ENDPOINT'` traceback (env var missing) — confirms it reads the endpoint from env. (With the endpoint set but admin creds missing, it exits with the admin-creds message.)

- [ ] **Step 3: Commit**

```bash
git add scripts/import_pyro_annotator.py
git commit -m "feat(explorer): add import_pyro_annotator CLI"
```

---

## Task 6: `source` selector in the app

Add a `source` selectbox at the top of the sidebar cascade and apply it as a filter, mirroring the existing `organization`/`camera` pattern. This is inline in `main()`, consistent with the existing (untested) org/camera filters — no dedicated app test.

**Files:**
- Modify: `src/temporal_model_explorer/app.py:515-527`

- [ ] **Step 1: Add the selectbox**

In `src/temporal_model_explorer/app.py`, replace this block (currently lines ~515–521):

```python
    st.sidebar.header("Select")
    orgs = sorted(x for x in df["organization_name"].dropna().unique())
    org = st.sidebar.selectbox("organization", orgs, key="org") if orgs else None
    org_df = df[df["organization_name"] == org] if org else df
    cameras = sorted(x for x in org_df["camera_name"].dropna().unique())
    camera = st.sidebar.selectbox("camera", cameras, key="camera") if cameras else None
    model = st.sidebar.selectbox("model", models, key="model")
```

with (adds the `source` selectbox + cascades the org list from the source-filtered frame):

```python
    st.sidebar.header("Select")
    sources = sorted(x for x in df["source"].dropna().unique())
    source = st.sidebar.selectbox("source", sources, key="source") if sources else None
    src_df = df[df["source"] == source] if source else df
    orgs = sorted(x for x in src_df["organization_name"].dropna().unique())
    org = st.sidebar.selectbox("organization", orgs, key="org") if orgs else None
    org_df = src_df[src_df["organization_name"] == org] if org else src_df
    cameras = sorted(x for x in org_df["camera_name"].dropna().unique())
    camera = st.sidebar.selectbox("camera", cameras, key="camera") if cameras else None
    model = st.sidebar.selectbox("model", models, key="model")
```

- [ ] **Step 2: Apply the source filter**

In the same function, replace this block (currently lines ~523–527):

```python
    view = df[df["model"] == model]
    if org:
        view = view[view["organization_name"] == org]
    if camera:
        view = view[view["camera_name"] == camera]
```

with:

```python
    view = df[df["model"] == model]
    if source:
        view = view[view["source"] == source]
    if org:
        view = view[view["organization_name"] == org]
    if camera:
        view = view[view["camera_name"] == camera]
```

- [ ] **Step 3: Verify the app imports cleanly + lint passes**

Run: `uv run python -c "import temporal_model_explorer.app"`
Expected: no output, exit 0 (module imports without error).

Run: `make lint`
Expected: ruff reports no errors.

- [ ] **Step 4: Commit**

```bash
git add src/temporal_model_explorer/app.py
git commit -m "feat(explorer): add source selector to the sidebar"
```

---

## Task 7: Document the new source

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add an import section**

In `README.md`, after the "Importing fresh sequences (optional)" section, add:

````markdown
## Importing the pyro-annotator export

The pyro-annotator zip is a human-labeled export (smoke / fp / unlabeled, with
subtype). Unzip it under `data/01_raw/`, then import — camera, organization, and
timestamps are enriched from the platform API. **Admin creds are required**: the
regular login is org-scoped and returns 403 for these sequences.

```bash
unzip data/01_raw/seq_annotation_done_by_label.zip -d data/01_raw/

export PLATFORM_API_ENDPOINT=https://...
export PLATFORM_ADMIN_LOGIN=...
export PLATFORM_ADMIN_PASSWORD=...

uv run python scripts/import_pyro_annotator.py \
    --src data/01_raw/seq_annotation_done_by_label
uv run dvc repro run_models     # score the newly imported sequences
```

These land under `data/03_primary/sequences/pyro-annotator/<org>/<camera>/` and
appear in the app under the **source** selector as `pyro-annotator`.
````

In the CLIs table, add a row:

```markdown
| `scripts/import_pyro_annotator.py` | Import the pyro-annotator zip export (label from folder, camera/org/timestamps enriched via admin API) | `--src` (required), `--out` |
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(explorer): document the pyro-annotator import"
```

---

## Task 8: Full suite + final verification

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (35 pre-existing + new pyro-annotator/slug tests).

- [ ] **Step 2: Lint + format check**

Run: `make lint`
Expected: no ruff errors.

- [ ] **Step 3: (Optional, needs creds + data) smoke-test a real import**

Only if admin creds and the unzipped data are present in this checkout:

Run:
```bash
unzip -q data/01_raw/seq_annotation_done_by_label.zip -d data/01_raw/
uv run python scripts/import_pyro_annotator.py --src data/01_raw/seq_annotation_done_by_label
```
Expected: `imported 332 pyro-annotator sequences into data/03_primary/sequences`, and `data/03_primary/sequences/pyro-annotator/sdis-77/nemours-02/seq_40972/meta.json` exists with `"source": "pyro-annotator"`.

If creds/data are unavailable here, note it and skip — the stubbed tests cover the logic.

- [ ] **Step 4: DVC-track the new store subtree (only after a real import)**

```bash
uv run dvc add data/03_primary/sequences/pyro-annotator
git add data/03_primary/sequences/pyro-annotator.dvc data/03_primary/sequences/.gitignore
git commit -m "data(explorer): track pyro-annotator sequences"
```

---

## Self-Review Notes

- **Spec coverage:** source selector (Task 6), importer + enrichment (Tasks 2–4), CLI/admin creds (Task 5), dedicated `pyro-annotator/` store tree + zero pipeline change (Task 4 paths; `run_models` unchanged), label-from-folder (Task 2), out-of-scope bbox/predictor files (never read — confirmed by `iter_zip_sequences` only touching `images/`), error handling fallback (Task 4 fallback test), docs (Task 7), DVC tracking (Task 8). All covered.
- **`source` stored value** is `"pyro-annotator"`; existing alert-API stays `"platform"`. The dropdown shows the raw stored values (approved).
- **Type consistency:** `parse_label` → `(str, str|None)` used identically in `_import_one`; `iter_zip_sequences` yields `(klass, subtype, seq_id, seq_dir)` consumed positionally in `import_pyro_annotator`; `list_detections` signature `(ep, tok, sid, limit, desc)` matches `platform_api.list_sequence_detections` and both test stubs.
