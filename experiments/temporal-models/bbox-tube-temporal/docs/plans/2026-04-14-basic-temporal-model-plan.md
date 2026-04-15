> Renamed 2026-04-15: smokeynet-adapted → bbox-tube-temporal. Old paths in this doc reflect the design-time state.

# Basic Temporal Model Implementation Plan

> **Historical note:** This plan was written when the original SmokeyNetAdapted
> stack (`model.py`, `backbone.py`, `net.py`, `heads.py`, `detector.py`,
> `spatial_attention.py`, `temporal_fusion.py`, `training.py`, `package.py`)
> still coexisted with the new classifier. That stack was deleted in `ad008c6`.
> References to those files below describe the state at plan time.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate two basic vision-based temporal smoke
classifiers (mean-pool baseline + GRU) on cropped tube patches, end-to-end
through DVC.

**Architecture:** New DVC pipeline `build_model_input → train_mean_pool +
train_gru → evaluate_*`. Patches saved as PNG folders so they're directly
browsable. Backbone is a frozen `timm` model (default `resnet18`) shared
between archs; only the temporal head differs.

**Tech Stack:** PyTorch, PyTorch Lightning, timm, torchvision, DVC, pytest.

**Spec:** `docs/specs/2026-04-14-basic-temporal-model-design.md`

**Working directory for all commands:** `experiments/temporal-models/bbox-tube-temporal/`

---

## Task 1: Add `timm` dependency

**Files:**
- Modify: `pyproject.toml` (dependencies list)

- [ ] **Step 1: Add timm to pyproject.toml dependencies**

Open `pyproject.toml` and add `"timm>=1.0",` after the `tensorboard` line so the dependency block ends with:

```toml
    "tensorboard>=2.16",
    "timm>=1.0",
    "torch>=2.2",
```

- [ ] **Step 2: Sync the lockfile**

Run: `uv sync`
Expected: timm is downloaded and installed; uv.lock is updated.

- [ ] **Step 3: Verify timm imports**

Run: `uv run python -c "import timm; m = timm.create_model('resnet18', pretrained=False, num_classes=0, global_pool='avg'); print(m.num_features)"`
Expected: prints `512`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(bbox-tube-temporal): add timm dependency for pretrained backbones"
```

---

## Task 2: Crop logic — `model_input.py` core functions

**Files:**
- Create: `src/bbox_tube_temporal/model_input.py`
- Test: `tests/test_model_input.py`

We build the pure crop math first (no I/O, no parallelism). Five small functions, each with its own TDD cycle.

### Function 1: `expand_bbox` — apply context factor

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_input.py`:

```python
"""Tests for model_input crop logic."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bbox_tube_temporal.model_input import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
    process_tube,
    save_patch,
)


def test_expand_bbox_scales_w_and_h_by_factor():
    cx, cy, w, h = expand_bbox(0.5, 0.5, 0.1, 0.2, factor=1.5)
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert w == pytest.approx(0.15)
    assert h == pytest.approx(0.30)


def test_expand_bbox_factor_one_is_identity():
    assert expand_bbox(0.3, 0.7, 0.04, 0.06, factor=1.0) == (0.3, 0.7, 0.04, 0.06)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model_input.py::test_expand_bbox_scales_w_and_h_by_factor -v`
Expected: FAIL with `ImportError: cannot import name 'expand_bbox'`.

- [ ] **Step 3: Implement `expand_bbox`**

Create `src/bbox_tube_temporal/model_input.py`:

```python
"""Crop tube bboxes from raw frames and save 224x224 PNG patches.

Pure functions for bbox math + crop/save; orchestration lives in
``scripts/build_model_input.py``.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def expand_bbox(
    cx: float, cy: float, w: float, h: float, factor: float
) -> tuple[float, float, float, float]:
    return cx, cy, w * factor, h * factor
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_model_input.py -v`
Expected: both expand_bbox tests PASS.

### Function 2: `norm_bbox_to_pixel_square` — convert + square + clip-with-pad

- [ ] **Step 5: Add tests**

Append to `tests/test_model_input.py`:

```python
def test_norm_bbox_to_pixel_square_returns_square_inside_bounds():
    # Fully inside: bbox at center of 1000x800 image, 0.1x0.2 normalized
    box = norm_bbox_to_pixel_square(0.5, 0.5, 0.1, 0.2, img_w=1000, img_h=800)
    x0, y0, x1, y1 = box
    side = x1 - x0
    assert side == y1 - y0
    # 0.2 * 800 = 160 pixels; 0.1 * 1000 = 100 pixels; max → 160
    assert side == 160
    assert (x0 + x1) // 2 == 500
    assert (y0 + y1) // 2 == 400


def test_norm_bbox_to_pixel_square_clips_at_left_edge():
    # bbox center near left edge; expected box would extend past x=0
    box = norm_bbox_to_pixel_square(0.02, 0.5, 0.1, 0.1, img_w=1000, img_h=1000)
    x0, y0, x1, y1 = box
    assert x0 >= 0
    assert y0 >= 0
    assert x1 <= 1000
    assert y1 <= 1000


def test_norm_bbox_to_pixel_square_returns_integer_coords():
    box = norm_bbox_to_pixel_square(0.333, 0.777, 0.111, 0.222, img_w=1280, img_h=720)
    for v in box:
        assert isinstance(v, int)
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_model_input.py -v -k norm_bbox`
Expected: FAIL with `ImportError`.

- [ ] **Step 7: Implement `norm_bbox_to_pixel_square`**

Append to `src/bbox_tube_temporal/model_input.py`:

```python
def norm_bbox_to_pixel_square(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Convert normalized bbox to a square pixel box, clipped to image.

    Returns ``(x0, y0, x1, y1)`` in pixel coords. The box is squared by
    enlarging the smaller side to match the larger (centered on the
    bbox center), then clipped to image bounds. If clipping breaks
    squareness, the caller pads the resulting crop in ``crop_and_resize``.
    """
    side_px = max(w * img_w, h * img_h)
    half = side_px / 2.0
    cx_px = cx * img_w
    cy_px = cy * img_h
    x0 = int(round(cx_px - half))
    y0 = int(round(cy_px - half))
    x1 = int(round(cx_px + half))
    y1 = int(round(cy_px + half))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img_w, x1)
    y1 = min(img_h, y1)
    return x0, y0, x1, y1
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_model_input.py -v`
Expected: all PASS so far.

### Function 3: `crop_and_resize` — extract patch + pad-to-square + resize

- [ ] **Step 9: Add tests**

Append to `tests/test_model_input.py`:

```python
def _solid_image(w: int, h: int, color: tuple[int, int, int]) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    return arr


def test_crop_and_resize_returns_uint8_rgb_at_target_size():
    img = _solid_image(800, 600, (255, 0, 0))
    patch = crop_and_resize(img, (100, 100, 300, 300), patch_size=224)
    assert patch.shape == (224, 224, 3)
    assert patch.dtype == np.uint8


def test_crop_and_resize_pads_non_square_crop_with_zeros():
    img = _solid_image(800, 600, (255, 255, 255))
    # Force non-square crop: 200 wide, 100 tall (clipped at top edge).
    patch = crop_and_resize(img, (300, 0, 500, 100), patch_size=224)
    assert patch.shape == (224, 224, 3)
    # Bottom half should contain zero-padding (after centering the 200x100 in
    # a 200x200 square, then resizing). Top quarter is white, bottom quarter
    # is white, middle band is white, but rows above/below the centered band
    # (top ~56 rows and bottom ~56 rows in the 224 patch) are zeros.
    assert patch[0, 100, :].sum() == 0
    assert patch[223, 100, :].sum() == 0
```

- [ ] **Step 10: Run tests**

Run: `uv run pytest tests/test_model_input.py -v -k crop_and_resize`
Expected: FAIL with `ImportError`.

- [ ] **Step 11: Implement `crop_and_resize`**

Append to `src/bbox_tube_temporal/model_input.py`:

```python
def crop_and_resize(
    image: np.ndarray, box: tuple[int, int, int, int], patch_size: int
) -> np.ndarray:
    """Extract a patch, pad to square if needed, resize to patch_size.

    ``image`` is HxWx3 uint8 RGB. ``box`` is ``(x0, y0, x1, y1)``.
    """
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1, :]
    h, w, _ = crop.shape
    side = max(h, w)
    if h != w:
        square = np.zeros((side, side, 3), dtype=np.uint8)
        y_off = (side - h) // 2
        x_off = (side - w) // 2
        square[y_off : y_off + h, x_off : x_off + w, :] = crop
        crop = square
    pil = Image.fromarray(crop)
    pil = pil.resize((patch_size, patch_size), Image.BILINEAR)
    return np.array(pil)
```

- [ ] **Step 12: Run tests**

Run: `uv run pytest tests/test_model_input.py -v`
Expected: all PASS.

### Function 4: `save_patch` — write PNG

- [ ] **Step 13: Add test**

Append to `tests/test_model_input.py`:

```python
def test_save_patch_writes_png_at_target_size(tmp_path):
    img = _solid_image(224, 224, (10, 20, 30))
    out_path = tmp_path / "frame_00.png"
    save_patch(img, out_path)
    assert out_path.is_file()
    loaded = np.array(Image.open(out_path))
    assert loaded.shape == (224, 224, 3)
    # PNG round-trip preserves pixels exactly
    assert loaded[0, 0, 0] == 10
    assert loaded[0, 0, 1] == 20
    assert loaded[0, 0, 2] == 30
```

- [ ] **Step 14: Run, expect fail, then implement**

Run: `uv run pytest tests/test_model_input.py -v -k save_patch`
Expected: FAIL with `ImportError`.

Append to `src/bbox_tube_temporal/model_input.py`:

```python
def save_patch(patch: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch).save(path, format="PNG", optimize=True)
```

- [ ] **Step 15: Run tests**

Run: `uv run pytest tests/test_model_input.py -v`
Expected: all PASS.

### Function 5: `process_tube` — end-to-end per-tube orchestration

- [ ] **Step 16: Add test**

Append to `tests/test_model_input.py`:

```python
def _write_jpg(path: Path, color: tuple[int, int, int], w: int = 1280, h: int = 720):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_solid_image(w, h, color)).save(path, format="JPEG", quality=95)


def _write_tube_record(path: Path, sequence_id: str, label: str, frame_ids: list[str]):
    record = {
        "sequence_id": sequence_id,
        "split": "train",
        "label": label,
        "source": "gt",
        "num_frames": len(frame_ids),
        "tube": {
            "start_frame": 0,
            "end_frame": len(frame_ids) - 1,
            "entries": [
                {
                    "frame_idx": i,
                    "frame_id": fid,
                    "bbox": [0.5, 0.5, 0.05, 0.05],
                    "is_gap": False,
                    "confidence": 0.9,
                }
                for i, fid in enumerate(frame_ids)
            ],
        },
    }
    path.write_text(json.dumps(record))
    return record


def test_process_tube_writes_patches_and_meta(tmp_path):
    # Build a tiny fake sequence with 3 frames under wildfire/<seq>/images/
    seq_id = "site_999_2023-05-23T17-18-31"
    seq_root = tmp_path / "raw" / "wildfire" / seq_id / "images"
    frame_ids = [f"{seq_id}_f{i}" for i in range(3)]
    for fid in frame_ids:
        _write_jpg(seq_root / f"{fid}.jpg", (255, 128, 64))

    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tube_record(tube_path, seq_id, "smoke", frame_ids)

    out_dir = tmp_path / "out"
    process_tube(
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
    )

    seq_out = out_dir / seq_id
    assert (seq_out / "frame_00.png").is_file()
    assert (seq_out / "frame_01.png").is_file()
    assert (seq_out / "frame_02.png").is_file()
    meta = json.loads((seq_out / "meta.json").read_text())
    assert meta["sequence_id"] == seq_id
    assert meta["label"] == "smoke"
    assert meta["label_int"] == 1
    assert meta["num_frames"] == 3
    assert meta["context_factor"] == 1.5
    assert meta["patch_size"] == 224
    assert len(meta["frames"]) == 3
    assert meta["frames"][0]["filename"] == "frame_00.png"
    assert meta["frames"][0]["is_gap"] is False


def test_process_tube_uses_filename_from_raw_directory(tmp_path):
    # FP labels live under fp/<seq>/images/
    seq_id = "site_999_2023-06-01T10-00-00"
    seq_root = tmp_path / "raw" / "fp" / seq_id / "images"
    frame_ids = [f"{seq_id}_f{i}" for i in range(2)]
    for fid in frame_ids:
        _write_jpg(seq_root / f"{fid}.jpg", (10, 20, 30))

    tube_path = tmp_path / "tubes" / f"{seq_id}.json"
    tube_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tube_record(tube_path, seq_id, "fp", frame_ids)

    out_dir = tmp_path / "out"
    process_tube(
        tube_path=tube_path,
        raw_dir=tmp_path / "raw",
        out_dir=out_dir,
        context_factor=1.5,
        patch_size=224,
    )
    meta = json.loads((out_dir / seq_id / "meta.json").read_text())
    assert meta["label"] == "fp"
    assert meta["label_int"] == 0
```

- [ ] **Step 17: Run, expect fail**

Run: `uv run pytest tests/test_model_input.py -v -k process_tube`
Expected: FAIL with `ImportError`.

- [ ] **Step 18: Implement `process_tube`**

Append to `src/bbox_tube_temporal/model_input.py`:

```python
import json

from .data import find_sequence_dir

LABEL_TO_INT = {"fp": 0, "smoke": 1}


def process_tube(
    tube_path: Path,
    raw_dir: Path,
    out_dir: Path,
    context_factor: float,
    patch_size: int,
) -> None:
    """Crop one tube's frames to PNGs and write meta.json."""
    record = json.loads(tube_path.read_text())
    sequence_id = record["sequence_id"]
    label = record["label"]
    seq_dir = find_sequence_dir(raw_dir, sequence_id)
    if seq_dir is None:
        raise FileNotFoundError(f"raw sequence dir not found for {sequence_id}")

    images_dir = seq_dir / "images"
    seq_out = out_dir / sequence_id
    seq_out.mkdir(parents=True, exist_ok=True)

    frame_meta: list[dict] = []
    for entry in record["tube"]["entries"]:
        frame_id = entry["frame_id"]
        frame_idx = entry["frame_idx"]
        bbox = entry["bbox"]
        is_gap = entry["is_gap"]

        img_path = images_dir / f"{frame_id}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w, _ = image.shape

        cx, cy, w, h = expand_bbox(bbox[0], bbox[1], bbox[2], bbox[3], context_factor)
        crop_box = norm_bbox_to_pixel_square(cx, cy, w, h, img_w, img_h)
        patch = crop_and_resize(image, crop_box, patch_size)

        filename = f"frame_{frame_idx:02d}.png"
        save_patch(patch, seq_out / filename)

        frame_meta.append(
            {
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "is_gap": is_gap,
                "orig_bbox": list(bbox),
                "crop_bbox_pixels": list(crop_box),
                "filename": filename,
            }
        )

    meta = {
        "sequence_id": sequence_id,
        "split": record["split"],
        "label": label,
        "label_int": LABEL_TO_INT[label],
        "num_frames": record["num_frames"],
        "context_factor": context_factor,
        "patch_size": patch_size,
        "frames": frame_meta,
    }
    (seq_out / "meta.json").write_text(json.dumps(meta, indent=2))
```

Note: `import json` and `from .data import find_sequence_dir` go to the **top of the file** (per repo convention — no in-function imports). Move the existing top-of-file imports + add these.

The final top-of-file import block should read:

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import find_sequence_dir

LABEL_TO_INT = {"fp": 0, "smoke": 1}
```

And remove the duplicate `import json` further down.

- [ ] **Step 19: Run all model_input tests**

Run: `uv run pytest tests/test_model_input.py -v`
Expected: all PASS.

- [ ] **Step 20: Run lint + format**

Run: `make lint && make format`
Expected: clean.

- [ ] **Step 21: Commit**

```bash
git add src/bbox_tube_temporal/model_input.py tests/test_model_input.py
git commit -m "feat(bbox-tube-temporal): crop logic for model input patches"
```

---

## Task 3: `build_model_input` script + DVC stage

**Files:**
- Create: `scripts/build_model_input.py`
- Modify: `dvc.yaml` (add `build_model_input` stage)
- Modify: `params.yaml` (add `model_input:` section)

- [ ] **Step 1: Add params section**

Open `params.yaml` and append after the existing `build_tubes:` block:

```yaml
model_input:
  context_factor: 1.5
  patch_size: 224
```

- [ ] **Step 2: Write the script**

Create `scripts/build_model_input.py`:

```python
"""Crop tube bboxes from raw frames into 224x224 PNG patches per tube.

For each tube JSON in ``--tubes-dir``, look up the source sequence
under ``--raw-dir`` and write a directory of cropped PNG frames + a
``meta.json`` to ``--output-dir/<sequence_id>/``. Also writes a
``_index.json`` listing all surviving tubes.

Wipes ``--output-dir`` at the start so stale outputs don't linger.
"""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import os

from bbox_tube_temporal.model_input import LABEL_TO_INT, process_tube


def _process_one(
    tube_path: Path,
    raw_dir: Path,
    out_dir: Path,
    context_factor: float,
    patch_size: int,
) -> tuple[str | None, str, str | None]:
    """Worker: returns (sequence_id, label, error_or_none)."""
    try:
        record = json.loads(tube_path.read_text())
        sequence_id = record["sequence_id"]
        label = record["label"]
        process_tube(
            tube_path=tube_path,
            raw_dir=raw_dir,
            out_dir=out_dir,
            context_factor=context_factor,
            patch_size=patch_size,
        )
        return sequence_id, label, None
    except Exception as exc:  # noqa: BLE001
        return None, "", f"{tube_path.name}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tubes-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-factor", type=float, required=True)
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = os.cpu_count()).",
    )
    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tube_paths = sorted(
        p for p in args.tubes_dir.glob("*.json") if p.name != "_summary.json"
    )

    index: list[dict] = []
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _process_one,
                p,
                args.raw_dir,
                args.output_dir,
                args.context_factor,
                args.patch_size,
            )
            for p in tube_paths
        ]
        for fut in as_completed(futures):
            sequence_id, label, err = fut.result()
            if err is not None:
                errors.append(err)
                continue
            assert sequence_id is not None
            num_frames = len(
                json.loads((args.output_dir / sequence_id / "meta.json").read_text())[
                    "frames"
                ]
            )
            index.append(
                {
                    "sequence_id": sequence_id,
                    "label_int": LABEL_TO_INT[label],
                    "num_frames": num_frames,
                }
            )

    index.sort(key=lambda r: r["sequence_id"])
    (args.output_dir / "_index.json").write_text(json.dumps(index, indent=2))

    split = args.tubes_dir.name
    print(
        f"[{split}] wrote {len(index)}/{len(tube_paths)} tubes "
        f"with {workers} workers (errors={len(errors)})"
    )
    for e in errors[:5]:
        print(f"  error: {e}")
    if len(errors) > 5:
        print(f"  ... and {len(errors) - 5} more")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add DVC stage**

Open `dvc.yaml` and add this stage **after** the `build_tubes:` stage (and before the commented-out section):

```yaml
  build_model_input:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/build_model_input.py
        --tubes-dir data/03_primary/tubes/${item}
        --raw-dir data/01_raw/datasets/${item}
        --output-dir data/05_model_input/${item}
        --context-factor ${model_input.context_factor}
        --patch-size ${model_input.patch_size}
      deps:
        - scripts/build_model_input.py
        - src/bbox_tube_temporal/model_input.py
        - data/03_primary/tubes/${item}
        - data/01_raw/datasets/${item}
      params:
        - model_input
      outs:
        - data/05_model_input/${item}
```

- [ ] **Step 4: Run the stage on val (smaller, faster)**

Run: `uv run dvc repro build_model_input@val`
Expected: completes; prints `[val] wrote 284/284 tubes ... (errors=0)`.

- [ ] **Step 5: Spot-check output structure**

Run: `ls data/05_model_input/val | head -5 && ls data/05_model_input/val/$(ls data/05_model_input/val | grep -v _index | head -1)`
Expected: a list of sequence_id directories, then ~20 `frame_NN.png` files + `meta.json`.

- [ ] **Step 6: Eyeball one patch**

Run: `uv run python -c "from PIL import Image; import json, sys; idx=json.loads(open('data/05_model_input/val/_index.json').read()); seq=idx[0]['sequence_id']; im=Image.open(f'data/05_model_input/val/{seq}/frame_00.png'); print(im.size, im.mode)"`
Expected: `(224, 224) RGB`.

- [ ] **Step 7: Run the train split too**

Run: `uv run dvc repro build_model_input@train`
Expected: `[train] wrote 2842/2842 tubes ... (errors=0)`.

- [ ] **Step 8: Lint + format**

Run: `make lint && make format`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add scripts/build_model_input.py dvc.yaml dvc.lock params.yaml
git commit -m "feat(bbox-tube-temporal): build_model_input dvc stage producing tube patches"
```

---

## Task 4: Rewrite `Dataset` for tube patches

**Files:**
- Modify (full rewrite): `src/bbox_tube_temporal/dataset.py`
- Modify (full rewrite): `tests/test_dataset.py`

- [ ] **Step 1: Replace test file with new tests**

Overwrite `tests/test_dataset.py` with:

```python
"""Tests for TubePatchDataset."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from bbox_tube_temporal.dataset import TubePatchDataset


def _make_split(tmp_path: Path, samples: list[tuple[str, int, int]]) -> Path:
    """Create a fake split dir. samples = [(seq_id, label_int, num_frames), ...]"""
    split = tmp_path / "split"
    split.mkdir()
    index = []
    for seq_id, label_int, num_frames in samples:
        seq_dir = split / seq_id
        seq_dir.mkdir()
        for i in range(num_frames):
            arr = np.full((224, 224, 3), 50 + i, dtype=np.uint8)
            Image.fromarray(arr).save(seq_dir / f"frame_{i:02d}.png")
        meta = {
            "sequence_id": seq_id,
            "split": "train",
            "label": "smoke" if label_int == 1 else "fp",
            "label_int": label_int,
            "num_frames": num_frames,
            "context_factor": 1.5,
            "patch_size": 224,
            "frames": [
                {
                    "frame_idx": i,
                    "frame_id": f"f{i}",
                    "is_gap": False,
                    "orig_bbox": [0.5, 0.5, 0.05, 0.05],
                    "crop_bbox_pixels": [0, 0, 100, 100],
                    "filename": f"frame_{i:02d}.png",
                }
                for i in range(num_frames)
            ],
        }
        (seq_dir / "meta.json").write_text(json.dumps(meta))
        index.append({"sequence_id": seq_id, "label_int": label_int, "num_frames": num_frames})
    (split / "_index.json").write_text(json.dumps(index))
    return split


def test_dataset_length_matches_index(tmp_path):
    split = _make_split(tmp_path, [("a", 1, 5), ("b", 0, 7)])
    ds = TubePatchDataset(split, max_frames=20)
    assert len(ds) == 2


def test_dataset_pads_short_sequences_left_aligned(tmp_path):
    split = _make_split(tmp_path, [("a", 1, 5)])
    ds = TubePatchDataset(split, max_frames=20)
    sample = ds[0]
    assert sample["patches"].shape == (20, 3, 224, 224)
    assert sample["mask"].shape == (20,)
    assert sample["mask"].dtype == torch.bool
    assert sample["mask"][:5].all()
    assert not sample["mask"][5:].any()
    # padded frames are zeros
    assert sample["patches"][5:].abs().sum() == 0


def test_dataset_truncates_too_long_sequences(tmp_path):
    split = _make_split(tmp_path, [("a", 1, 25)])
    ds = TubePatchDataset(split, max_frames=20)
    sample = ds[0]
    assert sample["patches"].shape == (20, 3, 224, 224)
    assert sample["mask"].all()


def test_dataset_returns_label_as_float_tensor(tmp_path):
    split = _make_split(tmp_path, [("smoke_seq", 1, 3), ("fp_seq", 0, 3)])
    ds = TubePatchDataset(split, max_frames=20)
    smoke = ds[0]
    fp = ds[1]
    assert smoke["label"].dtype == torch.float32
    assert smoke["label"].item() == 1.0
    assert fp["label"].item() == 0.0


def test_dataset_normalizes_with_imagenet_stats(tmp_path):
    split = _make_split(tmp_path, [("a", 1, 1)])
    ds = TubePatchDataset(split, max_frames=20)
    sample = ds[0]
    # All pixels were value 50 (uint8). After /255 then normalize per channel,
    # the result must NOT equal raw uint8 / 255 (mean/std applied).
    raw = 50.0 / 255.0
    assert not torch.allclose(sample["patches"][0], torch.full((3, 224, 224), raw))


def test_dataset_returns_sequence_id(tmp_path):
    split = _make_split(tmp_path, [("seq42", 1, 3)])
    ds = TubePatchDataset(split, max_frames=20)
    sample = ds[0]
    assert sample["sequence_id"] == "seq42"
```

- [ ] **Step 2: Run tests, expect fail**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ImportError: cannot import name 'TubePatchDataset'` (current dataset.py defines `SmokeyNetDataset`).

- [ ] **Step 3: Replace `dataset.py`**

Overwrite `src/bbox_tube_temporal/dataset.py` with:

```python
"""PyTorch Dataset for the basic temporal smoke classifier.

Reads cropped PNG patches produced by ``scripts/build_model_input.py``
and returns per-tube tensors padded to a fixed length with a mask.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class TubePatchDataset(Dataset):
    """Dataset of cropped tube patches stored as PNG folders.

    Each item:

    .. code-block:: python

        {
            "patches": Tensor[max_frames, 3, 224, 224],  # float32, ImageNet-normalized
            "mask":    Tensor[max_frames] bool,           # True = real frame
            "label":   Tensor[] float32,                  # 0.0 fp, 1.0 smoke
            "sequence_id": str,
        }

    Args:
        split_dir: Directory containing ``_index.json`` and one
            sub-directory per tube.
        max_frames: Pad/truncate length.
    """

    def __init__(self, split_dir: Path, max_frames: int) -> None:
        self.split_dir = Path(split_dir)
        self.max_frames = max_frames
        index = json.loads((self.split_dir / "_index.json").read_text())
        self.index: list[dict] = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Tensor | str]:
        record = self.index[idx]
        seq_id: str = record["sequence_id"]
        label_int: int = record["label_int"]
        seq_dir = self.split_dir / seq_id

        meta = json.loads((seq_dir / "meta.json").read_text())
        frame_files = [seq_dir / f["filename"] for f in meta["frames"]]
        n = min(len(frame_files), self.max_frames)

        patches = torch.zeros(self.max_frames, 3, 224, 224, dtype=torch.float32)
        mask = torch.zeros(self.max_frames, dtype=torch.bool)
        for i in range(n):
            img = Image.open(frame_files[i]).convert("RGB")
            tensor = to_tensor(img)  # CHW float32 in [0, 1]
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
            patches[i] = tensor
            mask[i] = True

        return {
            "patches": patches,
            "mask": mask,
            "label": torch.tensor(float(label_int), dtype=torch.float32),
            "sequence_id": seq_id,
        }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Lint + format**

Run: `make lint && make format`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/bbox_tube_temporal/dataset.py tests/test_dataset.py
git commit -m "feat(bbox-tube-temporal): TubePatchDataset for tube patch sequences"
```

---

## Task 5: Backbone wrapper (timm + frozen)

**Files:**
- Create: `src/bbox_tube_temporal/temporal_classifier.py`  *(new file — leaves `model.py` alone for this task)*
- Test: `tests/test_temporal_classifier.py`

We isolate the new classifier in its own module so the existing
`model.py` (`SmokeyNetModel`) keeps compiling. We will replace
`model.py` later if/when the heavier model is retired.

- [ ] **Step 1: Write the failing test for the backbone wrapper**

Create `tests/test_temporal_classifier.py`:

```python
"""Tests for TemporalSmokeClassifier and its components."""

import pytest
import torch

from bbox_tube_temporal.temporal_classifier import (
    FrozenTimmBackbone,
    TemporalSmokeClassifier,
)


def test_frozen_timm_backbone_outputs_features_per_frame():
    bb = FrozenTimmBackbone(name="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = bb(x)
    assert out.shape == (2, bb.feat_dim)
    assert bb.feat_dim == 512


def test_frozen_timm_backbone_has_no_trainable_params():
    bb = FrozenTimmBackbone(name="resnet18", pretrained=False)
    trainable = [p for p in bb.parameters() if p.requires_grad]
    assert trainable == []


def test_frozen_timm_backbone_stays_in_eval_mode_after_train_call():
    bb = FrozenTimmBackbone(name="resnet18", pretrained=False)
    bb.train()
    # Even after calling .train(), the inner backbone stays in eval mode.
    assert not bb.backbone.training
```

- [ ] **Step 2: Run, expect fail**

Run: `uv run pytest tests/test_temporal_classifier.py::test_frozen_timm_backbone_outputs_features_per_frame -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the backbone wrapper**

Create `src/bbox_tube_temporal/temporal_classifier.py`:

```python
"""Basic temporal smoke classifier: frozen timm backbone + temporal head."""

import timm
import torch
from torch import Tensor, nn


class FrozenTimmBackbone(nn.Module):
    """Wraps a pretrained timm model as a per-frame feature extractor.

    Always frozen: parameters have ``requires_grad=False`` and the inner
    model is forced to ``eval()`` mode regardless of the parent module's
    training flag (so BatchNorm/Dropout stay deterministic).
    """

    def __init__(self, name: str, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.feat_dim: int = self.backbone.num_features

    def train(self, mode: bool = True) -> "FrozenTimmBackbone":
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_temporal_classifier.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/temporal_classifier.py tests/test_temporal_classifier.py
git commit -m "feat(bbox-tube-temporal): FrozenTimmBackbone wrapper"
```

---

## Task 6: Mean-pool head + masked pooling

**Files:**
- Modify: `src/bbox_tube_temporal/temporal_classifier.py`
- Modify: `tests/test_temporal_classifier.py`

- [ ] **Step 1: Add tests**

Append to `tests/test_temporal_classifier.py`:

```python
from bbox_tube_temporal.temporal_classifier import MeanPoolHead


def test_mean_pool_head_returns_logits_per_batch():
    head = MeanPoolHead(feat_dim=512, hidden_dim=128)
    feats = torch.randn(4, 20, 512)
    mask = torch.ones(4, 20, dtype=torch.bool)
    logits = head(feats, mask)
    assert logits.shape == (4,)


def test_mean_pool_head_respects_mask():
    head = MeanPoolHead(feat_dim=4, hidden_dim=4)
    # Two batches: same first 2 frames, but second batch has noise in frames 2..19
    base = torch.zeros(20, 4)
    base[0] = 1.0
    base[1] = 2.0
    a = base.clone()
    b = base.clone()
    b[2:] = 999.0
    feats = torch.stack([a, b])
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[:, :2] = True
    logits = head(feats, mask)
    # Both should produce the same logit (masked positions ignored)
    assert torch.allclose(logits[0], logits[1], atol=1e-5)
```

- [ ] **Step 2: Run, expect fail**

Run: `uv run pytest tests/test_temporal_classifier.py -v -k mean_pool`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `MeanPoolHead`**

Append to `src/bbox_tube_temporal/temporal_classifier.py`:

```python
class MeanPoolHead(nn.Module):
    """Masked mean over time + 2-layer MLP → 1 logit."""

    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        # feats: (B, T, D); mask: (B, T) bool
        m = mask.unsqueeze(-1).to(feats.dtype)  # (B, T, 1)
        summed = (feats * m).sum(dim=1)  # (B, D)
        counts = m.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = summed / counts
        return self.mlp(pooled).squeeze(-1)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_temporal_classifier.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/temporal_classifier.py tests/test_temporal_classifier.py
git commit -m "feat(bbox-tube-temporal): MeanPoolHead for sanity-baseline temporal head"
```

---

## Task 7: GRU head with packed sequences

**Files:**
- Modify: `src/bbox_tube_temporal/temporal_classifier.py`
- Modify: `tests/test_temporal_classifier.py`

- [ ] **Step 1: Add tests**

Append to `tests/test_temporal_classifier.py`:

```python
from bbox_tube_temporal.temporal_classifier import GRUHead


def test_gru_head_returns_logits_per_batch():
    head = GRUHead(feat_dim=512, hidden_dim=128, num_layers=1, bidirectional=False)
    feats = torch.randn(3, 20, 512)
    mask = torch.ones(3, 20, dtype=torch.bool)
    logits = head(feats, mask)
    assert logits.shape == (3,)


def test_gru_head_respects_mask_via_packed_sequences():
    head = GRUHead(feat_dim=4, hidden_dim=4, num_layers=1, bidirectional=False)
    # Two identical batches truncated to 2 real frames; one has garbage in
    # the padded tail, the other has zeros. Packed sequences must skip both.
    real = torch.randn(2, 4)
    a = torch.zeros(20, 4)
    a[:2] = real
    b = a.clone()
    b[2:] = 1e6
    feats = torch.stack([a, b])
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[:, :2] = True
    logits = head(feats, mask)
    assert torch.allclose(logits[0], logits[1], atol=1e-4)


def test_gru_head_bidirectional_doubles_hidden_then_projects():
    head = GRUHead(feat_dim=8, hidden_dim=4, num_layers=1, bidirectional=True)
    feats = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = head(feats, mask)
    assert logits.shape == (2,)
```

- [ ] **Step 2: Run, expect fail**

Run: `uv run pytest tests/test_temporal_classifier.py -v -k gru_head`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `GRUHead`**

Append to `src/bbox_tube_temporal/temporal_classifier.py`:

```python
from torch.nn.utils.rnn import pack_padded_sequence


class GRUHead(nn.Module):
    """1+ layer GRU over time + MLP → 1 logit. Uses packed sequences."""

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(
            feats, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)
        # h_n: (num_layers * num_directions, B, H). Take last layer.
        if self.bidirectional:
            last_fwd = h_n[-2]
            last_bwd = h_n[-1]
            last = torch.cat([last_fwd, last_bwd], dim=-1)
        else:
            last = h_n[-1]
        return self.mlp(last).squeeze(-1)
```

Note: `pack_padded_sequence` import goes at the **top of file** with other imports — not inline. Move it up.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_temporal_classifier.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/temporal_classifier.py tests/test_temporal_classifier.py
git commit -m "feat(bbox-tube-temporal): GRUHead with packed sequences for masking"
```

---

## Task 8: `TemporalSmokeClassifier` (backbone + head)

**Files:**
- Modify: `src/bbox_tube_temporal/temporal_classifier.py`
- Modify: `tests/test_temporal_classifier.py`

- [ ] **Step 1: Add tests**

Append to `tests/test_temporal_classifier.py`:

```python
def test_classifier_mean_pool_forward_shape():
    clf = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="mean_pool",
        hidden_dim=64,
        pretrained=False,
    )
    patches = torch.randn(2, 5, 3, 224, 224)
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = clf(patches, mask)
    assert logits.shape == (2,)


def test_classifier_gru_forward_shape():
    clf = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=64,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
    )
    patches = torch.randn(2, 5, 3, 224, 224)
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = clf(patches, mask)
    assert logits.shape == (2,)


def test_classifier_only_head_params_are_trainable():
    clf = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=64,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
    )
    trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
    assert all(n.startswith("head.") for n in trainable)
    assert any(n.startswith("head.gru") for n in trainable)


def test_classifier_unknown_arch_raises():
    with pytest.raises(ValueError, match="arch"):
        TemporalSmokeClassifier(
            backbone="resnet18",
            arch="lstm",
            hidden_dim=64,
            pretrained=False,
        )


def test_classifier_handles_padded_batches():
    clf = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
    )
    patches = torch.randn(3, 20, 3, 224, 224)
    mask = torch.zeros(3, 20, dtype=torch.bool)
    mask[0, :20] = True
    mask[1, :10] = True
    mask[2, :3] = True
    logits = clf(patches, mask)
    assert logits.shape == (3,)
```

- [ ] **Step 2: Run, expect failures**

Run: `uv run pytest tests/test_temporal_classifier.py -v -k classifier`
Expected: 5 FAIL (class doesn't exist yet, since the import at the top of test file already pulls `TemporalSmokeClassifier` — this test_classifier_* group fails at instantiation).

- [ ] **Step 3: Implement `TemporalSmokeClassifier`**

Append to `src/bbox_tube_temporal/temporal_classifier.py`:

```python
class TemporalSmokeClassifier(nn.Module):
    """Frozen backbone applied per-frame + temporal head → binary logit per tube."""

    def __init__(
        self,
        backbone: str,
        arch: str,
        hidden_dim: int,
        pretrained: bool = True,
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = FrozenTimmBackbone(name=backbone, pretrained=pretrained)
        feat_dim = self.backbone.feat_dim
        if arch == "mean_pool":
            self.head: nn.Module = MeanPoolHead(feat_dim=feat_dim, hidden_dim=hidden_dim)
        elif arch == "gru":
            self.head = GRUHead(
                feat_dim=feat_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"unknown arch: {arch!r} (expected 'mean_pool' or 'gru')")
        self.arch = arch

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        # patches: (B, T, 3, H, W); mask: (B, T) bool
        b, t, c, h, w = patches.shape
        flat = patches.reshape(b * t, c, h, w)
        feats = self.backbone(flat).reshape(b, t, -1)  # (B, T, D)
        return self.head(feats, mask)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_temporal_classifier.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint + format**

Run: `make lint && make format`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/bbox_tube_temporal/temporal_classifier.py tests/test_temporal_classifier.py
git commit -m "feat(bbox-tube-temporal): TemporalSmokeClassifier combining backbone + head"
```

---

## Task 9: Lightning module for training

**Files:**
- Create: `src/bbox_tube_temporal/lit_temporal.py`  *(new file — keeps existing `training.py` untouched)*
- Test: `tests/test_lit_temporal.py`

- [ ] **Step 1: Write tests**

Create `tests/test_lit_temporal.py`:

```python
"""Tests for the basic temporal classifier Lightning module."""

import torch

from bbox_tube_temporal.lit_temporal import LitTemporalClassifier


def _batch(b: int = 2, t: int = 5) -> dict:
    return {
        "patches": torch.randn(b, t, 3, 224, 224),
        "mask": torch.ones(b, t, dtype=torch.bool),
        "label": torch.tensor([1.0, 0.0][:b]),
        "sequence_id": [f"seq_{i}" for i in range(b)],
    }


def test_lit_module_training_step_returns_loss_scalar():
    lit = LitTemporalClassifier(
        backbone="resnet18",
        arch="mean_pool",
        hidden_dim=32,
        learning_rate=1e-3,
        weight_decay=1e-2,
        pretrained=False,
    )
    loss = lit.training_step(_batch(), batch_idx=0)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_lit_module_validation_step_runs_without_error():
    lit = LitTemporalClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        learning_rate=1e-3,
        weight_decay=1e-2,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
    )
    lit.validation_step(_batch(), batch_idx=0)


def test_lit_module_optimizer_only_includes_head_params():
    lit = LitTemporalClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        learning_rate=1e-3,
        weight_decay=1e-2,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
    )
    opt = lit.configure_optimizers()
    head_param_ids = {id(p) for p in lit.model.head.parameters()}
    optim_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert optim_param_ids == head_param_ids
```

- [ ] **Step 2: Run, expect fail**

Run: `uv run pytest tests/test_lit_temporal.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement Lightning module**

Create `src/bbox_tube_temporal/lit_temporal.py`:

```python
"""PyTorch Lightning wrapper around TemporalSmokeClassifier."""

import lightning as L
import torch
from torch import Tensor

from .temporal_classifier import TemporalSmokeClassifier


class LitTemporalClassifier(L.LightningModule):
    """Lightning module: BCE loss, AdamW on head params only.

    Args:
        backbone: timm model name.
        arch: ``"mean_pool"`` or ``"gru"``.
        hidden_dim: head hidden width.
        learning_rate: AdamW lr.
        weight_decay: AdamW weight decay.
        pretrained: whether to load pretrained backbone weights.
        num_layers: GRU layers (ignored when arch != gru).
        bidirectional: GRU direction (ignored when arch != gru).
    """

    def __init__(
        self,
        backbone: str,
        arch: str,
        hidden_dim: int,
        learning_rate: float,
        weight_decay: float,
        pretrained: bool = True,
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TemporalSmokeClassifier(
            backbone=backbone,
            arch=arch,
            hidden_dim=hidden_dim,
            pretrained=pretrained,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self._val_preds: list[float] = []
        self._val_labels: list[float] = []

    def forward(self, patches: Tensor, mask: Tensor) -> Tensor:
        return self.model(patches, mask)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        logits = self(batch["patches"], batch["mask"])
        loss = self.loss_fn(logits, batch["label"])
        self.log("train/loss", loss, prog_bar=True, batch_size=logits.shape[0])
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["patches"], batch["mask"])
        loss = self.loss_fn(logits, batch["label"])
        probs = torch.sigmoid(logits).detach().cpu()
        labels = batch["label"].detach().cpu()
        self._val_preds.extend(probs.tolist())
        self._val_labels.extend(labels.tolist())
        self.log("val/loss", loss, prog_bar=True, batch_size=logits.shape[0])

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        probs = torch.tensor(self._val_preds)
        labels = torch.tensor(self._val_labels)
        preds = (probs > 0.5).float()
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        tn = ((preds == 0) & (labels == 0)).sum().float()
        acc = (tp + tn) / (tp + tn + fp + fn).clamp(min=1)
        prec = tp / (tp + fp).clamp(min=1)
        rec = tp / (tp + fn).clamp(min=1)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-8)
        self.log("val/accuracy", acc, prog_bar=True)
        self.log("val/precision", prec)
        self.log("val/recall", rec)
        self.log("val/f1", f1, prog_bar=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        head_params = [p for p in self.model.head.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            head_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_lit_temporal.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Lint + format + full test suite**

Run: `make lint && make format && make test`
Expected: all tests pass (existing tests for other modules untouched).

- [ ] **Step 6: Commit**

```bash
git add src/bbox_tube_temporal/lit_temporal.py tests/test_lit_temporal.py
git commit -m "feat(bbox-tube-temporal): LitTemporalClassifier lightning module"
```

---

## Task 10: `train.py` script + DVC stages for both archs

**Files:**
- Create: `scripts/train.py`
- Modify: `dvc.yaml` (add `train_mean_pool` and `train_gru` stages)
- Modify: `params.yaml` (add `train_mean_pool` and `train_gru` sections; remove old `train:` section)

- [ ] **Step 1: Update `params.yaml`**

In `params.yaml`, **delete** the existing `train:` section (the big one with `d_model`, `lstm_layers`, etc.). Append:

```yaml
train_mean_pool:
  arch: mean_pool
  backbone: resnet18
  hidden_dim: 128
  max_frames: 20
  batch_size: 32
  num_workers: 4
  learning_rate: 0.001
  weight_decay: 0.01
  max_epochs: 30
  early_stop_patience: 5
  seed: 42

train_gru:
  arch: gru
  backbone: resnet18
  hidden_dim: 128
  num_layers: 1
  bidirectional: false
  max_frames: 20
  batch_size: 32
  num_workers: 4
  learning_rate: 0.001
  weight_decay: 0.01
  max_epochs: 30
  early_stop_patience: 5
  seed: 42
```

- [ ] **Step 2: Write `train.py`**

Create `scripts/train.py`:

```python
"""Train the basic temporal smoke classifier (mean_pool or gru arch).

Reads a single named section from ``params.yaml`` (e.g. ``train_gru``)
so each DVC stage owns its own params.
"""

import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from bbox_tube_temporal.dataset import TubePatchDataset
from bbox_tube_temporal.lit_temporal import LitTemporalClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["mean_pool", "gru"], required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--params-key", required=True, help="Key in params.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.params_path.read_text())[args.params_key]
    if cfg["arch"] != args.arch:
        raise ValueError(
            f"--arch={args.arch} mismatches params[{args.params_key}].arch={cfg['arch']}"
        )

    L.seed_everything(cfg["seed"], workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TubePatchDataset(args.train_dir, max_frames=cfg["max_frames"])
    val_ds = TubePatchDataset(args.val_dir, max_frames=cfg["max_frames"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        persistent_workers=cfg["num_workers"] > 0,
    )

    lit = LitTemporalClassifier(
        backbone=cfg["backbone"],
        arch=cfg["arch"],
        hidden_dim=cfg["hidden_dim"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        pretrained=True,
        num_layers=cfg.get("num_layers", 1),
        bidirectional=cfg.get("bidirectional", False),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="best",
            monitor="val/f1",
            mode="max",
            save_top_k=1,
            save_weights_only=False,
        ),
        EarlyStopping(monitor="val/f1", mode="max", patience=cfg["early_stop_patience"]),
    ]
    loggers = [
        CSVLogger(save_dir=args.output_dir, name="csv_logs"),
        TensorBoardLogger(save_dir=args.output_dir, name="tb_logs"),
    ]

    trainer = L.Trainer(
        max_epochs=cfg["max_epochs"],
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        deterministic=True,
    )
    trainer.fit(lit, train_loader, val_loader)

    # DVC expects a stable filename — Lightning saves "best.ckpt"; rename to .pt.
    best = args.output_dir / "best.ckpt"
    target = args.output_dir / "best_checkpoint.pt"
    if best.exists():
        if target.exists():
            target.unlink()
        best.rename(target)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add DVC stages**

Append to `dvc.yaml` (after `build_model_input`):

```yaml
  train_mean_pool:
    cmd: >-
      uv run python scripts/train.py
      --arch mean_pool
      --train-dir data/05_model_input/train
      --val-dir data/05_model_input/val
      --output-dir data/06_models/mean_pool
      --params-path params.yaml
      --params-key train_mean_pool
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - data/05_model_input/train
      - data/05_model_input/val
    params:
      - train_mean_pool
    outs:
      - data/06_models/mean_pool/best_checkpoint.pt
    plots:
      - data/06_models/mean_pool/csv_logs/

  train_gru:
    cmd: >-
      uv run python scripts/train.py
      --arch gru
      --train-dir data/05_model_input/train
      --val-dir data/05_model_input/val
      --output-dir data/06_models/gru
      --params-path params.yaml
      --params-key train_gru
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - data/05_model_input/train
      - data/05_model_input/val
    params:
      - train_gru
    outs:
      - data/06_models/gru/best_checkpoint.pt
    plots:
      - data/06_models/gru/csv_logs/
```

- [ ] **Step 4: Smoke-test mean_pool with a small max_epochs override**

Run a quick sanity check using DVC param override (does NOT modify the file):

```bash
uv run dvc exp run -S train_mean_pool.max_epochs=2 train_mean_pool
```

Expected: completes in a few minutes; produces `data/06_models/mean_pool/best_checkpoint.pt` and prints val/f1 each epoch.

- [ ] **Step 5: Reset and run the real mean_pool stage**

```bash
uv run dvc exp apply HEAD  # or simply re-run; we want the params=30 epochs result
uv run dvc repro train_mean_pool
```

Expected: trains to convergence (or early-stops) and produces `best_checkpoint.pt`. val/f1 > 0.6.

- [ ] **Step 6: Run the gru stage**

```bash
uv run dvc repro train_gru
```

Expected: produces `data/06_models/gru/best_checkpoint.pt` with val/f1 > mean_pool's val/f1.

- [ ] **Step 7: Lint + format**

Run: `make lint && make format`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add scripts/train.py dvc.yaml dvc.lock params.yaml
git commit -m "feat(bbox-tube-temporal): train stages for mean_pool and gru classifiers"
```

---

## Task 11: `evaluate.py` script + DVC stages

**Files:**
- Create: `scripts/evaluate.py`
- Modify: `dvc.yaml` (add `evaluate_mean_pool` and `evaluate_gru` foreach stages)

- [ ] **Step 1: Write `evaluate.py`**

Create `scripts/evaluate.py`:

```python
"""Evaluate a trained temporal classifier on a split.

Loads the best checkpoint, runs inference over the dataset, computes
classification metrics + PR/ROC curves, and writes them under
``--output-dir``.
"""

import argparse
import json
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from bbox_tube_temporal.dataset import TubePatchDataset
from bbox_tube_temporal.lit_temporal import LitTemporalClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["mean_pool", "gru"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--params-key", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.params_path.read_text())[args.params_key]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(cfg["seed"], workers=True)

    lit = LitTemporalClassifier.load_from_checkpoint(
        str(args.checkpoint),
        backbone=cfg["backbone"],
        arch=cfg["arch"],
        hidden_dim=cfg["hidden_dim"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        pretrained=False,
        num_layers=cfg.get("num_layers", 1),
        bidirectional=cfg.get("bidirectional", False),
    )
    lit.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit.to(device)

    ds = TubePatchDataset(args.data_dir, max_frames=cfg["max_frames"])
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    all_probs: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            patches = batch["patches"].to(device)
            mask = batch["mask"].to(device)
            logits = lit(patches, mask)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(batch["label"].tolist())

    probs = np.asarray(all_probs)
    labels = np.asarray(all_labels)
    preds = (probs > 0.5).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    pr_auc = float(average_precision_score(labels, probs)) if labels.sum() > 0 else 0.0
    roc_auc = float(roc_auc_score(labels, probs)) if 0 < labels.sum() < len(labels) else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # PR curve
    p, r, _ = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(r, p)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"PR (AP={pr_auc:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(args.output_dir / "pr_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"ROC (AUC={roc_auc:.3f})")
    fig.savefig(args.output_dir / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add scikit-learn to dependencies**

Open `pyproject.toml` and add `"scikit-learn>=1.4",` to the dependencies list. Run `uv sync`.

- [ ] **Step 3: Add DVC stages**

Append to `dvc.yaml`:

```yaml
  evaluate_mean_pool:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/evaluate.py
        --arch mean_pool
        --data-dir data/05_model_input/${item}
        --checkpoint data/06_models/mean_pool/best_checkpoint.pt
        --output-dir data/08_reporting/${item}/mean_pool
        --params-path params.yaml
        --params-key train_mean_pool
      deps:
        - scripts/evaluate.py
        - src/bbox_tube_temporal/temporal_classifier.py
        - src/bbox_tube_temporal/lit_temporal.py
        - src/bbox_tube_temporal/dataset.py
        - data/06_models/mean_pool/best_checkpoint.pt
        - data/05_model_input/${item}
      params:
        - train_mean_pool
      metrics:
        - data/08_reporting/${item}/mean_pool/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/mean_pool/pr_curve.png
        - data/08_reporting/${item}/mean_pool/roc_curve.png

  evaluate_gru:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/evaluate.py
        --arch gru
        --data-dir data/05_model_input/${item}
        --checkpoint data/06_models/gru/best_checkpoint.pt
        --output-dir data/08_reporting/${item}/gru
        --params-path params.yaml
        --params-key train_gru
      deps:
        - scripts/evaluate.py
        - src/bbox_tube_temporal/temporal_classifier.py
        - src/bbox_tube_temporal/lit_temporal.py
        - src/bbox_tube_temporal/dataset.py
        - data/06_models/gru/best_checkpoint.pt
        - data/05_model_input/${item}
      params:
        - train_gru
      metrics:
        - data/08_reporting/${item}/gru/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/gru/pr_curve.png
        - data/08_reporting/${item}/gru/roc_curve.png
```

- [ ] **Step 4: Run all four evaluate stages**

Run: `uv run dvc repro evaluate_mean_pool evaluate_gru`
Expected: writes 4 metrics.json + 8 PNGs under `data/08_reporting/`.

- [ ] **Step 5: Compare metrics**

Run: `uv run dvc metrics show`
Expected: a table with all 4 (split × arch) rows. gru/val should have F1 > mean_pool/val.

- [ ] **Step 6: Lint + format**

Run: `make lint && make format`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/evaluate.py dvc.yaml dvc.lock pyproject.toml uv.lock
git commit -m "feat(bbox-tube-temporal): evaluate stages reporting metrics + PR/ROC plots"
```

---

## Task 12: End-to-end `dvc repro` from clean state

**Files:** none (validation only).

- [ ] **Step 1: Verify pipeline graph**

Run: `uv run dvc dag`
Expected: shows `truncate@{train,val} → build_tubes@{train,val} → build_model_input@{train,val} → train_mean_pool + train_gru → evaluate_*@{train,val}`.

- [ ] **Step 2: Verify the full pipeline is up-to-date**

Run: `uv run dvc status`
Expected: "Data and pipelines are up to date."

- [ ] **Step 3: Read `metrics.json` and confirm success criteria**

Run:
```bash
for f in data/08_reporting/val/mean_pool/metrics.json data/08_reporting/val/gru/metrics.json; do
  echo "=== $f ==="; cat "$f"
done
```

Expected:
- both files exist
- both `f1` values > 0.6
- gru's `f1` >= mean_pool's `f1`

If gru does NOT beat mean_pool, file an issue or revisit hyperparameters before declaring the experiment complete — but this is a finding to report, not a hard failure.

- [ ] **Step 4: Spot-check patches one more time**

Open one or two `data/05_model_input/val/<seq_id>/` directories in a file browser (or `eog`/`feh`) and confirm:
- positive (smoke) tubes show recognizable smoke roughly centered
- negative (fp) tubes look like the FPs that the YOLO detector fires on

- [ ] **Step 5: Final commit (only if anything changed)**

```bash
git status
# If only dvc.lock was updated:
git add dvc.lock
git commit -m "chore(bbox-tube-temporal): refresh dvc.lock after full repro"
```

---

## Self-Review Notes

**Spec coverage:** every spec section maps to one or more tasks:
- Stage 1 `build_model_input` → Tasks 2, 3
- Dataset → Task 4
- Model (backbone + heads + classifier) → Tasks 5, 6, 7, 8
- Lightning module → Task 9
- `train` stages × 2 → Task 10
- `evaluate` stages × 4 → Task 11
- Params (per-stage flat sections) → Task 10 step 1
- DVC stage definitions → Tasks 3, 10, 11
- Success criteria → Task 12

**Naming consistency check:** `TubePatchDataset`, `FrozenTimmBackbone`,
`MeanPoolHead`, `GRUHead`, `TemporalSmokeClassifier`, `LitTemporalClassifier`,
`process_tube`, `expand_bbox`, `norm_bbox_to_pixel_square`,
`crop_and_resize`, `save_patch`, `LABEL_TO_INT` — all referenced
identically across tasks.

**Why we kept existing modules:** `model.py` (`SmokeyNetModel`),
`training.py`, `backbone.py`, `net.py`, `heads.py`, `detector.py`,
`spatial_attention.py`, `temporal_fusion.py`, `package.py` are
untouched. Their tests continue to pass. The basic temporal model
lives in NEW files (`temporal_classifier.py`, `lit_temporal.py`,
`model_input.py`) so this iteration adds without removing.
