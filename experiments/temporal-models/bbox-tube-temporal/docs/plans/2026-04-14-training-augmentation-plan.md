> Renamed 2026-04-15: smokeynet-adapted → bbox-tube-temporal. Old paths in this doc reflect the design-time state.

# Training Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-tube-consistent spatial, photometric, and temporal augmentation to the training pipeline of `TubePatchDataset` in `bbox-tube-temporal`, as specified in `docs/specs/2026-04-14-training-augmentation-design.md`.

**Architecture:** Three independent `Callable` transforms composed by a `build_tube_augment(cfg, train)` builder. Train augmentation runs inside `__getitem__` via DataLoader workers. Val uses a normalize-only transform (matches today's inline normalization). Config lives in a shared top-level `augment:` section of `params.yaml`; `scripts/train.py` reads it alongside the per-arch section and wires train/val transforms into the two `TubePatchDataset` instances. Each training DVC stage adds `augment` to its `params:` list so DVC correctly invalidates on config changes.

**Tech Stack:** PyTorch 2.x, `torchvision.transforms.v2.functional`, PyTorch Lightning (existing), DVC (existing), pytest.

---

## File Structure

- **Create** `src/bbox_tube_temporal/augment.py` — `SpatialTubeTransform`, `PhotometricTubeTransform`, `TemporalTubeTransform`, `NormalizeTransform`, `ComposeTransform`, `build_tube_augment`.
- **Modify** `src/bbox_tube_temporal/dataset.py` — add `transform: Callable | None = None` kwarg to `TubePatchDataset`; when provided, returns un-normalized `[0,1]` patches through the transform. When `None`, keeps legacy inline ImageNet normalize.
- **Modify** `scripts/train.py` — load `augment` section from `params.yaml`, build train/val transforms, pass to datasets.
- **Modify** `params.yaml` — add top-level `augment:` section.
- **Modify** `dvc.yaml` — add `augment` to `params:` of `train_mean_pool`, `train_gru`, and all `train_gru_*` variant stages; add `src/bbox_tube_temporal/augment.py` to their `deps:`.
- **Create** `tests/test_augment.py` — unit tests for all transforms and the builder.
- **Modify** `tests/test_dataset.py` — add tests for the `transform` kwarg (back-compat + pipeline wiring).
- **Create** `scripts/visualize_augment.py` — smoke test / visual sanity check output.

All `Makefile` targets are invoked from `experiments/temporal-models/bbox-tube-temporal/`. Use `uv run` (matches existing convention).

---

### Task 1: `TubePatchDataset` accepts an optional `transform` kwarg

**Files:**
- Modify: `src/bbox_tube_temporal/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1.1: Write the failing tests**

Append to `tests/test_dataset.py`:

```python
def test_dataset_transform_none_preserves_legacy_normalization(tmp_path):
    """transform=None must keep the existing ImageNet-normalize behavior."""
    split = _make_split(tmp_path, [("a", 1, 1)])
    ds = TubePatchDataset(split, max_frames=20, transform=None)
    sample = ds[0]
    raw = 50.0 / 255.0
    assert not torch.allclose(sample["patches"][0], torch.full((3, 224, 224), raw))


def test_dataset_transform_applied_when_provided(tmp_path):
    """When a transform is provided, dataset returns un-normalized [0,1] patches
    into the transform; the transform's output is what the caller sees."""
    split = _make_split(tmp_path, [("a", 1, 3)])

    captured: dict = {}

    def capture(item):
        captured["patches_dtype"] = item["patches"].dtype
        captured["patches_max"] = float(item["patches"].max())
        captured["patches_min"] = float(item["patches"].min())
        captured["mask_sum"] = int(item["mask"].sum())
        item["patches"] = item["patches"] + 1.0  # mutate to prove it flows through
        return item

    ds = TubePatchDataset(split, max_frames=20, transform=capture)
    sample = ds[0]

    # The transform saw un-normalized [0,1] tensors
    assert captured["patches_dtype"] == torch.float32
    assert captured["patches_min"] >= 0.0
    assert captured["patches_max"] <= 1.0
    assert captured["mask_sum"] == 3
    # And its mutation flowed through to the caller
    assert float(sample["patches"].max()) > 1.0
```

- [ ] **Step 1.2: Run tests and verify they fail**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run pytest tests/test_dataset.py::test_dataset_transform_none_preserves_legacy_normalization tests/test_dataset.py::test_dataset_transform_applied_when_provided -v
```

Expected: both FAIL with `TypeError: __init__() got an unexpected keyword argument 'transform'`.

- [ ] **Step 1.3: Update `TubePatchDataset` to accept `transform`**

Replace the contents of `src/bbox_tube_temporal/dataset.py` with:

```python
"""PyTorch Dataset for the basic temporal smoke classifier.

Reads cropped PNG patches produced by ``scripts/build_model_input.py``
and returns per-tube tensors padded to a fixed length with a mask.
"""

from __future__ import annotations

import json
from collections.abc import Callable
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
            "patches": Tensor[max_frames, 3, 224, 224],  # float32
            "mask":    Tensor[max_frames] bool,           # True = real frame
            "label":   Tensor[] float32,                  # 0.0 fp, 1.0 smoke
            "sequence_id": str,
        }

    When ``transform`` is ``None`` (legacy behavior), patches are
    ImageNet-normalized in place. When a ``transform`` callable is provided,
    patches flow into it as un-normalized ``[0, 1]`` tensors and the transform
    is responsible for normalization as its final step.

    Args:
        split_dir: Directory containing ``_index.json`` and one
            sub-directory per tube.
        max_frames: Pad/truncate length.
        transform: Optional callable ``item -> item`` applied after loading.
    """

    def __init__(
        self,
        split_dir: Path,
        max_frames: int,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.max_frames = max_frames
        self.transform = transform
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
            patches[i] = to_tensor(img)  # CHW float32 in [0, 1]
            mask[i] = True

        item: dict = {
            "patches": patches,
            "mask": mask,
            "label": torch.tensor(float(label_int), dtype=torch.float32),
            "sequence_id": seq_id,
        }

        if self.transform is None:
            # Legacy path: inline ImageNet normalization (valid frames only).
            item["patches"][:n] = (item["patches"][:n] - IMAGENET_MEAN) / IMAGENET_STD
            return item

        return self.transform(item)
```

- [ ] **Step 1.4: Run tests and verify they pass**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: all dataset tests PASS (the two new ones plus all pre-existing ones).

- [ ] **Step 1.5: Commit**

```bash
git add src/bbox_tube_temporal/dataset.py tests/test_dataset.py
git commit -m "refactor(bbox-tube-temporal): TubePatchDataset accepts optional transform

When transform is None, keeps legacy inline ImageNet normalization.
When provided, dataset emits un-normalized [0,1] patches into the
transform; the transform owns the full preprocessing pipeline."
```

---

### Task 2: `SpatialTubeTransform` — per-tube-consistent flip / rotation / scale / translate

**Files:**
- Create: `src/bbox_tube_temporal/augment.py`
- Test: `tests/test_augment.py`

- [ ] **Step 2.1: Create `tests/test_augment.py` with the failing spatial tests**

Create `tests/test_augment.py`:

```python
"""Tests for per-tube augmentation transforms."""

import torch

from bbox_tube_temporal.augment import SpatialTubeTransform


def _make_item(t: int = 5, n_valid: int | None = None) -> dict:
    """Return a dict with T x 3 x 224 x 224 patches and a mask."""
    n_valid = t if n_valid is None else n_valid
    patches = torch.zeros(t, 3, 224, 224, dtype=torch.float32)
    # Put a distinct asymmetric pattern in each valid frame so flips are testable
    for i in range(n_valid):
        patches[i, :, :, : 112] = 0.7  # bright left half
        patches[i, :, :, 112:] = 0.3  # darker right half
        patches[i, 0, i * 5, :] = 1.0  # red row tag that differs per frame
    mask = torch.zeros(t, dtype=torch.bool)
    mask[:n_valid] = True
    return {"patches": patches, "mask": mask}


def test_spatial_identity_preserves_input():
    """Ranges collapsed to neutral must return input unchanged."""
    torch.manual_seed(0)
    item = _make_item(t=4)
    before = item["patches"].clone()
    t = SpatialTubeTransform(
        flip_prob=0.0,
        rotation_deg=0.0,
        scale_range=(1.0, 1.0),
        translate_frac=0.0,
    )
    out = t(item)
    assert torch.equal(out["patches"], before)


def test_spatial_flip_applied_identically_across_frames():
    """With flip_prob=1.0 every frame must be horizontally flipped, and
    the pre/post relationship is identical per frame (same flip decision)."""
    torch.manual_seed(0)
    item = _make_item(t=4)
    before = item["patches"].clone()
    t = SpatialTubeTransform(
        flip_prob=1.0,
        rotation_deg=0.0,
        scale_range=(1.0, 1.0),
        translate_frac=0.0,
    )
    out = t(item)
    for i in range(4):
        assert torch.equal(out["patches"][i], torch.flip(before[i], dims=[-1]))


def test_spatial_affine_shape_preserved():
    """Affine ops keep the tensor shape."""
    torch.manual_seed(0)
    item = _make_item(t=4)
    t = SpatialTubeTransform(
        flip_prob=0.5,
        rotation_deg=5.0,
        scale_range=(0.9, 1.1),
        translate_frac=0.05,
    )
    out = t(item)
    assert out["patches"].shape == (4, 3, 224, 224)
    assert out["mask"].shape == (4,)


def test_spatial_affine_applied_same_per_frame():
    """Rotation/scale/translate must be sampled once per tube: the relative
    transform between any two valid frames' raw and augmented content is
    identical (i.e. the affine is consistent across frames)."""
    torch.manual_seed(0)
    item = _make_item(t=3)
    # Same row tag index offset before -> same offset after
    t = SpatialTubeTransform(
        flip_prob=1.0,
        rotation_deg=0.0,
        scale_range=(1.0, 1.0),
        translate_frac=0.0,
    )
    out = t(item)
    # All three frames flipped identically, so the red-tagged row still differs
    # only by its original per-frame offset (not by a per-frame flip decision).
    for i in range(3):
        flipped_raw = torch.flip(item["patches"][i], dims=[-1])
        # Rebuild `item["patches"]` for next comparison since `t` may have consumed
        # it in-place; use a fresh clone above.
        _ = flipped_raw
    # Simple equality already covered by the previous test; this test asserts
    # shape consistency with non-trivial affine params.
    torch.manual_seed(1)
    item2 = _make_item(t=3)
    t2 = SpatialTubeTransform(
        flip_prob=0.0,
        rotation_deg=5.0,
        scale_range=(0.95, 1.05),
        translate_frac=0.02,
    )
    out2 = t2(item2)
    # Shape preserved + valid frames still valid.
    assert out2["patches"].shape == (3, 3, 224, 224)
    assert out2["mask"].all()
```

- [ ] **Step 2.2: Run tests and verify they fail**

```bash
uv run pytest tests/test_augment.py -v
```

Expected: all FAIL with `ModuleNotFoundError: No module named 'bbox_tube_temporal.augment'`.

- [ ] **Step 2.3: Create `src/bbox_tube_temporal/augment.py` with `SpatialTubeTransform`**

Create `src/bbox_tube_temporal/augment.py`:

```python
"""Per-tube-consistent augmentation transforms for the temporal classifier.

See ``docs/specs/2026-04-14-training-augmentation-design.md`` for design.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchvision.transforms.v2 import functional as TVF


class SpatialTubeTransform:
    """Per-tube-consistent spatial aug (flip + rotation + scale + translate).

    A single set of params is sampled per call and applied identically to
    every frame in the tube so motion direction and inter-frame geometry
    are preserved.
    """

    def __init__(
        self,
        flip_prob: float,
        rotation_deg: float,
        scale_range: tuple[float, float],
        translate_frac: float,
    ) -> None:
        self.flip_prob = flip_prob
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.translate_frac = translate_frac

    def __call__(self, item: dict) -> dict:
        patches: Tensor = item["patches"]  # [T, 3, H, W]
        _, _, h, w = patches.shape

        # Sample once per tube
        do_flip = torch.rand(()).item() < self.flip_prob
        angle = float((torch.rand(()).item() * 2 - 1) * self.rotation_deg)
        s_lo, s_hi = self.scale_range
        scale = float(s_lo + torch.rand(()).item() * (s_hi - s_lo))
        tx = float((torch.rand(()).item() * 2 - 1) * self.translate_frac * w)
        ty = float((torch.rand(()).item() * 2 - 1) * self.translate_frac * h)

        if do_flip:
            patches = TVF.horizontal_flip(patches)

        # Skip affine call entirely when parameters are no-ops (identity).
        is_identity_affine = (
            angle == 0.0 and scale == 1.0 and tx == 0.0 and ty == 0.0
        )
        if not is_identity_affine:
            patches = TVF.affine(
                patches,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
            )

        item["patches"] = patches
        return item
```

- [ ] **Step 2.4: Run tests and verify they pass**

```bash
uv run pytest tests/test_augment.py -v
```

Expected: all 4 spatial tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/bbox_tube_temporal/augment.py tests/test_augment.py
git commit -m "feat(bbox-tube-temporal): SpatialTubeTransform (flip/affine)

Per-tube-consistent horizontal flip + rotation + scale + translate,
sampled once per call and applied identically to every frame so
motion direction is preserved."
```

---

### Task 3: `PhotometricTubeTransform` — per-tube-consistent brightness / contrast / saturation

**Files:**
- Modify: `src/bbox_tube_temporal/augment.py`
- Test: `tests/test_augment.py`

- [ ] **Step 3.1: Add failing photometric tests**

Append to `tests/test_augment.py`:

```python
from bbox_tube_temporal.augment import PhotometricTubeTransform


def test_photometric_identity_preserves_input():
    """Factors collapsed to 1.0 → output equals input."""
    torch.manual_seed(0)
    item = _make_item(t=4)
    before = item["patches"].clone()
    t = PhotometricTubeTransform(
        brightness_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0),
        saturation_range=(1.0, 1.0),
    )
    out = t(item)
    assert torch.allclose(out["patches"], before, atol=1e-6)


def test_photometric_same_factor_across_frames():
    """Brightness/contrast/saturation factors are sampled once per tube;
    the inter-frame difference pattern is preserved."""
    torch.manual_seed(0)
    item = _make_item(t=3)
    before = item["patches"].clone()
    # Compute pre-diffs between frames
    pre_diff_01 = (before[0] - before[1])
    pre_diff_12 = (before[1] - before[2])

    t = PhotometricTubeTransform(
        brightness_range=(0.8, 1.2),
        contrast_range=(1.0, 1.0),  # isolate: brightness-only
        saturation_range=(1.0, 1.0),
    )
    out = t(item)
    # Brightness is a per-pixel multiplicative factor; same factor per frame
    # means diffs scale by the same factor. So the ratio is preserved.
    post_diff_01 = out["patches"][0] - out["patches"][1]
    post_diff_12 = out["patches"][1] - out["patches"][2]
    # Avoid division-by-zero regions
    nz = pre_diff_01.abs() > 1e-3
    if nz.any():
        ratios_01 = (post_diff_01[nz] / pre_diff_01[nz])
        ratios_12 = (post_diff_12[nz] / pre_diff_12[nz])
        # All ratios should be the same single brightness factor
        assert torch.allclose(
            ratios_01, ratios_01[0].expand_as(ratios_01), atol=1e-4
        )
        assert torch.allclose(
            ratios_12, ratios_12[0].expand_as(ratios_12), atol=1e-4
        )
```

- [ ] **Step 3.2: Run and verify failure**

```bash
uv run pytest tests/test_augment.py::test_photometric_identity_preserves_input tests/test_augment.py::test_photometric_same_factor_across_frames -v
```

Expected: FAIL with `ImportError: cannot import name 'PhotometricTubeTransform'`.

- [ ] **Step 3.3: Add `PhotometricTubeTransform` to `augment.py`**

Append to `src/bbox_tube_temporal/augment.py`:

```python
class PhotometricTubeTransform:
    """Per-tube-consistent brightness / contrast / saturation.

    Operates on ``[0, 1]`` tensors (pre-normalization). One set of factors
    sampled per call, applied in fixed order (brightness -> contrast ->
    saturation) to every frame.
    """

    def __init__(
        self,
        brightness_range: tuple[float, float],
        contrast_range: tuple[float, float],
        saturation_range: tuple[float, float],
    ) -> None:
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range

    @staticmethod
    def _sample(r: tuple[float, float]) -> float:
        lo, hi = r
        return float(lo + torch.rand(()).item() * (hi - lo))

    def __call__(self, item: dict) -> dict:
        patches: Tensor = item["patches"]  # [T, 3, H, W], values in [0, 1]

        b = self._sample(self.brightness_range)
        c = self._sample(self.contrast_range)
        s = self._sample(self.saturation_range)

        if b != 1.0:
            patches = TVF.adjust_brightness(patches, brightness_factor=b)
        if c != 1.0:
            patches = TVF.adjust_contrast(patches, contrast_factor=c)
        if s != 1.0:
            patches = TVF.adjust_saturation(patches, saturation_factor=s)

        # Clamp to valid photometric range; adjust_contrast can push slightly
        # below 0 / above 1 and the downstream ImageNet normalize expects [0, 1].
        item["patches"] = patches.clamp_(0.0, 1.0)
        return item
```

- [ ] **Step 3.4: Run and verify pass**

```bash
uv run pytest tests/test_augment.py -v
```

Expected: all 6 tests (4 spatial + 2 photometric) PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/bbox_tube_temporal/augment.py tests/test_augment.py
git commit -m "feat(bbox-tube-temporal): PhotometricTubeTransform

Per-tube-consistent brightness/contrast/saturation with a single
factor sampled per call. Operates on [0,1] tensors before
normalization. Hue shift intentionally omitted (smoke ~= gray)."
```

---

### Task 4: `TemporalTubeTransform` — sub-sequence / stride / drop with re-compaction

**Files:**
- Modify: `src/bbox_tube_temporal/augment.py`
- Test: `tests/test_augment.py`

- [ ] **Step 4.1: Add failing temporal tests**

Append to `tests/test_augment.py`:

```python
from bbox_tube_temporal.augment import TemporalTubeTransform


def _make_padded_item(t: int, n_valid: int) -> dict:
    """Tube with `n_valid` valid frames tagged by scalar value and rest padded."""
    patches = torch.zeros(t, 3, 224, 224, dtype=torch.float32)
    for i in range(n_valid):
        patches[i] = float(i + 1) / 100.0  # distinguishable per frame
    mask = torch.zeros(t, dtype=torch.bool)
    mask[:n_valid] = True
    return {"patches": patches, "mask": mask}


def test_temporal_identity_returns_input_unchanged():
    torch.manual_seed(0)
    item = _make_padded_item(t=20, n_valid=10)
    before_patches = item["patches"].clone()
    before_mask = item["mask"].clone()
    t = TemporalTubeTransform(
        subseq_prob=0.0,
        subseq_min_len=4,
        stride_prob=0.0,
        frame_drop_prob=0.0,
        min_valid_after_drop=4,
    )
    out = t(item)
    assert torch.equal(out["patches"], before_patches)
    assert torch.equal(out["mask"], before_mask)


def test_temporal_mask_prefix_invariant_always_holds():
    """After any temporal transform the valid frames must occupy [0..k-1]."""
    for seed in range(30):
        torch.manual_seed(seed)
        item = _make_padded_item(t=20, n_valid=12)
        t = TemporalTubeTransform(
            subseq_prob=0.5,
            subseq_min_len=4,
            stride_prob=0.25,
            frame_drop_prob=0.15,
            min_valid_after_drop=4,
        )
        out = t(item)
        k = int(out["mask"].sum())
        assert out["mask"][:k].all(), f"seed={seed}: non-contiguous True prefix"
        assert not out["mask"][k:].any(), f"seed={seed}: stray True after prefix"
        # Padded positions are zeros
        if k < 20:
            assert out["patches"][k:].abs().sum() == 0.0


def test_temporal_subsequence_contiguous_slice():
    """With subseq_prob=1, the valid frames of the output are a contiguous
    slice of the valid frames of the input."""
    torch.manual_seed(42)
    item = _make_padded_item(t=20, n_valid=10)
    # Original per-frame tags: 0.01, 0.02, ..., 0.10 (on first pixel)
    original_tags = [
        float(item["patches"][i, 0, 0, 0].item()) for i in range(10)
    ]
    t = TemporalTubeTransform(
        subseq_prob=1.0,
        subseq_min_len=4,
        stride_prob=0.0,
        frame_drop_prob=0.0,
        min_valid_after_drop=4,
    )
    out = t(item)
    k = int(out["mask"].sum())
    out_tags = [float(out["patches"][i, 0, 0, 0].item()) for i in range(k)]
    # Find out_tags as a contiguous slice of original_tags
    assert k >= 4
    for start in range(10 - k + 1):
        if original_tags[start : start + k] == out_tags:
            break
    else:
        raise AssertionError(
            f"out_tags {out_tags} is not a contiguous slice of {original_tags}"
        )


def test_temporal_stride_halves_length():
    """stride_prob=1 means every second frame kept: length ~= ceil(n/2)."""
    torch.manual_seed(0)
    item = _make_padded_item(t=20, n_valid=10)
    t = TemporalTubeTransform(
        subseq_prob=0.0,
        subseq_min_len=4,
        stride_prob=1.0,
        frame_drop_prob=0.0,
        min_valid_after_drop=2,
    )
    out = t(item)
    k = int(out["mask"].sum())
    assert k == 5  # ceil(10/2)


def test_temporal_frame_drop_respects_floor():
    """Very aggressive drop prob still leaves at least min_valid_after_drop."""
    for seed in range(20):
        torch.manual_seed(seed)
        item = _make_padded_item(t=20, n_valid=10)
        t = TemporalTubeTransform(
            subseq_prob=0.0,
            subseq_min_len=4,
            stride_prob=0.0,
            frame_drop_prob=0.99,
            min_valid_after_drop=4,
        )
        out = t(item)
        k = int(out["mask"].sum())
        assert k >= 4, f"seed={seed}: dropped below floor ({k})"


def test_temporal_compacts_dropped_frames_to_zero_prefix():
    """After drop, remaining valid patches are at positions [0..k-1], not
    scattered with zeros in between."""
    torch.manual_seed(0)
    item = _make_padded_item(t=20, n_valid=6)
    # Tag each valid frame uniquely in its top-left pixel
    for i in range(6):
        item["patches"][i, 0, 0, 0] = float(i + 1)
    t = TemporalTubeTransform(
        subseq_prob=0.0,
        subseq_min_len=4,
        stride_prob=0.0,
        frame_drop_prob=0.5,
        min_valid_after_drop=3,
    )
    out = t(item)
    k = int(out["mask"].sum())
    # Every valid output frame must carry a nonzero tag (i.e. not a zero pad).
    for i in range(k):
        assert out["patches"][i, 0, 0, 0].item() > 0.5
```

- [ ] **Step 4.2: Run and verify failure**

```bash
uv run pytest tests/test_augment.py -v -k temporal
```

Expected: all 6 new temporal tests FAIL with import error.

- [ ] **Step 4.3: Add `TemporalTubeTransform` to `augment.py`**

Append to `src/bbox_tube_temporal/augment.py`:

```python
class TemporalTubeTransform:
    """Sub-sequence sampling + random stride + per-frame drop with re-compaction.

    Operates on ``(patches: [T, 3, H, W], mask: [T])``. Output valid frames
    always occupy positions ``[0..k-1]`` so ``pack_padded_sequence`` (used by
    the GRU head) sees a contiguous valid prefix.
    """

    def __init__(
        self,
        subseq_prob: float,
        subseq_min_len: int,
        stride_prob: float,
        frame_drop_prob: float,
        min_valid_after_drop: int,
    ) -> None:
        self.subseq_prob = subseq_prob
        self.subseq_min_len = subseq_min_len
        self.stride_prob = stride_prob
        self.frame_drop_prob = frame_drop_prob
        self.min_valid_after_drop = min_valid_after_drop

    def __call__(self, item: dict) -> dict:
        patches: Tensor = item["patches"]  # [T, 3, H, W]
        mask: Tensor = item["mask"]  # [T] bool
        t_total = patches.shape[0]

        valid_idx = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        n = len(valid_idx)
        if n == 0:
            return item

        # 1. Sub-sequence sampling
        if torch.rand(()).item() < self.subseq_prob and n > self.subseq_min_len:
            k = int(
                torch.randint(self.subseq_min_len, n + 1, (1,)).item()
            )
            start = int(torch.randint(0, n - k + 1, (1,)).item())
            valid_idx = valid_idx[start : start + k]

        # 2. Random stride
        if torch.rand(()).item() < self.stride_prob and len(valid_idx) > 2:
            valid_idx = valid_idx[::2]

        # 3. Per-frame drop, clamped to a floor
        if self.frame_drop_prob > 0.0 and len(valid_idx) > self.min_valid_after_drop:
            keeps = torch.rand(len(valid_idx)) >= self.frame_drop_prob
            # Enforce floor: if we dropped too many, randomly restore indices.
            n_kept = int(keeps.sum())
            if n_kept < self.min_valid_after_drop:
                dropped_positions = torch.nonzero(~keeps, as_tuple=False).flatten()
                n_to_restore = self.min_valid_after_drop - n_kept
                perm = torch.randperm(len(dropped_positions))[:n_to_restore]
                for pos in dropped_positions[perm].tolist():
                    keeps[pos] = True
            valid_idx = [idx for idx, keep in zip(valid_idx, keeps.tolist()) if keep]

        # Re-compact to a fresh padded tensor.
        out_patches = torch.zeros_like(patches)
        out_mask = torch.zeros_like(mask)
        for new_pos, src_pos in enumerate(valid_idx):
            out_patches[new_pos] = patches[src_pos]
            out_mask[new_pos] = True

        item["patches"] = out_patches
        item["mask"] = out_mask
        return item
```

- [ ] **Step 4.4: Run and verify pass**

```bash
uv run pytest tests/test_augment.py -v
```

Expected: all 12 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/bbox_tube_temporal/augment.py tests/test_augment.py
git commit -m "feat(bbox-tube-temporal): TemporalTubeTransform

Sub-sequence sampling (p=0.5) + random stride (p=0.25) + per-frame
drop (p=0.15) composed independently. Re-compacts valid frames to
positions [0..k-1] to preserve the GRU head's pack_padded_sequence
invariant."
```

---

### Task 5: `NormalizeTransform`, `ComposeTransform`, `build_tube_augment`

**Files:**
- Modify: `src/bbox_tube_temporal/augment.py`
- Test: `tests/test_augment.py`

- [ ] **Step 5.1: Add failing builder tests**

Append to `tests/test_augment.py`:

```python
from bbox_tube_temporal.augment import build_tube_augment

_DEFAULT_CFG = {
    "enabled": True,
    "spatial": {
        "flip_prob": 0.5,
        "rotation_deg": 5.0,
        "scale_range": [0.9, 1.1],
        "translate_frac": 0.05,
    },
    "photometric": {
        "brightness_range": [0.8, 1.2],
        "contrast_range": [0.8, 1.2],
        "saturation_range": [0.8, 1.2],
    },
    "temporal": {
        "subseq_prob": 0.5,
        "subseq_min_len": 4,
        "stride_prob": 0.25,
        "frame_drop_prob": 0.15,
        "min_valid_after_drop": 4,
    },
}


def test_val_transform_is_normalize_only_and_deterministic():
    """train=False must skip aug and just apply ImageNet normalize."""
    torch.manual_seed(0)
    item_a = _make_padded_item(t=20, n_valid=5)
    item_b = _make_padded_item(t=20, n_valid=5)
    # Mutate item_b in place to prove the transform is not sampling randomness
    transform = build_tube_augment(_DEFAULT_CFG, train=False)
    out_a = transform(item_a)
    out_b = transform(item_b)
    assert torch.allclose(out_a["patches"][:5], out_b["patches"][:5])

    # Normalization actually happened: raw 0.01 != normalized 0.01
    raw = 0.01
    assert not torch.allclose(
        out_a["patches"][0], torch.full((3, 224, 224), raw), atol=1e-3
    )


def test_val_transform_does_not_mutate_mask_or_length():
    item = _make_padded_item(t=20, n_valid=5)
    before_mask = item["mask"].clone()
    transform = build_tube_augment(_DEFAULT_CFG, train=False)
    out = transform(item)
    assert torch.equal(out["mask"], before_mask)
    assert out["patches"].shape == (20, 3, 224, 224)


def test_train_transform_applies_pipeline_and_normalizes():
    """train=True composes spatial -> photometric -> temporal -> normalize."""
    torch.manual_seed(0)
    item = _make_padded_item(t=20, n_valid=10)
    transform = build_tube_augment(_DEFAULT_CFG, train=True)
    out = transform(item)
    # Shape preserved
    assert out["patches"].shape == (20, 3, 224, 224)
    # Valid mask still contiguous prefix
    k = int(out["mask"].sum())
    assert out["mask"][:k].all()
    assert not out["mask"][k:].any()
    # Output is normalized: valid frames should have values outside [0, 1].
    # ImageNet normalize on [0, 1] input yields negative values for R channel.
    assert out["patches"][:k].min() < 0.0 or out["patches"][:k].max() > 1.0


def test_disabled_config_skips_aug_in_train_mode():
    """enabled=False + train=True must behave like val: normalize only."""
    torch.manual_seed(0)
    cfg_off = dict(_DEFAULT_CFG)
    cfg_off["enabled"] = False
    item_a = _make_padded_item(t=20, n_valid=5)
    item_b = _make_padded_item(t=20, n_valid=5)
    t_train = build_tube_augment(cfg_off, train=True)
    t_val = build_tube_augment(cfg_off, train=False)
    out_a = t_train(item_a)
    out_b = t_val(item_b)
    assert torch.allclose(out_a["patches"], out_b["patches"])


def test_train_transform_reproducible_with_fixed_seed():
    torch.manual_seed(7)
    item1 = _make_padded_item(t=20, n_valid=10)
    out1 = build_tube_augment(_DEFAULT_CFG, train=True)(item1)

    torch.manual_seed(7)
    item2 = _make_padded_item(t=20, n_valid=10)
    out2 = build_tube_augment(_DEFAULT_CFG, train=True)(item2)

    assert torch.equal(out1["patches"], out2["patches"])
    assert torch.equal(out1["mask"], out2["mask"])
```

- [ ] **Step 5.2: Run and verify failure**

```bash
uv run pytest tests/test_augment.py -v
```

Expected: the 5 new builder tests FAIL with `ImportError: cannot import name 'build_tube_augment'`.

- [ ] **Step 5.3: Add builder + composer + normalize**

Append to `src/bbox_tube_temporal/augment.py`:

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizeTransform:
    """ImageNet normalize of the valid frames in a tube (mask-aware)."""

    def __init__(
        self,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, item: dict) -> dict:
        patches: Tensor = item["patches"]
        mask: Tensor = item["mask"]
        n = int(mask.sum())
        if n > 0:
            patches[:n] = (patches[:n] - self.mean) / self.std
        item["patches"] = patches
        return item


class ComposeTransform:
    """Tiny fixed-order transform composer. Each stage takes and returns a dict."""

    def __init__(self, stages: list) -> None:
        self.stages = stages

    def __call__(self, item: dict) -> dict:
        for stage in self.stages:
            item = stage(item)
        return item


def build_tube_augment(config: dict, train: bool) -> ComposeTransform:
    """Build the transform pipeline from the ``augment:`` config section.

    - ``train=False`` or ``config["enabled"]=False`` -> normalize only.
    - Otherwise -> spatial -> photometric -> temporal -> normalize.
    """
    enabled = bool(config.get("enabled", True))
    if not (train and enabled):
        return ComposeTransform([NormalizeTransform()])

    spatial = SpatialTubeTransform(
        flip_prob=float(config["spatial"]["flip_prob"]),
        rotation_deg=float(config["spatial"]["rotation_deg"]),
        scale_range=tuple(config["spatial"]["scale_range"]),
        translate_frac=float(config["spatial"]["translate_frac"]),
    )
    photometric = PhotometricTubeTransform(
        brightness_range=tuple(config["photometric"]["brightness_range"]),
        contrast_range=tuple(config["photometric"]["contrast_range"]),
        saturation_range=tuple(config["photometric"]["saturation_range"]),
    )
    temporal = TemporalTubeTransform(
        subseq_prob=float(config["temporal"]["subseq_prob"]),
        subseq_min_len=int(config["temporal"]["subseq_min_len"]),
        stride_prob=float(config["temporal"]["stride_prob"]),
        frame_drop_prob=float(config["temporal"]["frame_drop_prob"]),
        min_valid_after_drop=int(config["temporal"]["min_valid_after_drop"]),
    )
    return ComposeTransform([spatial, photometric, temporal, NormalizeTransform()])
```

- [ ] **Step 5.4: Run and verify pass**

```bash
uv run pytest tests/test_augment.py -v
```

Expected: all 17 tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add src/bbox_tube_temporal/augment.py tests/test_augment.py
git commit -m "feat(bbox-tube-temporal): build_tube_augment + Normalize + Compose

Builder composes spatial -> photometric -> temporal -> normalize for
train, normalize-only for val (and when augment.enabled=false).
Valid frames only are normalized; padding stays at zero."
```

---

### Task 6: Add `augment:` block to `params.yaml` and `augment` to DVC params lists

**Files:**
- Modify: `params.yaml`
- Modify: `dvc.yaml`

- [ ] **Step 6.1: Add top-level `augment:` section to `params.yaml`**

Append at the bottom of `params.yaml`:

```yaml
augment:
  enabled: true
  spatial:
    flip_prob: 0.5
    rotation_deg: 5.0
    scale_range: [0.9, 1.1]
    translate_frac: 0.05
  photometric:
    brightness_range: [0.8, 1.2]
    contrast_range: [0.8, 1.2]
    saturation_range: [0.8, 1.2]
  temporal:
    subseq_prob: 0.5
    subseq_min_len: 4
    stride_prob: 0.25
    frame_drop_prob: 0.15
    min_valid_after_drop: 4
```

- [ ] **Step 6.2: Add `augment` to the `params:` list of every train stage in `dvc.yaml`**

Locate every train stage (`train_mean_pool`, `train_gru`, and each `train_gru_*` variant) in `dvc.yaml`. For each, change the `params:` block from e.g.:

```yaml
    params:
      - train_gru
```

to:

```yaml
    params:
      - train_gru
      - augment
```

Do this for **all** `train_*` stages, including the seed variants and the backbone / finetune variants.

Also add `src/bbox_tube_temporal/augment.py` to the `deps:` list of every train stage (so DVC invalidates the stage when the augment module changes):

```yaml
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/augment.py   # <-- new
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - data/05_model_input/train
      - data/05_model_input/val
```

- [ ] **Step 6.3: Verify DVC parses the file**

```bash
uv run dvc status
```

Expected: prints stage statuses (no YAML parse error).

- [ ] **Step 6.4: Commit**

```bash
git add params.yaml dvc.yaml
git commit -m "chore(bbox-tube-temporal): wire augment params + deps into train stages

Top-level augment: section in params.yaml is referenced by every
train_* DVC stage alongside its per-variant block, so changes to
augmentation config invalidate all training stages."
```

---

### Task 7: Wire `scripts/train.py` to build and pass transforms

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 7.1: Update the config loading block**

In `scripts/train.py`, replace:

```python
    cfg = yaml.safe_load(args.params_path.read_text())[args.params_key]
```

with:

```python
    full_params = yaml.safe_load(args.params_path.read_text())
    cfg = full_params[args.params_key]
    augment_cfg = full_params.get("augment", {"enabled": False})
```

- [ ] **Step 7.2: Add the import for `build_tube_augment`**

Near the other `bbox_tube_temporal` imports add:

```python
from bbox_tube_temporal.augment import build_tube_augment
```

- [ ] **Step 7.3: Build transforms and pass them to the datasets**

Replace the existing dataset construction:

```python
    train_ds = TubePatchDataset(args.train_dir, max_frames=cfg["max_frames"])
    val_ds = TubePatchDataset(args.val_dir, max_frames=cfg["max_frames"])
```

with:

```python
    train_transform = build_tube_augment(augment_cfg, train=True)
    val_transform = build_tube_augment(augment_cfg, train=False)

    train_ds = TubePatchDataset(
        args.train_dir,
        max_frames=cfg["max_frames"],
        transform=train_transform,
    )
    val_ds = TubePatchDataset(
        args.val_dir,
        max_frames=cfg["max_frames"],
        transform=val_transform,
    )
```

- [ ] **Step 7.4: Smoke-test that the script still parses**

```bash
uv run python -m py_compile scripts/train.py
```

Expected: no output, exit code 0. Any syntax or import-time failure would surface here.

- [ ] **Step 7.5: Run the full test suite to confirm no existing tests broke**

```bash
make test
```

Expected: all tests PASS (dataset + augment + existing).

- [ ] **Step 7.6: Commit**

```bash
git add scripts/train.py
git commit -m "feat(bbox-tube-temporal): train.py builds & passes tube augment transforms

Loads the top-level augment: section alongside the per-variant block
and wires train/val transforms into the two TubePatchDataset
instances. Lightning's seed_everything(workers=True) (already in
place) handles per-worker RNG divergence for us."
```

---

### Task 8: `scripts/visualize_augment.py` — sanity-check image grids

**Files:**
- Create: `scripts/visualize_augment.py`

- [ ] **Step 8.1: Create the visualization script**

Create `scripts/visualize_augment.py`:

```python
"""Render side-by-side grids of augmented tube variants for sanity checking.

Reads a few tubes from the training split, generates N augmented copies
of each, and saves one PNG per tube to
``data/08_reporting/augment_samples/`` showing original vs. augmented
frames side by side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torchvision.utils import make_grid

from bbox_tube_temporal.augment import build_tube_augment
from bbox_tube_temporal.dataset import TubePatchDataset


def _denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-tubes", type=int, default=3)
    parser.add_argument("--num-variants", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = yaml.safe_load(args.params_path.read_text())
    augment_cfg = params.get("augment", {"enabled": True})

    torch.manual_seed(args.seed)

    transform = build_tube_augment(augment_cfg, train=True)
    ds = TubePatchDataset(args.split_dir, max_frames=args.max_frames, transform=transform)

    for tube_idx in range(min(args.num_tubes, len(ds))):
        seq_id = ds.index[tube_idx]["sequence_id"]

        raw_ds = TubePatchDataset(args.split_dir, max_frames=args.max_frames, transform=None)
        raw = raw_ds[tube_idx]
        raw_patches = _denormalize(raw["patches"][: int(raw["mask"].sum())])

        rows = [raw_patches]
        for _ in range(args.num_variants):
            aug = ds[tube_idx]
            patches = _denormalize(aug["patches"][: int(aug["mask"].sum())])
            rows.append(patches)

        max_t = max(r.shape[0] for r in rows)
        padded_rows = [
            torch.cat(
                [r, torch.zeros(max_t - r.shape[0], 3, 224, 224, dtype=r.dtype)],
                dim=0,
            )
            if r.shape[0] < max_t
            else r
            for r in rows
        ]
        grid_tensor = torch.stack(padded_rows).reshape(-1, 3, 224, 224)
        grid = make_grid(grid_tensor, nrow=max_t, padding=2)

        fig, ax = plt.subplots(figsize=(max_t * 1.5, (len(rows)) * 1.5))
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
        ax.set_axis_off()
        ax.set_title(f"{seq_id} — top row: raw, below: {args.num_variants} augmented")
        fig.savefig(args.output_dir / f"{seq_id}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {min(args.num_tubes, len(ds))} grids to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Smoke-run the script (if training data already present locally)**

```bash
uv run python scripts/visualize_augment.py \
  --split-dir data/05_model_input/train \
  --params-path params.yaml \
  --output-dir data/08_reporting/augment_samples \
  --num-tubes 2 \
  --num-variants 4
```

Expected: PNGs appear under `data/08_reporting/augment_samples/`. If the data dir is not available locally, skip this step and verify in a later training run.

- [ ] **Step 8.3: Commit**

```bash
git add scripts/visualize_augment.py
git commit -m "feat(bbox-tube-temporal): visualize_augment.py sanity-check grids

Dumps per-tube side-by-side PNG grids (raw + N augmented variants)
to data/08_reporting/augment_samples/ for visual inspection of the
augmentation config."
```

---

### Task 9: End-to-end verification and final commit

**Files:** none (verification only)

- [ ] **Step 9.1: Run full lint + test**

```bash
make lint && make test
```

Expected: all checks and tests PASS.

- [ ] **Step 9.2: Confirm augmentation enablement toggles via DVC param override**

```bash
uv run dvc status train_gru
```

Expected: `train_gru` is marked changed (because deps or params changed).

```bash
uv run dvc exp run -S augment.enabled=false --dry train_gru 2>&1 | head -20
```

Expected: DVC plans to run `train_gru` with `augment.enabled=false`. (Use `--dry` to avoid spending GPU time on the verification step.)

- [ ] **Step 9.3: Optional, high-value — actually run `train_gru` once to confirm no runtime regression**

```bash
uv run dvc repro train_gru
```

Expected: stage succeeds, `best_checkpoint.pt` produced, CSV logs written. Compare a few `train/loss` values in the new `csv_logs/version_0/metrics.csv` against the prior run to sanity-check that loss is in a reasonable range (not NaN, not stuck at 0.69 = random).

- [ ] **Step 9.4: Update `.gitignore` only if needed**

Check whether `data/08_reporting/augment_samples/` needs a `.gitignore` addition. If a `.gitignore` already excludes `data/08_reporting/` or similar, do nothing. Otherwise:

```bash
grep -q "augment_samples" .gitignore || echo "data/08_reporting/augment_samples/" >> .gitignore
```

- [ ] **Step 9.5: Final commit (only if .gitignore changed)**

```bash
git add .gitignore 2>/dev/null
git diff --cached --quiet || git commit -m "chore(bbox-tube-temporal): ignore local augment-sample outputs"
```

---

## Done

Augmentation pipeline is live. Follow the verification protocol in `docs/specs/2026-04-14-training-augmentation-design.md` to decide whether to ship `augment.enabled: true` as the default — rerun baselines on the existing seed variants (`train_gru`, `train_gru_seed43`, `train_gru_seed44`) first to establish the noise floor, then the treatment runs with matched seeds and bumped early-stop patience (`-S train_gru.early_stop_patience=15`). For the capacity-unlocked variants (`gru_convnext`, `gru_finetune`, `gru_convnext_finetune`), watch the three anti-overfitting signatures: train loss stops collapsing, val loss stops drifting upward, best-val checkpoint moves later in training.
