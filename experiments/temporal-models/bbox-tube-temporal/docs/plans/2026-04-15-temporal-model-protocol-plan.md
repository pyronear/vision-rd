# TemporalModel protocol for smokeynet-adapted — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `pyrocore.TemporalModel` for `smokeynet-adapted`, packaging the `gru_convnext_finetune` variant together with a YOLO companion detector into a single self-contained archive that the leaderboard can consume end-to-end.

**Architecture:** A thin `BboxTubeTemporalModel` class wiring together five pure inference helpers (YOLO detect → tube build/filter/interpolate → crop patches → classifier forward → decision). A `package.py` module bundles YOLO weights, classifier checkpoint, and config into a `.zip`, mirroring `tracking-fsm-baseline/package.py`. A packager CLI calibrates the sigmoid threshold on val (smallest threshold achieving target recall) and writes it into the bundled config. DVC wires one hard-coded `package` stage for `gru_convnext_finetune`.

**Tech Stack:** Python 3.11+, PyTorch, timm, ultralytics YOLO, Lightning (loading the checkpoint only), PyYAML, pytest.

**Reference spec:** `docs/specs/2026-04-15-temporal-model-protocol-design.md`.

**Working directory for every task:** `experiments/temporal-models/smokeynet-adapted/`. All paths below are relative to that directory unless absolute.

---

## File structure (produced end-to-end)

**New source files**
- `src/bbox_tube_temporal/package.py` — `ModelPackage`, `build_model_package`, `load_model_package`, format constants.
- `src/bbox_tube_temporal/inference.py` — pure helpers: `run_yolo_on_frames`, `filter_and_interpolate_tubes`, `crop_tube_patches`, `score_tubes`, `pick_winner_and_trigger`.
- `src/bbox_tube_temporal/calibration.py` — `calibrate_threshold(probs, labels, target_recall) -> float`.
- `src/bbox_tube_temporal/val_predict.py` — `collect_val_probabilities(classifier, tubes_dir, patches_dir, cfg) -> tuple[np.ndarray, np.ndarray]`.
- `src/bbox_tube_temporal/model.py` — `BboxTubeTemporalModel(TemporalModel)` with `from_package()`.

**New scripts**
- `scripts/package_model.py` — CLI building the `.zip` archive.

**New tests**
- `tests/test_package.py`
- `tests/test_inference_units.py`
- `tests/test_calibration.py`
- `tests/test_model_edge_cases.py`
- `tests/test_model_parity.py`

**Modified files**
- `pyproject.toml` — add `ultralytics` dep.
- `params.yaml` — add `package:` block.
- `dvc.yaml` — add `package` stage.
- `README.md` — document the `model.zip` artefact and `BboxTubeTemporalModel`.

---

## Task 1: Add ultralytics dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dep line**

Edit `pyproject.toml`, add `"ultralytics>=8.3"` to the `dependencies` list (preserving alphabetical order: insert between `"tqdm>=4.66",` and the closing `]`).

After edit, the `dependencies` block ends with:

```toml
    "torch>=2.2",
    "torchvision>=0.17",
    "tqdm>=4.66",
    "ultralytics>=8.3",
]
```

- [ ] **Step 2: Resolve the lockfile**

Run: `uv sync`
Expected: completes without error; `uv.lock` updated.

- [ ] **Step 3: Run the existing test suite as a smoke check**

Run: `uv run pytest tests/ -x -q`
Expected: all pre-existing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(smokeynet-adapted): add ultralytics dependency"
```

---

## Task 2: Create package module skeleton and constants

**Files:**
- Create: `src/bbox_tube_temporal/package.py`
- Test: (added in Task 3)

- [ ] **Step 1: Write the module with constants and dataclass only**

Create `src/bbox_tube_temporal/package.py`:

```python
"""Model packaging: bundle YOLO weights, classifier checkpoint, and config.

The archive is a standard .zip file containing:

- ``manifest.yaml`` — entry point with format version and file pointers.
- ``yolo_weights.pt`` — ultralytics YOLO checkpoint for the companion detector.
- ``classifier.ckpt`` — Lightning checkpoint for ``TemporalSmokeClassifier``.
- ``config.yaml`` — inference config (infer / tubes / model_input / classifier / decision).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.yaml"
YOLO_WEIGHTS_FILENAME = "yolo_weights.pt"
CLASSIFIER_CKPT_FILENAME = "classifier.ckpt"
CONFIG_FILENAME = "config.yaml"
DEFAULT_EXTRACT_DIR = Path(".cache/smokeynet_model")


@dataclass
class ModelPackage:
    """A loaded model package: classifier, YOLO model, and full config."""

    classifier: Any  # TemporalSmokeClassifier; Any avoids import cycles in this module
    yolo_model: Any  # ultralytics.YOLO; same reason
    config: dict[str, Any]

    @property
    def infer(self) -> dict[str, Any]:
        return self.config["infer"]

    @property
    def tubes(self) -> dict[str, Any]:
        return self.config["tubes"]

    @property
    def model_input(self) -> dict[str, Any]:
        return self.config["model_input"]

    @property
    def classifier_cfg(self) -> dict[str, Any]:
        return self.config["classifier"]

    @property
    def decision(self) -> dict[str, Any]:
        return self.config["decision"]
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `uv run python -c "from bbox_tube_temporal.package import ModelPackage, FORMAT_VERSION; print(FORMAT_VERSION)"`
Expected: `1`

- [ ] **Step 3: Commit**

```bash
git add src/bbox_tube_temporal/package.py
git commit -m "feat(smokeynet-adapted): add package module skeleton"
```

---

## Task 3: `build_model_package` + round-trip tests

**Files:**
- Modify: `src/bbox_tube_temporal/package.py`
- Create: `tests/test_package.py`

- [ ] **Step 1: Write failing test for zip creation**

Create `tests/test_package.py`:

```python
"""Tests for BboxTubeTemporalModel packaging."""

import zipfile
from pathlib import Path

import pytest
import yaml

from bbox_tube_temporal.package import (
    CLASSIFIER_CKPT_FILENAME,
    CONFIG_FILENAME,
    FORMAT_VERSION,
    MANIFEST_FILENAME,
    YOLO_WEIGHTS_FILENAME,
    build_model_package,
)

SAMPLE_CONFIG: dict = {
    "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 1024},
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "min_tube_length": 4,
        "infer_min_tube_length": 2,
        "min_detected_entries": 2,
        "interpolate_gaps": True,
    },
    "model_input": {
        "context_factor": 1.5,
        "patch_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "convnext_tiny",
        "arch": "gru",
        "hidden_dim": 128,
        "num_layers": 1,
        "bidirectional": False,
        "max_frames": 20,
        "pretrained": False,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.42,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


@pytest.fixture()
def dummy_yolo_weights(tmp_path: Path) -> Path:
    p = tmp_path / "yolo.pt"
    p.write_bytes(b"fake-yolo")
    return p


@pytest.fixture()
def dummy_classifier_ckpt(tmp_path: Path) -> Path:
    p = tmp_path / "classifier.ckpt"
    p.write_bytes(b"fake-classifier")
    return p


@pytest.fixture()
def built_archive(
    tmp_path: Path, dummy_yolo_weights: Path, dummy_classifier_ckpt: Path
) -> Path:
    out = tmp_path / "model.zip"
    build_model_package(
        yolo_weights_path=dummy_yolo_weights,
        classifier_ckpt_path=dummy_classifier_ckpt,
        config=SAMPLE_CONFIG,
        variant="gru_convnext_finetune",
        output_path=out,
    )
    return out


class TestBuildArchive:
    def test_output_exists(self, built_archive: Path) -> None:
        assert built_archive.exists()

    def test_is_valid_zip(self, built_archive: Path) -> None:
        assert zipfile.is_zipfile(built_archive)

    def test_contains_all_entries(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            names = set(zf.namelist())
        assert names == {
            MANIFEST_FILENAME,
            YOLO_WEIGHTS_FILENAME,
            CLASSIFIER_CKPT_FILENAME,
            CONFIG_FILENAME,
        }

    def test_yolo_weights_preserved(
        self, built_archive: Path, dummy_yolo_weights: Path
    ) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            assert zf.read(YOLO_WEIGHTS_FILENAME) == dummy_yolo_weights.read_bytes()

    def test_classifier_ckpt_preserved(
        self, built_archive: Path, dummy_classifier_ckpt: Path
    ) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            assert (
                zf.read(CLASSIFIER_CKPT_FILENAME)
                == dummy_classifier_ckpt.read_bytes()
            )


class TestManifest:
    def test_format_version(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["format_version"] == FORMAT_VERSION

    def test_variant_recorded(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["variant"] == "gru_convnext_finetune"

    def test_file_pointers(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))
        assert manifest["yolo_weights"] == YOLO_WEIGHTS_FILENAME
        assert manifest["classifier_checkpoint"] == CLASSIFIER_CKPT_FILENAME
        assert manifest["config"] == CONFIG_FILENAME


class TestConfigRoundTrip:
    def test_config_bytes_match(self, built_archive: Path) -> None:
        with zipfile.ZipFile(built_archive, "r") as zf:
            loaded = yaml.safe_load(zf.read(CONFIG_FILENAME))
        assert loaded == SAMPLE_CONFIG


class TestBuildMissingWeightsRaises:
    def test_missing_yolo(
        self, tmp_path: Path, dummy_classifier_ckpt: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            build_model_package(
                yolo_weights_path=tmp_path / "nope.pt",
                classifier_ckpt_path=dummy_classifier_ckpt,
                config=SAMPLE_CONFIG,
                variant="gru_convnext_finetune",
                output_path=tmp_path / "out.zip",
            )

    def test_missing_classifier_ckpt(
        self, tmp_path: Path, dummy_yolo_weights: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            build_model_package(
                yolo_weights_path=dummy_yolo_weights,
                classifier_ckpt_path=tmp_path / "nope.ckpt",
                config=SAMPLE_CONFIG,
                variant="gru_convnext_finetune",
                output_path=tmp_path / "out.zip",
            )
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_package.py -v`
Expected: ImportError for `build_model_package`.

- [ ] **Step 3: Implement `build_model_package`**

Append to `src/bbox_tube_temporal/package.py`:

```python
import zipfile
import yaml


def build_model_package(
    *,
    yolo_weights_path: Path,
    classifier_ckpt_path: Path,
    config: dict[str, Any],
    variant: str,
    output_path: Path,
) -> Path:
    """Bundle YOLO weights + classifier checkpoint + config into a .zip archive.

    Args:
        yolo_weights_path: Path to the ultralytics YOLO ``.pt`` file.
        classifier_ckpt_path: Path to the Lightning ``.ckpt`` for
            ``TemporalSmokeClassifier``.
        config: Full package config dict (see module docstring for schema).
        variant: Identifier recorded in the manifest (informational).
        output_path: Destination ``.zip`` path.

    Returns:
        The resolved ``output_path``.

    Raises:
        FileNotFoundError: If either input file is missing.
    """
    if not yolo_weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {yolo_weights_path}")
    if not classifier_ckpt_path.exists():
        raise FileNotFoundError(
            f"Classifier checkpoint not found: {classifier_ckpt_path}"
        )

    manifest = {
        "format_version": FORMAT_VERSION,
        "variant": variant,
        "yolo_weights": YOLO_WEIGHTS_FILENAME,
        "classifier_checkpoint": CLASSIFIER_CKPT_FILENAME,
        "config": CONFIG_FILENAME,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest, default_flow_style=False))
        zf.write(yolo_weights_path, YOLO_WEIGHTS_FILENAME)
        zf.write(classifier_ckpt_path, CLASSIFIER_CKPT_FILENAME)
        zf.writestr(CONFIG_FILENAME, yaml.dump(config, default_flow_style=False))
    return output_path.resolve()
```

Move the `import zipfile` and `import yaml` to the top of the file with the other imports (above the `FORMAT_VERSION` constant).

- [ ] **Step 4: Run tests — should now pass**

Run: `uv run pytest tests/test_package.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/package.py tests/test_package.py
git commit -m "feat(smokeynet-adapted): add build_model_package with tests"
```

---

## Task 4: `load_model_package` + tests

**Files:**
- Modify: `src/bbox_tube_temporal/package.py`
- Modify: `tests/test_package.py`

- [ ] **Step 1: Add failing load tests**

Append to `tests/test_package.py`:

```python
from unittest.mock import MagicMock, patch

import torch


# Build a real tiny classifier state_dict so load_model_package can construct
# and populate a TemporalSmokeClassifier from it.
@pytest.fixture()
def real_tiny_classifier_ckpt(tmp_path: Path) -> Path:
    """A Lightning-style ckpt holding a TemporalSmokeClassifier state_dict.

    Uses backbone=resnet18 for speed in tests, regardless of the production
    variant.
    """
    from bbox_tube_temporal.temporal_classifier import TemporalSmokeClassifier

    model = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
    )
    # Lightning ckpt schema: torch.save({"state_dict": {...}, ...})
    state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": state_dict}, ckpt_path)
    return ckpt_path


@pytest.fixture()
def real_tiny_config() -> dict:
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in SAMPLE_CONFIG.items()}
    cfg["classifier"] = dict(cfg["classifier"])
    cfg["classifier"]["backbone"] = "resnet18"
    cfg["classifier"]["hidden_dim"] = 32
    return cfg


@pytest.fixture()
def real_tiny_archive(
    tmp_path: Path, dummy_yolo_weights: Path, real_tiny_classifier_ckpt: Path,
    real_tiny_config: dict,
) -> Path:
    out = tmp_path / "tiny_model.zip"
    build_model_package(
        yolo_weights_path=dummy_yolo_weights,
        classifier_ckpt_path=real_tiny_classifier_ckpt,
        config=real_tiny_config,
        variant="tiny",
        output_path=out,
    )
    return out


class TestLoadRoundtrip:
    @patch("bbox_tube_temporal.package._load_yolo")
    def test_config_passthrough(
        self,
        mock_yolo: MagicMock,
        real_tiny_archive: Path,
        tmp_path: Path,
        real_tiny_config: dict,
    ) -> None:
        from bbox_tube_temporal.package import load_model_package

        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        pkg = load_model_package(real_tiny_archive, extract_dir=tmp_path / "ext")
        assert pkg.config == real_tiny_config

    @patch("bbox_tube_temporal.package._load_yolo")
    def test_yolo_returned(
        self, mock_yolo: MagicMock, real_tiny_archive: Path, tmp_path: Path
    ) -> None:
        from bbox_tube_temporal.package import load_model_package

        sentinel = MagicMock(name="FakeYOLO")
        mock_yolo.return_value = sentinel
        pkg = load_model_package(real_tiny_archive, extract_dir=tmp_path / "ext")
        assert pkg.yolo_model is sentinel

    @patch("bbox_tube_temporal.package._load_yolo")
    def test_classifier_forward_runs(
        self,
        mock_yolo: MagicMock,
        real_tiny_archive: Path,
        tmp_path: Path,
    ) -> None:
        from bbox_tube_temporal.package import load_model_package

        mock_yolo.return_value = MagicMock(name="FakeYOLO")
        pkg = load_model_package(real_tiny_archive, extract_dir=tmp_path / "ext")

        patches = torch.zeros(1, 4, 3, 224, 224)
        mask = torch.tensor([[True, True, True, True]])
        with torch.no_grad():
            logit = pkg.classifier(patches, mask)
        assert logit.shape == (1,)


class TestLoadRejectsBadArchive:
    @patch("bbox_tube_temporal.package._load_yolo")
    def test_missing_manifest(
        self, mock_yolo: MagicMock, tmp_path: Path
    ) -> None:
        from bbox_tube_temporal.package import load_model_package

        bad = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr(CONFIG_FILENAME, "infer: {}")
        with pytest.raises(KeyError):
            load_model_package(bad, extract_dir=tmp_path / "ext")

    @patch("bbox_tube_temporal.package._load_yolo")
    def test_unsupported_format_version(
        self, mock_yolo: MagicMock, tmp_path: Path
    ) -> None:
        from bbox_tube_temporal.package import load_model_package

        bad = tmp_path / "bad.zip"
        manifest = {
            "format_version": 99,
            "variant": "x",
            "yolo_weights": YOLO_WEIGHTS_FILENAME,
            "classifier_checkpoint": CLASSIFIER_CKPT_FILENAME,
            "config": CONFIG_FILENAME,
        }
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr(MANIFEST_FILENAME, yaml.dump(manifest))
            zf.writestr(YOLO_WEIGHTS_FILENAME, b"x")
            zf.writestr(CLASSIFIER_CKPT_FILENAME, b"x")
            zf.writestr(CONFIG_FILENAME, "{}")
        with pytest.raises(ValueError, match="format_version"):
            load_model_package(bad, extract_dir=tmp_path / "ext")
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_package.py::TestLoadRoundtrip -v`
Expected: ImportError for `load_model_package`.

- [ ] **Step 3: Implement `load_model_package`**

Append to `src/bbox_tube_temporal/package.py`:

```python
import torch

from .temporal_classifier import TemporalSmokeClassifier


def _load_yolo(weights_path: Path) -> Any:
    """Thin wrapper around ultralytics.YOLO — kept importable at call time
    only so tests can mock it without triggering the heavy import."""
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def _load_classifier(
    ckpt_path: Path, classifier_cfg: dict[str, Any]
) -> TemporalSmokeClassifier:
    """Build a ``TemporalSmokeClassifier`` from config and load its weights.

    Accepts both Lightning-style ckpts (``{"state_dict": {"model.xxx": ...}}``)
    and plain state_dicts (``{"xxx": ...}``).
    """
    model = TemporalSmokeClassifier(
        backbone=classifier_cfg["backbone"],
        arch=classifier_cfg["arch"],
        hidden_dim=classifier_cfg["hidden_dim"],
        pretrained=classifier_cfg.get("pretrained", False),
        num_layers=classifier_cfg.get("num_layers", 1),
        bidirectional=classifier_cfg.get("bidirectional", False),
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        raw = blob["state_dict"]
        sd = {
            k.removeprefix("model."): v for k, v in raw.items() if k.startswith("model.")
        }
    else:
        sd = blob
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def load_model_package(
    package_path: Path,
    extract_dir: Path = DEFAULT_EXTRACT_DIR,
) -> ModelPackage:
    """Load a packaged model archive.

    Args:
        package_path: Path to a ``.zip`` built by :func:`build_model_package`.
        extract_dir: Where to extract YOLO weights and classifier ckpt.

    Raises:
        FileNotFoundError: if ``package_path`` does not exist.
        KeyError: if the archive is missing expected entries.
        ValueError: if ``format_version`` is unsupported.
    """
    if not package_path.exists():
        raise FileNotFoundError(f"Archive not found: {package_path}")

    with zipfile.ZipFile(package_path, "r") as zf:
        names = zf.namelist()
        if MANIFEST_FILENAME not in names:
            raise KeyError(f"Archive missing {MANIFEST_FILENAME}")
        manifest = yaml.safe_load(zf.read(MANIFEST_FILENAME))

        version = manifest.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format_version {version} (expected {FORMAT_VERSION})"
            )

        yolo_name = manifest["yolo_weights"]
        ckpt_name = manifest["classifier_checkpoint"]
        config_name = manifest["config"]
        for n in (yolo_name, ckpt_name, config_name):
            if n not in names:
                raise KeyError(f"Archive missing {n}")

        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extract(yolo_name, extract_dir)
        zf.extract(ckpt_name, extract_dir)
        config = yaml.safe_load(zf.read(config_name))

    yolo_model = _load_yolo(extract_dir / yolo_name)
    classifier = _load_classifier(extract_dir / ckpt_name, config["classifier"])
    return ModelPackage(classifier=classifier, yolo_model=yolo_model, config=config)
```

- [ ] **Step 4: Run all package tests**

Run: `uv run pytest tests/test_package.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/package.py tests/test_package.py
git commit -m "feat(smokeynet-adapted): add load_model_package with round-trip tests"
```

---

## Task 5: Inference helper — `run_yolo_on_frames`

**Files:**
- Create: `src/bbox_tube_temporal/inference.py`
- Create: `tests/test_inference_units.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_inference_units.py`:

```python
"""Pure-function unit tests for inference helpers."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from pyrocore.types import Frame
from bbox_tube_temporal.inference import run_yolo_on_frames
from bbox_tube_temporal.types import Detection, FrameDetections


def _fake_yolo_result(xywhn: list[tuple[float, float, float, float, float]]) -> MagicMock:
    """Build a MagicMock shaped like ultralytics Results for one image.

    xywhn: list of (cx, cy, w, h, conf) tuples.
    """
    boxes = MagicMock()
    boxes.__len__ = lambda self: len(xywhn)
    boxes.xywhn = torch.tensor([[c, cy, w, h] for (c, cy, w, h, _) in xywhn])
    boxes.conf = torch.tensor([conf for (_, _, _, _, conf) in xywhn])
    boxes.cls = torch.zeros(len(xywhn))
    result = MagicMock()
    result.boxes = boxes
    return result


class TestRunYoloOnFrames:
    def test_batches_all_frames_in_single_call(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = [_fake_yolo_result([]), _fake_yolo_result([])]
        frames = [
            Frame(frame_id="f0", image_path=Path("/x/f0.jpg"), timestamp=None),
            Frame(frame_id="f1", image_path=Path("/x/f1.jpg"), timestamp=None),
        ]

        run_yolo_on_frames(yolo, frames, confidence_threshold=0.01, iou_nms=0.2, image_size=1024)

        assert yolo.predict.call_count == 1
        args, kwargs = yolo.predict.call_args
        assert args[0] == ["/x/f0.jpg", "/x/f1.jpg"]
        assert kwargs["conf"] == 0.01
        assert kwargs["iou"] == 0.2
        assert kwargs["imgsz"] == 1024
        assert kwargs["verbose"] is False

    def test_converts_detections(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = [
            _fake_yolo_result([(0.5, 0.4, 0.1, 0.2, 0.9)]),
            _fake_yolo_result([]),
        ]
        ts = datetime(2024, 1, 1, 12, 0, 0)
        frames = [
            Frame(frame_id="f0", image_path=Path("/x/f0.jpg"), timestamp=ts),
            Frame(frame_id="f1", image_path=Path("/x/f1.jpg"), timestamp=None),
        ]

        result = run_yolo_on_frames(
            yolo, frames, confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )

        assert len(result) == 2
        assert result[0] == FrameDetections(
            frame_idx=0,
            frame_id="f0",
            timestamp=ts,
            detections=[
                Detection(class_id=0, cx=0.5, cy=0.4, w=0.1, h=0.2, confidence=0.9)
            ],
        )
        assert result[1] == FrameDetections(
            frame_idx=1, frame_id="f1", timestamp=None, detections=[]
        )

    def test_empty_frames_returns_empty(self) -> None:
        yolo = MagicMock()
        yolo.predict.return_value = []
        result = run_yolo_on_frames(
            yolo, [], confidence_threshold=0.01, iou_nms=0.2, image_size=1024
        )
        assert result == []
        yolo.predict.assert_not_called()
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_inference_units.py::TestRunYoloOnFrames -v`
Expected: ImportError.

- [ ] **Step 3: Implement `run_yolo_on_frames`**

Create `src/bbox_tube_temporal/inference.py`:

```python
"""Pure-function helpers used by :class:`BboxTubeTemporalModel.predict`.

Each helper corresponds to one stage of the six-stage pipeline described in
``docs/specs/2026-04-15-temporal-model-protocol-design.md``. Kept separate so
``predict()`` is thin and each stage is unit-testable in isolation.
"""

from typing import Any

from pyrocore.types import Frame

from .types import Detection, FrameDetections


def run_yolo_on_frames(
    yolo_model: Any,
    frames: list[Frame],
    *,
    confidence_threshold: float,
    iou_nms: float,
    image_size: int,
) -> list[FrameDetections]:
    """Run YOLO once over all frames in a single batched call.

    Args:
        yolo_model: An ultralytics ``YOLO`` instance (or any object exposing
            ``predict(list_of_paths, ...)`` with the same return shape).
        frames: Temporally ordered Pyronear :class:`Frame` objects.
        confidence_threshold: Minimum detection confidence.
        iou_nms: IoU threshold for YOLO's internal NMS.
        image_size: Inference resolution passed to YOLO.

    Returns:
        One :class:`FrameDetections` per input frame (possibly with zero
        detections), in the same order, with ``frame_idx`` = position.
    """
    if not frames:
        return []

    paths = [str(f.image_path) for f in frames]
    results = yolo_model.predict(
        paths,
        conf=confidence_threshold,
        iou=iou_nms,
        imgsz=image_size,
        verbose=False,
    )

    out: list[FrameDetections] = []
    for idx, (frame, pred) in enumerate(zip(frames, results, strict=True)):
        detections: list[Detection] = []
        boxes = pred.boxes
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn
            confs = boxes.conf
            cls = boxes.cls
            for i in range(len(boxes)):
                row = xywhn[i].tolist()
                detections.append(
                    Detection(
                        class_id=int(cls[i].item()),
                        cx=row[0],
                        cy=row[1],
                        w=row[2],
                        h=row[3],
                        confidence=float(confs[i].item()),
                    )
                )
        out.append(
            FrameDetections(
                frame_idx=idx,
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                detections=detections,
            )
        )
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_inference_units.py::TestRunYoloOnFrames -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/inference.py tests/test_inference_units.py
git commit -m "feat(smokeynet-adapted): add run_yolo_on_frames helper"
```

---

## Task 6: Inference helper — `filter_and_interpolate_tubes`

**Files:**
- Modify: `src/bbox_tube_temporal/inference.py`
- Modify: `tests/test_inference_units.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_inference_units.py`:

```python
from bbox_tube_temporal.inference import filter_and_interpolate_tubes
from bbox_tube_temporal.types import Tube, TubeEntry


def _tube(tid: int, entries: list[tuple[int, Detection | None]]) -> Tube:
    return Tube(
        tube_id=tid,
        entries=[TubeEntry(frame_idx=i, detection=d) for (i, d) in entries],
        start_frame=entries[0][0],
        end_frame=entries[-1][0],
    )


def _det(cx: float = 0.5, cy: float = 0.5, w: float = 0.1, h: float = 0.1) -> Detection:
    return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=0.9)


class TestFilterAndInterpolate:
    def test_drops_tubes_shorter_than_min_length(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, _det())]),             # length 2 - keep
            _tube(1, [(3, _det())]),                           # length 1 - drop
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=1, interpolate_gaps=False
        )
        assert [t.tube_id for t in out] == [0]

    def test_drops_tubes_with_too_few_observed(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, None), (2, None), (3, None)]),  # 1 obs, drop
            _tube(1, [(0, _det()), (1, _det()), (2, None), (3, None)]),  # 2 obs, keep
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=False
        )
        assert [t.tube_id for t in out] == [1]

    def test_interpolation_applied_when_enabled(self) -> None:
        tubes = [
            _tube(
                0,
                [
                    (0, _det(cx=0.2)),
                    (1, None),
                    (2, _det(cx=0.4)),
                ],
            ),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=True
        )
        assert len(out) == 1
        mid = out[0].entries[1]
        assert mid.is_gap is True
        assert mid.detection is not None
        assert mid.detection.cx == pytest.approx(0.3)

    def test_interpolation_skipped_when_disabled(self) -> None:
        tubes = [
            _tube(0, [(0, _det()), (1, None), (2, _det())]),
        ]
        out = filter_and_interpolate_tubes(
            tubes, min_tube_length=2, min_detected_entries=2, interpolate_gaps=False
        )
        assert out[0].entries[1].detection is None

    def test_empty_input(self) -> None:
        assert filter_and_interpolate_tubes(
            [], min_tube_length=2, min_detected_entries=1, interpolate_gaps=True
        ) == []
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_inference_units.py::TestFilterAndInterpolate -v`
Expected: ImportError.

- [ ] **Step 3: Implement `filter_and_interpolate_tubes`**

Append to `src/bbox_tube_temporal/inference.py`:

```python
from .tubes import interpolate_gaps as _interpolate_gaps
from .types import Tube


def filter_and_interpolate_tubes(
    tubes: list[Tube],
    *,
    min_tube_length: int,
    min_detected_entries: int,
    interpolate_gaps: bool,
) -> list[Tube]:
    """Filter tubes by length / observation count, then optionally interpolate gaps.

    Args:
        tubes: Candidate tubes (output of :func:`~bbox_tube_temporal.tubes.build_tubes`).
        min_tube_length: Keep tubes with ``end_frame - start_frame + 1 >= min_tube_length``.
        min_detected_entries: Keep tubes with at least this many non-gap entries.
        interpolate_gaps: If True, fill gap entries in surviving tubes via
            :func:`~bbox_tube_temporal.tubes.interpolate_gaps`.

    Returns:
        Surviving tubes in original order.
    """
    survivors: list[Tube] = []
    for t in tubes:
        length = t.end_frame - t.start_frame + 1
        if length < min_tube_length:
            continue
        n_obs = sum(1 for e in t.entries if e.detection is not None)
        if n_obs < min_detected_entries:
            continue
        if interpolate_gaps:
            _interpolate_gaps(t)
        survivors.append(t)
    return survivors
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_inference_units.py::TestFilterAndInterpolate -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/inference.py tests/test_inference_units.py
git commit -m "feat(smokeynet-adapted): add filter_and_interpolate_tubes helper"
```

---

## Task 7: Inference helper — `crop_tube_patches`

**Files:**
- Modify: `src/bbox_tube_temporal/inference.py`
- Modify: `tests/test_inference_units.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_inference_units.py`:

```python
import numpy as np
from PIL import Image

from bbox_tube_temporal.inference import crop_tube_patches


@pytest.fixture()
def red_image_sequence(tmp_path: Path) -> list[Path]:
    """Three 128x128 solid-red JPGs."""
    paths = []
    for i in range(3):
        img = np.full((128, 128, 3), fill_value=[200, 30, 30], dtype=np.uint8)
        p = tmp_path / f"frame_{i:02d}.jpg"
        Image.fromarray(img).save(p, format="JPEG", quality=95)
        paths.append(p)
    return paths


class TestCropTubePatches:
    def test_output_shape(self, red_image_sequence: list[Path]) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in red_image_sequence
        ]
        tube = _tube(
            0,
            [
                (0, _det(cx=0.5, cy=0.5, w=0.2, h=0.2)),
                (1, _det(cx=0.5, cy=0.5, w=0.2, h=0.2)),
                (2, _det(cx=0.5, cy=0.5, w=0.2, h=0.2)),
            ],
        )
        patches, mask = crop_tube_patches(
            tube,
            frames,
            context_factor=1.5,
            patch_size=224,
            max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert patches.shape == (5, 3, 224, 224)
        assert patches.dtype == torch.float32
        assert mask.shape == (5,)
        assert mask.tolist() == [True, True, True, False, False]

    def test_padding_slots_are_zero(self, red_image_sequence: list[Path]) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in red_image_sequence
        ]
        tube = _tube(0, [(0, _det()), (1, _det())])
        patches, mask = crop_tube_patches(
            tube,
            frames,
            context_factor=1.5,
            patch_size=224,
            max_frames=5,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert torch.all(patches[2:] == 0.0)

    def test_truncates_tubes_longer_than_max_frames(
        self, red_image_sequence: list[Path]
    ) -> None:
        frames = [
            Frame(frame_id=p.stem, image_path=p, timestamp=None)
            for p in red_image_sequence
        ]
        tube = _tube(0, [(0, _det()), (1, _det()), (2, _det())])
        patches, mask = crop_tube_patches(
            tube,
            frames,
            context_factor=1.5,
            patch_size=224,
            max_frames=2,
            normalization_mean=[0.485, 0.456, 0.406],
            normalization_std=[0.229, 0.224, 0.225],
        )
        assert patches.shape == (2, 3, 224, 224)
        assert mask.tolist() == [True, True]
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_inference_units.py::TestCropTubePatches -v`
Expected: ImportError.

- [ ] **Step 3: Implement `crop_tube_patches`**

Append to `src/bbox_tube_temporal/inference.py`:

```python
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .model_input import crop_and_resize, expand_bbox, norm_bbox_to_pixel_square


def crop_tube_patches(
    tube: Tube,
    frames: list[Frame],
    *,
    context_factor: float,
    patch_size: int,
    max_frames: int,
    normalization_mean: list[float],
    normalization_std: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop patches for a single tube, padded/truncated to ``max_frames``.

    Matches ``TubePatchDataset.__getitem__`` exactly: PIL→uint8 array,
    ``expand_bbox → norm_bbox_to_pixel_square → crop_and_resize``, then
    ``to_tensor`` (CHW, float32, [0,1]), then mean/std normalization.
    """
    frame_by_idx = {i: f for i, f in enumerate(frames)}

    n = min(len(tube.entries), max_frames)
    patches = torch.zeros(max_frames, 3, patch_size, patch_size, dtype=torch.float32)
    mask = torch.zeros(max_frames, dtype=torch.bool)
    mean_t = torch.tensor(normalization_mean).view(3, 1, 1)
    std_t = torch.tensor(normalization_std).view(3, 1, 1)

    for slot, entry in enumerate(tube.entries[:n]):
        det = entry.detection
        if det is None:
            # Shouldn't happen post-interpolation; leave zero-padded + mask=False.
            continue
        frame = frame_by_idx[entry.frame_idx]
        image = np.array(Image.open(frame.image_path).convert("RGB"))
        img_h, img_w, _ = image.shape

        cx, cy, w, h = expand_bbox(det.cx, det.cy, det.w, det.h, context_factor)
        box = norm_bbox_to_pixel_square(cx, cy, w, h, img_w, img_h)
        patch_np = crop_and_resize(image, box, patch_size)
        patch_t = to_tensor(Image.fromarray(patch_np))  # CHW float32 [0,1]
        patches[slot] = (patch_t - mean_t) / std_t
        mask[slot] = True

    return patches, mask
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_inference_units.py::TestCropTubePatches -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/inference.py tests/test_inference_units.py
git commit -m "feat(smokeynet-adapted): add crop_tube_patches helper"
```

---

## Task 8: Inference helper — `score_tubes`

**Files:**
- Modify: `src/bbox_tube_temporal/inference.py`
- Modify: `tests/test_inference_units.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_inference_units.py`:

```python
from bbox_tube_temporal.inference import score_tubes


class TestScoreTubes:
    def test_empty_input_returns_empty(self) -> None:
        classifier = MagicMock()
        logits = score_tubes(classifier, patches_per_tube=[], masks_per_tube=[])
        assert logits.shape == (0,)
        classifier.assert_not_called()

    def test_single_batched_forward(self) -> None:
        classifier = MagicMock(return_value=torch.tensor([1.2, -0.3]))
        patches = [torch.zeros(4, 3, 8, 8), torch.zeros(4, 3, 8, 8)]
        masks = [torch.tensor([True, True, True, True]),
                 torch.tensor([True, True, False, False])]

        logits = score_tubes(classifier, patches_per_tube=patches, masks_per_tube=masks)

        assert classifier.call_count == 1
        args, _ = classifier.call_args
        assert args[0].shape == (2, 4, 3, 8, 8)
        assert args[1].shape == (2, 4)
        assert logits.tolist() == [1.2, -0.3]
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_inference_units.py::TestScoreTubes -v`
Expected: ImportError.

- [ ] **Step 3: Implement `score_tubes`**

Append to `src/bbox_tube_temporal/inference.py`:

```python
def score_tubes(
    classifier: Any,
    *,
    patches_per_tube: list[torch.Tensor],
    masks_per_tube: list[torch.Tensor],
) -> torch.Tensor:
    """Run one batched classifier forward over all tubes.

    Args:
        classifier: A callable ``(patches[N,T,3,H,W], mask[N,T]) -> logits[N]``.
        patches_per_tube: One ``[T, 3, H, W]`` tensor per tube.
        masks_per_tube: One ``[T]`` bool tensor per tube.

    Returns:
        ``Tensor[N]`` of logits (empty tensor if no tubes).
    """
    if not patches_per_tube:
        return torch.zeros(0)
    patches = torch.stack(patches_per_tube, dim=0)
    mask = torch.stack(masks_per_tube, dim=0)
    with torch.no_grad():
        return classifier(patches, mask)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_inference_units.py::TestScoreTubes -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/inference.py tests/test_inference_units.py
git commit -m "feat(smokeynet-adapted): add score_tubes helper"
```

---

## Task 9: Inference helper — `pick_winner_and_trigger`

**Files:**
- Modify: `src/bbox_tube_temporal/inference.py`
- Modify: `tests/test_inference_units.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_inference_units.py`:

```python
from bbox_tube_temporal.inference import pick_winner_and_trigger


class TestPickWinnerAndTrigger:
    def test_no_tubes_returns_negative(self) -> None:
        res = pick_winner_and_trigger(tubes=[], logits=torch.zeros(0), threshold=0.0)
        assert res == (False, None, None)

    def test_argmax_and_threshold_crossed(self) -> None:
        tubes = [
            _tube(10, [(0, _det()), (1, _det())]),       # end_frame = 1
            _tube(20, [(2, _det()), (3, _det()), (4, _det())]),  # end_frame = 4
        ]
        logits = torch.tensor([-1.0, 0.5])
        res = pick_winner_and_trigger(tubes=tubes, logits=logits, threshold=0.0)
        assert res == (True, 4, 20)

    def test_argmax_below_threshold(self) -> None:
        tubes = [
            _tube(1, [(0, _det()), (1, _det())]),
            _tube(2, [(2, _det()), (3, _det())]),
        ]
        logits = torch.tensor([-2.0, -0.5])
        is_positive, trigger, winner = pick_winner_and_trigger(
            tubes=tubes, logits=logits, threshold=0.0
        )
        assert is_positive is False
        assert trigger is None
        assert winner == 2
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_inference_units.py::TestPickWinnerAndTrigger -v`
Expected: ImportError.

- [ ] **Step 3: Implement `pick_winner_and_trigger`**

Append to `src/bbox_tube_temporal/inference.py`:

```python
def pick_winner_and_trigger(
    *,
    tubes: list[Tube],
    logits: torch.Tensor,
    threshold: float,
) -> tuple[bool, int | None, int | None]:
    """Aggregate per-tube logits into a sequence-level decision.

    Rule: ``winner = argmax(logits)``. If ``logits[winner] >= threshold``,
    the sequence is positive and the trigger frame is the winner tube's
    ``end_frame``. Otherwise the sequence is negative (but ``winner_tube_id``
    is still returned for diagnostics).

    Returns:
        Tuple ``(is_positive, trigger_frame_index, winner_tube_id)``.
        All three are ``None``-ish when ``tubes`` is empty.
    """
    if not tubes:
        return False, None, None
    idx = int(torch.argmax(logits).item())
    winner = tubes[idx]
    is_positive = float(logits[idx].item()) >= threshold
    trigger = winner.end_frame if is_positive else None
    return is_positive, trigger, winner.tube_id
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_inference_units.py::TestPickWinnerAndTrigger -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/inference.py tests/test_inference_units.py
git commit -m "feat(smokeynet-adapted): add pick_winner_and_trigger helper"
```

---

## Task 10: `BboxTubeTemporalModel` skeleton + `from_package`

**Files:**
- Create: `src/bbox_tube_temporal/model.py`

- [ ] **Step 1: Create the class with constructor + factory**

Create `src/bbox_tube_temporal/model.py`:

```python
"""TemporalModel implementation for smokeynet-adapted.

Wires the YOLO companion + tube building + patch cropping + the trained
temporal classifier into the pyrocore :class:`TemporalModel` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pyrocore import Frame, TemporalModel, TemporalModelOutput

from .inference import (
    crop_tube_patches,
    filter_and_interpolate_tubes,
    pick_winner_and_trigger,
    run_yolo_on_frames,
    score_tubes,
)
from .package import ModelPackage, load_model_package
from .tubes import build_tubes


class BboxTubeTemporalModel(TemporalModel):
    """YOLO companion + tube classifier.

    See ``docs/specs/2026-04-15-temporal-model-protocol-design.md`` for the
    full pipeline description.
    """

    def __init__(
        self,
        *,
        yolo_model: Any,
        classifier: Any,
        config: dict[str, Any],
    ) -> None:
        self._yolo = yolo_model
        self._classifier = classifier
        self._cfg = config

    @classmethod
    def from_package(cls, package_path: Path) -> Self:
        pkg: ModelPackage = load_model_package(package_path)
        return cls(
            yolo_model=pkg.yolo_model,
            classifier=pkg.classifier,
            config=pkg.config,
        )

    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        raise NotImplementedError  # implemented in the next task
```

- [ ] **Step 2: Verify it imports**

Run: `uv run python -c "from bbox_tube_temporal.model import BboxTubeTemporalModel; print(BboxTubeTemporalModel)"`
Expected: prints the class.

- [ ] **Step 3: Commit**

```bash
git add src/bbox_tube_temporal/model.py
git commit -m "feat(smokeynet-adapted): add BboxTubeTemporalModel skeleton"
```

---

## Task 11: Implement `BboxTubeTemporalModel.predict` + edge-case tests

**Files:**
- Modify: `src/bbox_tube_temporal/model.py`
- Create: `tests/test_model_edge_cases.py`

- [ ] **Step 1: Write failing edge-case tests**

Create `tests/test_model_edge_cases.py`:

```python
"""Edge-case tests for BboxTubeTemporalModel.predict()."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from pyrocore.types import Frame
from bbox_tube_temporal.model import BboxTubeTemporalModel

TEST_CONFIG: dict = {
    "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 1024},
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "min_tube_length": 4,
        "infer_min_tube_length": 2,
        "min_detected_entries": 2,
        "interpolate_gaps": True,
    },
    "model_input": {
        "context_factor": 1.5,
        "patch_size": 8,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "resnet18",
        "arch": "gru",
        "hidden_dim": 32,
        "num_layers": 1,
        "bidirectional": False,
        "max_frames": 6,
        "pretrained": False,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


def _fake_yolo_factory(per_frame_xywhn: list[list[tuple[float, float, float, float, float]]]):
    """Return a mock YOLO whose ``.predict`` yields fixed detections per frame."""

    def fake_predict(paths: list[str], **_kwargs):
        assert len(paths) == len(per_frame_xywhn)
        results = []
        for boxes in per_frame_xywhn:
            r = MagicMock()
            if not boxes:
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self: 0
                r.boxes.xywhn = torch.zeros(0, 4)
                r.boxes.conf = torch.zeros(0)
                r.boxes.cls = torch.zeros(0)
            else:
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self, n=len(boxes): n
                r.boxes.xywhn = torch.tensor([[c, cy, w, h] for (c, cy, w, h, _) in boxes])
                r.boxes.conf = torch.tensor([conf for (_, _, _, _, conf) in boxes])
                r.boxes.cls = torch.zeros(len(boxes))
            results.append(r)
        return results

    m = MagicMock()
    m.predict.side_effect = fake_predict
    return m


@pytest.fixture()
def tiny_classifier():
    from bbox_tube_temporal.temporal_classifier import TemporalSmokeClassifier

    model = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        pretrained=False,
    )
    model.eval()
    return model


@pytest.fixture()
def red_frames(tmp_path: Path) -> list[Frame]:
    frames = []
    for i in range(6):
        arr = np.full((64, 64, 3), fill_value=[180, 30, 30], dtype=np.uint8)
        p = tmp_path / f"f_{i:02d}.jpg"
        Image.fromarray(arr).save(p, format="JPEG")
        frames.append(Frame(frame_id=p.stem, image_path=p, timestamp=None))
    return frames


class TestEmptyFrames:
    def test_returns_negative(self, tiny_classifier) -> None:
        yolo = MagicMock()
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=[])
        assert out.is_positive is False
        assert out.trigger_frame_index is None
        assert out.details["num_frames"] == 0
        yolo.predict.assert_not_called()


class TestZeroDetections:
    def test_no_tubes_means_negative(
        self, tiny_classifier, red_frames: list[Frame]
    ) -> None:
        yolo = _fake_yolo_factory([[] for _ in red_frames])
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=red_frames)
        assert out.is_positive is False
        assert out.trigger_frame_index is None
        assert out.details["num_tubes_total"] == 0
        assert out.details["num_tubes_kept"] == 0


class TestShortTubeBelowInferFloor:
    def test_single_frame_detection_discarded(
        self, tiny_classifier, red_frames: list[Frame]
    ) -> None:
        # Only frame 0 has a detection — tube length 1, below infer_min_tube_length=2.
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)]] + [[] for _ in red_frames[1:]]
        yolo = _fake_yolo_factory(per_frame)
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=red_frames)
        assert out.is_positive is False
        assert out.details["num_tubes_total"] == 1
        assert out.details["num_tubes_kept"] == 0


class TestTruncation:
    def test_sequence_longer_than_max_frames(
        self, tiny_classifier, red_frames: list[Frame]
    ) -> None:
        # red_frames has 6; TEST_CONFIG max_frames=6; extend to 9.
        extra = red_frames + red_frames[:3]
        per_frame = [[(0.5, 0.5, 0.1, 0.1, 0.9)] for _ in extra]
        yolo = _fake_yolo_factory(per_frame[:6])  # only first 6 passed to YOLO
        model = BboxTubeTemporalModel(
            yolo_model=yolo, classifier=tiny_classifier, config=TEST_CONFIG
        )
        out = model.predict(frames=extra)
        assert out.details["num_frames"] == 9
        assert out.details["num_truncated"] == 3
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_model_edge_cases.py -v`
Expected: fails with `NotImplementedError`.

- [ ] **Step 3: Implement `predict`**

Replace the `raise NotImplementedError` line in `src/bbox_tube_temporal/model.py` with:

```python
    def predict(self, frames: list[Frame]) -> TemporalModelOutput:
        import torch

        infer = self._cfg["infer"]
        tubes_cfg = self._cfg["tubes"]
        mi = self._cfg["model_input"]
        clf_cfg = self._cfg["classifier"]
        dec = self._cfg["decision"]

        original_len = len(frames)
        if original_len == 0:
            return TemporalModelOutput(
                is_positive=False,
                trigger_frame_index=None,
                details={
                    "num_frames": 0,
                    "num_truncated": 0,
                    "num_detections_per_frame": [],
                    "num_tubes_total": 0,
                    "num_tubes_kept": 0,
                    "tube_logits": [],
                    "winner_tube_id": None,
                    "winner_tube_entries": [],
                    "threshold": float(dec["threshold"]),
                },
            )

        truncated = frames[: clf_cfg["max_frames"]]
        n_truncated = original_len - len(truncated)

        frame_dets = run_yolo_on_frames(
            self._yolo,
            truncated,
            confidence_threshold=infer["confidence_threshold"],
            iou_nms=infer["iou_nms"],
            image_size=infer["image_size"],
        )
        num_dets_per_frame = [len(fd.detections) for fd in frame_dets]

        candidate_tubes = build_tubes(
            frame_dets,
            iou_threshold=tubes_cfg["iou_threshold"],
            max_misses=tubes_cfg["max_misses"],
        )
        kept = filter_and_interpolate_tubes(
            candidate_tubes,
            min_tube_length=tubes_cfg["infer_min_tube_length"],
            min_detected_entries=tubes_cfg["min_detected_entries"],
            interpolate_gaps=tubes_cfg["interpolate_gaps"],
        )

        if not kept:
            return TemporalModelOutput(
                is_positive=False,
                trigger_frame_index=None,
                details={
                    "num_frames": original_len,
                    "num_truncated": n_truncated,
                    "num_detections_per_frame": num_dets_per_frame,
                    "num_tubes_total": len(candidate_tubes),
                    "num_tubes_kept": 0,
                    "tube_logits": [],
                    "winner_tube_id": None,
                    "winner_tube_entries": [],
                    "threshold": float(dec["threshold"]),
                },
            )

        patches_per_tube: list[torch.Tensor] = []
        masks_per_tube: list[torch.Tensor] = []
        for t in kept:
            p, m = crop_tube_patches(
                t,
                truncated,
                context_factor=mi["context_factor"],
                patch_size=mi["patch_size"],
                max_frames=clf_cfg["max_frames"],
                normalization_mean=mi["normalization"]["mean"],
                normalization_std=mi["normalization"]["std"],
            )
            patches_per_tube.append(p)
            masks_per_tube.append(m)

        logits = score_tubes(
            self._classifier,
            patches_per_tube=patches_per_tube,
            masks_per_tube=masks_per_tube,
        )

        is_positive, trigger, winner_id = pick_winner_and_trigger(
            tubes=kept, logits=logits, threshold=float(dec["threshold"])
        )

        winner_entries: list[dict] = []
        if winner_id is not None:
            winner = next(t for t in kept if t.tube_id == winner_id)
            for e in winner.entries:
                d = e.detection
                winner_entries.append(
                    {
                        "frame_idx": e.frame_idx,
                        "bbox": [d.cx, d.cy, d.w, d.h] if d is not None else None,
                        "is_gap": e.is_gap,
                        "confidence": d.confidence if d is not None else None,
                    }
                )

        return TemporalModelOutput(
            is_positive=is_positive,
            trigger_frame_index=trigger,
            details={
                "num_frames": original_len,
                "num_truncated": n_truncated,
                "num_detections_per_frame": num_dets_per_frame,
                "num_tubes_total": len(candidate_tubes),
                "num_tubes_kept": len(kept),
                "tube_logits": logits.tolist(),
                "winner_tube_id": winner_id,
                "winner_tube_entries": winner_entries,
                "threshold": float(dec["threshold"]),
            },
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_model_edge_cases.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/model.py tests/test_model_edge_cases.py
git commit -m "feat(smokeynet-adapted): implement BboxTubeTemporalModel.predict with edge-case tests"
```

---

## Task 12: Train/inference parity test

**Files:**
- Create: `tests/test_model_parity.py`
- Create: `tests/fixtures/parity/sequence_with_labels/` (test fixture)

- [ ] **Step 1: Create fixture directory with a real tiny sequence**

The parity test uses one **synthetic** sequence that both paths can consume. The sequence directory layout mirrors the real pyro-dataset (`images/`, `labels/`), so `load_frame_detections` and `build_tubes` work unchanged.

Run this Python snippet once to create the fixture:

```bash
uv run python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image

root = Path("tests/fixtures/parity/wildfire/seq_synth01")
images = root / "images"
labels = root / "labels"
images.mkdir(parents=True, exist_ok=True)
labels.mkdir(parents=True, exist_ok=True)

# Make 5 frames; draw a drifting bright rectangle for the "smoke" so crops differ.
for i in range(5):
    arr = np.full((96, 128, 3), 30, dtype=np.uint8)  # dark background
    x0 = 40 + i * 2  # drift right
    y0 = 30
    arr[y0:y0 + 24, x0:x0 + 24, :] = [220, 220, 220]  # bright smoke region
    stem = f"synth_2024-01-01T00-00-{i:02d}"
    Image.fromarray(arr).save(images / f"{stem}.jpg", format="JPEG", quality=95)
    # GT label format: 5-col class cx cy w h (normalized)
    cx = (x0 + 12) / 128
    cy = (y0 + 12) / 96
    w = 24 / 128
    h = 24 / 96
    (labels / f"{stem}.txt").write_text(f"0 {cx} {cy} {w} {h}\n")
print("fixture created")
PY
```

Expected: prints `fixture created`. Verify `tests/fixtures/parity/wildfire/seq_synth01/images/` has 5 `.jpg` files and `labels/` has 5 `.txt` files.

- [ ] **Step 2: Write failing parity test**

Create `tests/test_model_parity.py`:

```python
"""Parity test: offline training path logits == predict() logits on same input.

Methodology:
- Build tubes offline from GT labels via load_frame_detections + build_tubes
  + select_longest_tube + interpolate_gaps.
- Crop patches via TubePatchDataset's exact preprocessing (saved to temp dir
  via process_tube, then read through TubePatchDataset).
- Forward through the classifier → reference logit.
- Run BboxTubeTemporalModel.predict() with a fake YOLO that returns the
  same GT detections per frame.
- Assert predict()'s winning-tube logit == reference logit to 1e-5.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from pyrocore.types import Frame
from bbox_tube_temporal.data import load_frame_detections
from bbox_tube_temporal.model import BboxTubeTemporalModel
from bbox_tube_temporal.model_input import (
    crop_and_resize,
    expand_bbox,
    norm_bbox_to_pixel_square,
)
from bbox_tube_temporal.temporal_classifier import TemporalSmokeClassifier
from bbox_tube_temporal.tubes import (
    build_tubes,
    interpolate_gaps,
    select_longest_tube,
)

FIXTURE = Path(__file__).parent / "fixtures" / "parity" / "wildfire" / "seq_synth01"


CFG: dict = {
    "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 128},
    "tubes": {
        "iou_threshold": 0.2,
        "max_misses": 2,
        "min_tube_length": 2,
        "infer_min_tube_length": 2,
        "min_detected_entries": 2,
        "interpolate_gaps": True,
    },
    "model_input": {
        "context_factor": 1.5,
        "patch_size": 32,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "resnet18",
        "arch": "gru",
        "hidden_dim": 32,
        "num_layers": 1,
        "bidirectional": False,
        "max_frames": 5,
        "pretrained": False,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


def _offline_logit(classifier: TemporalSmokeClassifier) -> float:
    """Run the offline training path and return the classifier logit."""
    fdets = load_frame_detections(FIXTURE)
    tubes = build_tubes(fdets, iou_threshold=0.2, max_misses=2)
    tube = select_longest_tube(tubes)
    assert tube is not None
    interpolate_gaps(tube)

    mi = CFG["model_input"]
    T = CFG["classifier"]["max_frames"]
    patches = torch.zeros(T, 3, mi["patch_size"], mi["patch_size"])
    mask = torch.zeros(T, dtype=torch.bool)
    frame_paths = sorted((FIXTURE / "images").glob("*.jpg"))
    mean = torch.tensor(mi["normalization"]["mean"]).view(3, 1, 1)
    std = torch.tensor(mi["normalization"]["std"]).view(3, 1, 1)
    for slot, entry in enumerate(tube.entries[:T]):
        det = entry.detection
        assert det is not None
        img = np.array(Image.open(frame_paths[entry.frame_idx]).convert("RGB"))
        H, W, _ = img.shape
        cx, cy, w, h = expand_bbox(det.cx, det.cy, det.w, det.h, mi["context_factor"])
        box = norm_bbox_to_pixel_square(cx, cy, w, h, W, H)
        p = crop_and_resize(img, box, mi["patch_size"])
        pt = to_tensor(Image.fromarray(p))
        patches[slot] = (pt - mean) / std
        mask[slot] = True

    with torch.no_grad():
        logit = classifier(patches.unsqueeze(0), mask.unsqueeze(0))
    return float(logit.item())


def _fake_yolo_from_gt(fixture: Path) -> MagicMock:
    """Build a fake YOLO that returns GT detections per frame."""
    fdets = load_frame_detections(fixture)
    # Keyed by image path (str) for lookup in predict():
    by_path = {}
    for fd in fdets:
        img_path = fixture / "images" / f"{fd.frame_id}.jpg"
        by_path[str(img_path)] = fd.detections

    def predict(paths, **kwargs):
        results = []
        for p in paths:
            dets = by_path[p]
            r = MagicMock()
            if not dets:
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self: 0
                r.boxes.xywhn = torch.zeros(0, 4)
                r.boxes.conf = torch.zeros(0)
                r.boxes.cls = torch.zeros(0)
            else:
                r.boxes = MagicMock()
                r.boxes.__len__ = lambda self, n=len(dets): n
                r.boxes.xywhn = torch.tensor([[d.cx, d.cy, d.w, d.h] for d in dets])
                r.boxes.conf = torch.tensor([d.confidence for d in dets])
                r.boxes.cls = torch.tensor([d.class_id for d in dets]).float()
            results.append(r)
        return results

    m = MagicMock()
    m.predict.side_effect = predict
    return m


@pytest.fixture(scope="module")
def classifier() -> TemporalSmokeClassifier:
    torch.manual_seed(0)
    model = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        pretrained=False,
    )
    model.eval()
    return model


def test_parity_logit_matches(classifier: TemporalSmokeClassifier) -> None:
    offline = _offline_logit(classifier)

    frames = [
        Frame(frame_id=p.stem, image_path=p, timestamp=None)
        for p in sorted((FIXTURE / "images").glob("*.jpg"))
    ]
    yolo = _fake_yolo_from_gt(FIXTURE)
    model = BboxTubeTemporalModel(yolo_model=yolo, classifier=classifier, config=CFG)
    out = model.predict(frames=frames)

    assert out.details["num_tubes_kept"] >= 1
    online = max(out.details["tube_logits"])

    assert online == pytest.approx(offline, abs=1e-5)
```

- [ ] **Step 3: Run test**

Run: `uv run pytest tests/test_model_parity.py -v`
Expected: passes. If it fails due to floating-point drift, investigate which stage introduced the gap (most likely candidate: crop bbox rounding or normalization order). Do not relax the tolerance — fix the cause.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/parity tests/test_model_parity.py
git commit -m "test(smokeynet-adapted): add train/inference parity test"
```

---

## Task 13: Threshold calibration helper

**Files:**
- Create: `src/bbox_tube_temporal/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_calibration.py`:

```python
"""Tests for threshold calibration."""

import numpy as np
import pytest

from bbox_tube_temporal.calibration import calibrate_threshold


class TestCalibrateThreshold:
    def test_picks_smallest_threshold_achieving_recall(self) -> None:
        # 4 positives at probs [0.9, 0.8, 0.7, 0.2]; 4 negatives at [0.1..0.4].
        probs = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.15, 0.3, 0.4])
        labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        # target_recall = 0.75 → need 3 of 4 positives. Threshold just below 0.7.
        t = calibrate_threshold(probs, labels, target_recall=0.75)
        assert 0.4 < t <= 0.7

    def test_recall_1_requires_lowest_positive(self) -> None:
        probs = np.array([0.9, 0.2, 0.1])
        labels = np.array([1, 1, 0])
        t = calibrate_threshold(probs, labels, target_recall=1.0)
        assert t <= 0.2

    def test_raises_if_no_positives(self) -> None:
        probs = np.array([0.1, 0.2])
        labels = np.array([0, 0])
        with pytest.raises(ValueError, match="no positives"):
            calibrate_threshold(probs, labels, target_recall=0.95)

    def test_raises_if_unreachable_recall(self) -> None:
        # target_recall > 1 is silly; enforce 0 < r <= 1.
        probs = np.array([0.5, 0.5])
        labels = np.array([1, 0])
        with pytest.raises(ValueError):
            calibrate_threshold(probs, labels, target_recall=1.5)
```

- [ ] **Step 2: Run failing test**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `calibrate_threshold`**

Create `src/bbox_tube_temporal/calibration.py`:

```python
"""Decision threshold calibration from val predictions.

Picks the smallest sigmoid probability threshold that achieves at least the
requested recall on the supplied (probs, labels) pairs. Used by the packager
to pin ``decision.threshold`` before building the archive.
"""

import numpy as np


def calibrate_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    target_recall: float,
) -> float:
    """Return the smallest ``p`` such that recall at threshold ``p`` >= ``target_recall``.

    Args:
        probs: 1D array of sigmoid probabilities (one per val sample).
        labels: 1D array of 0/1 ground truth, same length as ``probs``.
        target_recall: Desired recall on the positive class; must be in ``(0, 1]``.

    Returns:
        Threshold in ``[0, 1]``.

    Raises:
        ValueError: if ``labels`` has no positives, if arrays are mis-shaped,
            or if ``target_recall`` is not in ``(0, 1]``.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall!r}")
    if probs.shape != labels.shape or probs.ndim != 1:
        raise ValueError("probs and labels must be equal-length 1D arrays")

    pos_probs = np.sort(probs[labels == 1])
    if pos_probs.size == 0:
        raise ValueError("no positives in labels; cannot calibrate recall")

    # We want recall = (#pos with prob >= t) / n_pos >= target_recall.
    # Equivalently: at most floor(n_pos * (1 - target_recall)) positives may
    # fall below t. Sorting pos_probs ascending, the threshold is the
    # (n_drop)-th element; one epsilon below it would still admit it, so
    # we return that element's value (inclusive >=).
    n_pos = pos_probs.size
    n_drop = int(np.floor(n_pos * (1.0 - target_recall)))
    # If target_recall == 1.0, n_drop = 0 → threshold = lowest positive prob.
    return float(pos_probs[n_drop])
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/calibration.py tests/test_calibration.py
git commit -m "feat(smokeynet-adapted): add threshold calibration helper"
```

---

## Task 14: Val-prediction collector

**Files:**
- Create: `src/bbox_tube_temporal/val_predict.py`

This helper runs the classifier over the val `05_model_input/<split>/` directory (patch PNGs) and returns `(probs, labels)`. No new tests — it's a thin composition of existing primitives (`TubePatchDataset`, `LitTemporalClassifier`) and will be exercised end-to-end by Task 15's integration check.

- [ ] **Step 1: Create the helper**

Create `src/bbox_tube_temporal/val_predict.py`:

```python
"""Run a trained classifier over val patches and collect per-sample probabilities.

Used by ``scripts/package_model.py`` to calibrate the decision threshold
without touching the evaluate stage.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import TubePatchDataset
from .temporal_classifier import TemporalSmokeClassifier


def collect_val_probabilities(
    classifier: TemporalSmokeClassifier,
    val_patches_dir: Path,
    *,
    max_frames: int,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the classifier over ``val_patches_dir`` and collect probs/labels.

    Args:
        classifier: A loaded (eval-mode) ``TemporalSmokeClassifier``.
        val_patches_dir: Directory with ``_index.json`` and per-sequence
            patch sub-dirs (output of ``build_model_input`` stage on val).
        max_frames: Same as training config.
        batch_size: DataLoader batch size.
        num_workers: DataLoader workers.
        device: ``"cuda"`` / ``"cpu"``; auto-detect when ``None``.

    Returns:
        ``(probs, labels)`` as 1D numpy arrays.
    """
    dev = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    classifier.to(dev).eval()

    ds = TubePatchDataset(val_patches_dir, max_frames=max_frames)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    probs: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            patches = batch["patches"].to(dev)
            mask = batch["mask"].to(dev)
            logits = classifier(patches, mask)
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch["label"].tolist())

    return np.asarray(probs), np.asarray(labels)
```

- [ ] **Step 2: Import smoke check**

Run: `uv run python -c "from bbox_tube_temporal.val_predict import collect_val_probabilities; print(collect_val_probabilities)"`
Expected: prints the function object.

- [ ] **Step 3: Commit**

```bash
git add src/bbox_tube_temporal/val_predict.py
git commit -m "feat(smokeynet-adapted): add val_predict helper for threshold calibration"
```

---

## Task 15: Packager CLI

**Files:**
- Create: `scripts/package_model.py`

- [ ] **Step 1: Create the script**

Create `scripts/package_model.py`:

```python
"""Build a deployable model archive for one smokeynet-adapted variant.

Usage:
    uv run python scripts/package_model.py \
        --variant gru_convnext_finetune \
        --output data/06_models/gru_convnext_finetune/model.zip
"""

import argparse
from pathlib import Path

import yaml

from bbox_tube_temporal.calibration import calibrate_threshold
from bbox_tube_temporal.package import build_model_package
from bbox_tube_temporal.temporal_classifier import TemporalSmokeClassifier
from bbox_tube_temporal.val_predict import collect_val_probabilities


def _load_classifier_from_ckpt(
    ckpt_path: Path, variant_cfg: dict
) -> TemporalSmokeClassifier:
    import torch

    model = TemporalSmokeClassifier(
        backbone=variant_cfg["backbone"],
        arch=variant_cfg["arch"],
        hidden_dim=variant_cfg["hidden_dim"],
        pretrained=False,
        num_layers=variant_cfg.get("num_layers", 1),
        bidirectional=variant_cfg.get("bidirectional", False),
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    sd = {
        k.removeprefix("model."): v for k, v in raw.items() if k.startswith("model.")
    } or raw
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _build_config(
    all_params: dict, variant_cfg: dict, package_params: dict, threshold: float
) -> dict:
    return {
        "infer": package_params["infer"],
        "tubes": {
            "iou_threshold": all_params["tubes"]["iou_threshold"],
            "max_misses": all_params["tubes"]["max_misses"],
            "min_tube_length": all_params["build_tubes"]["min_tube_length"],
            "infer_min_tube_length": package_params["infer_min_tube_length"],
            "min_detected_entries": all_params["build_tubes"]["min_detected_entries"],
            "interpolate_gaps": True,
        },
        "model_input": {
            "context_factor": all_params["model_input"]["context_factor"],
            "patch_size": all_params["model_input"]["patch_size"],
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "classifier": {
            "backbone": variant_cfg["backbone"],
            "arch": variant_cfg["arch"],
            "hidden_dim": variant_cfg["hidden_dim"],
            "num_layers": variant_cfg.get("num_layers", 1),
            "bidirectional": variant_cfg.get("bidirectional", False),
            "max_frames": variant_cfg["max_frames"],
            "pretrained": False,
        },
        "decision": {
            "aggregation": "max_logit",
            "threshold": float(threshold),
            "target_recall": package_params["target_recall"],
            "trigger_rule": "end_of_winner",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, help="e.g. gru_convnext_finetune")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Default: data/06_models/<variant>/best_checkpoint.pt",
    )
    parser.add_argument(
        "--yolo-weights-path",
        type=Path,
        default=Path("data/01_raw/models/best.pt"),
    )
    parser.add_argument(
        "--val-patches-dir",
        type=Path,
        default=Path("data/05_model_input/val"),
    )
    args = parser.parse_args()

    all_params = yaml.safe_load(args.params_path.read_text())
    variant_key = f"train_{args.variant}"
    if variant_key not in all_params:
        raise KeyError(f"{variant_key} not found in {args.params_path}")
    variant_cfg = all_params[variant_key]

    if "package" not in all_params:
        raise KeyError(f"'package' section missing from {args.params_path}")
    package_params = all_params["package"]

    checkpoint = args.checkpoint_path or (
        Path("data/06_models") / args.variant / "best_checkpoint.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    classifier = _load_classifier_from_ckpt(checkpoint, variant_cfg)
    probs, labels = collect_val_probabilities(
        classifier,
        args.val_patches_dir,
        max_frames=variant_cfg["max_frames"],
        batch_size=variant_cfg.get("batch_size", 32),
        num_workers=variant_cfg.get("num_workers", 4),
    )
    threshold = calibrate_threshold(
        probs, labels, target_recall=package_params["target_recall"]
    )

    config = _build_config(all_params, variant_cfg, package_params, threshold)
    build_model_package(
        yolo_weights_path=args.yolo_weights_path,
        classifier_ckpt_path=checkpoint,
        config=config,
        variant=args.variant,
        output_path=args.output,
    )
    print(
        f"[package] wrote {args.output} | variant={args.variant} "
        f"threshold={threshold:.4f} target_recall={package_params['target_recall']}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `ruff check` pass**

Run: `uv run ruff check scripts/package_model.py src/bbox_tube_temporal/`
Expected: no issues (fix any ruff complaints inline before continuing).

- [ ] **Step 3: Commit**

```bash
git add scripts/package_model.py
git commit -m "feat(smokeynet-adapted): add package_model.py CLI"
```

---

## Task 16: Add `package` section to `params.yaml`

**Files:**
- Modify: `params.yaml`

- [ ] **Step 1: Append the `package` block**

Add to the end of `params.yaml`:

```yaml
package:
  target_recall: 0.95
  infer_min_tube_length: 2
  infer:
    confidence_threshold: 0.01
    iou_nms: 0.2
    image_size: 1024
```

- [ ] **Step 2: Smoke check**

Run:
```bash
uv run python -c "import yaml; p = yaml.safe_load(open('params.yaml')); print(p['package'])"
```
Expected: prints the dict you just added.

- [ ] **Step 3: Commit**

```bash
git add params.yaml
git commit -m "chore(smokeynet-adapted): add package params block"
```

---

## Task 17: Add `package` stage to `dvc.yaml`

**Files:**
- Modify: `dvc.yaml`

- [ ] **Step 1: Identify the current best_checkpoint.pt path**

Run: `ls data/06_models/gru_convnext_finetune/`
Expected: shows `best_checkpoint.pt` (and possibly `best-v1.ckpt`).

If the file is named differently, adjust the `deps` line in step 2 below to match.

- [ ] **Step 2: Append the stage**

Add to `dvc.yaml` under `stages:`:

```yaml
  package:
    cmd: uv run python scripts/package_model.py
         --variant gru_convnext_finetune
         --output data/06_models/gru_convnext_finetune/model.zip
    deps:
      - data/06_models/gru_convnext_finetune/best_checkpoint.pt
      - data/01_raw/models/best.pt
      - data/05_model_input/val
      - scripts/package_model.py
      - src/bbox_tube_temporal/package.py
      - src/bbox_tube_temporal/calibration.py
      - src/bbox_tube_temporal/val_predict.py
    params:
      - package
      - tubes
      - build_tubes
      - model_input
      - train_gru_convnext_finetune
    outs:
      - data/06_models/gru_convnext_finetune/model.zip
```

- [ ] **Step 3: Validate the DVC file**

Run: `uv run dvc stage list`
Expected: lists `package` (and all pre-existing stages) with no parse errors.

- [ ] **Step 4: Commit**

```bash
git add dvc.yaml
git commit -m "chore(smokeynet-adapted): add dvc package stage for gru_convnext_finetune"
```

---

## Task 18: Run the packager and verify the archive loads

**Files:**
- Produces: `data/06_models/gru_convnext_finetune/model.zip`

- [ ] **Step 1: Run the stage**

Run: `uv run dvc repro package`
Expected: completes; prints the threshold; writes `data/06_models/gru_convnext_finetune/model.zip`. If a GPU is available this should take <2 minutes.

If the val patches dir doesn't exist, ensure upstream stages have run: `uv run dvc repro build_model_input_val`.

- [ ] **Step 2: Verify the archive loads end-to-end**

Run:
```bash
uv run python - <<'PY'
from pathlib import Path
from bbox_tube_temporal.model import BboxTubeTemporalModel

model = BboxTubeTemporalModel.from_package(
    Path("data/06_models/gru_convnext_finetune/model.zip")
)
print("variant:", model._cfg.get("classifier", {}).get("backbone"))
print("threshold:", model._cfg["decision"]["threshold"])
print("ok")
PY
```
Expected: prints backbone, a float threshold, and `ok`.

- [ ] **Step 3: Smoke test on one real val sequence**

Pick one sequence path and run predict end-to-end:

```bash
uv run python - <<'PY'
from pathlib import Path
from bbox_tube_temporal.model import BboxTubeTemporalModel
from bbox_tube_temporal.data import get_sorted_frames, list_sequences

model = BboxTubeTemporalModel.from_package(
    Path("data/06_models/gru_convnext_finetune/model.zip")
)
seqs = list_sequences(Path("data/01_raw/datasets_full/val")) or list_sequences(
    Path("data/01_raw/datasets/val")
)
assert seqs, "no val sequences found"
seq = seqs[0]
print("sequence:", seq.name)
frame_paths = get_sorted_frames(seq)
print("num_frames:", len(frame_paths))
out = model.predict_sequence(frame_paths)
print("is_positive:", out.is_positive)
print("trigger_frame_index:", out.trigger_frame_index)
print("num_tubes_kept:", out.details["num_tubes_kept"])
PY
```
Expected: prints a decision and diagnostic counts, no exceptions.

- [ ] **Step 4: Commit (DVC lock + model output)**

```bash
git add dvc.lock data/06_models/gru_convnext_finetune/.gitignore
git commit -m "chore(smokeynet-adapted): produce gru_convnext_finetune model.zip via dvc"
```

(If there is no `.gitignore` update, drop that path from the `git add`. The zip itself is DVC-tracked, not git-tracked.)

---

## Task 19: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Deployment / TemporalModel" section**

Append to `README.md`:

```markdown
## Deployment (TemporalModel)

`BboxTubeTemporalModel` (in `src/bbox_tube_temporal/model.py`) implements
`pyrocore.TemporalModel`. It ships with a YOLO companion detector inside a
single archive built by `scripts/package_model.py`.

Pipeline inside `predict()`: truncate → YOLO → build+filter tubes → crop
224x224 patches → classifier forward → `max_logit` aggregation →
threshold-based decision (`trigger_frame_index = winner_tube.end_frame`).

### Build the archive

```bash
uv run dvc repro package
# -> data/06_models/gru_convnext_finetune/model.zip
```

The packager also calibrates `decision.threshold` on val for
`target_recall=0.95` and bakes it into the archive's `config.yaml`.

### Use the archive

```python
from pathlib import Path
from bbox_tube_temporal.model import BboxTubeTemporalModel

model = BboxTubeTemporalModel.from_package(
    Path("data/06_models/gru_convnext_finetune/model.zip")
)
output = model.predict_sequence(frame_paths)  # list[Path]
```

See `docs/specs/2026-04-15-temporal-model-protocol-design.md` for the full
design, including the train/inference parity guarantees enforced by
`tests/test_model_parity.py`.
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 3: Run lint/format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: no issues.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(smokeynet-adapted): document BboxTubeTemporalModel and packaging"
```

---

## Self-review notes

- **Spec coverage:** Tasks 2-11 implement the six-stage `predict()` pipeline (Key decision 1) and the package format (Keys 2, 3). Task 12 enforces training/inference parity (parity checklist). Task 13 implements threshold calibration (Key 7). Task 11's edge-case tests cover every row of the edge-case table including `infer_min_tube_length=2` (Key 6). Task 15 stitches calibration + packaging into one CLI invocation. Task 17 implements the single-variant DVC stage from the spec's "DVC stages" section verbatim. Known limitations (train/infer distribution gap, prefix scoring, other aggregations, leaderboard integration) remain explicitly out of scope.
- **Placeholder scan:** No TODOs, TBDs, or "fill in later" — every code block is complete. Task 17's stage lists params/deps that are fully specified; the only conditional is the checkpoint filename (step 1 has the check-and-adjust instruction).
- **Type consistency:** `ModelPackage` properties (`infer`, `tubes`, `model_input`, `classifier_cfg`, `decision`) align with the config sections consumed in `predict()`. The parameter names on `crop_tube_patches` (`normalization_mean`/`normalization_std`) match what Task 11's `predict()` passes. Test fixture names (`_tube`, `_det`) are defined before use in each test module.
