# Performance Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce false-positive count at target recall 0.95–0.99 on the per-tube classifier by swapping/fine-tuning the backbone. Produce 5 new training variants + a comparison report stage, all wired into DVC.

**Architecture:** Generalize `FrozenTimmBackbone` to `TimmBackbone` with a `finetune` flag that unfreezes the last N blocks per family (resnet / convnext). Extend `LitTemporalClassifier` with a per-group optimizer (backbone LR ≠ head LR). Add 5 new `train_*` + matching `evaluate_*` stages and one `compare_variants` reporting stage.

**Tech Stack:** PyTorch, PyTorch Lightning, timm, DVC, pytest.

**Spec:** `docs/specs/2026-04-14-performance-improvements-design.md`

**Working directory for all commands:** `experiments/temporal-models/smokeynet-adapted/`

---

## Task 1: Rename `FrozenTimmBackbone` to `TimmBackbone` (no behavior change yet)

**Files:**
- Modify: `src/bbox_tube_temporal/temporal_classifier.py`
- Modify: `tests/test_temporal_classifier.py`

- [ ] **Step 1: Rename the class and export**

In `src/bbox_tube_temporal/temporal_classifier.py`, rename `class FrozenTimmBackbone(nn.Module):` to `class TimmBackbone(nn.Module):`. Update the single reference inside `TemporalSmokeClassifier.__init__` from `FrozenTimmBackbone(...)` to `TimmBackbone(...)`.

- [ ] **Step 2: Update existing tests to use the new name**

In `tests/test_temporal_classifier.py`, replace every `FrozenTimmBackbone` occurrence with `TimmBackbone`. This includes the import line and all four test functions (`test_frozen_timm_backbone_outputs_features_per_frame`, `test_frozen_timm_backbone_has_no_trainable_params`, `test_frozen_timm_backbone_stays_in_eval_mode_after_train_call`). Rename the test functions too by replacing `frozen_timm_backbone` with `timm_backbone` in their names.

- [ ] **Step 3: Run tests and verify green**

Run: `uv run pytest tests/test_temporal_classifier.py tests/test_lit_temporal.py -v`
Expected: PASS (all tests). Rename is pure — no behavior change.

- [ ] **Step 4: Verify no stale references remain**

Run: `grep -rn FrozenTimmBackbone src/ tests/ scripts/`
Expected: no matches.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/temporal_classifier.py tests/test_temporal_classifier.py
git commit -m "refactor(smokeynet-adapted): rename FrozenTimmBackbone to TimmBackbone"
```

---

## Task 2: Add `finetune` + `finetune_last_n_blocks` flags to `TimmBackbone` (resnet family)

**Files:**
- Modify: `src/bbox_tube_temporal/temporal_classifier.py`
- Modify: `tests/test_temporal_classifier.py`

- [ ] **Step 1: Write failing tests for the new flags (resnet18)**

Append to `tests/test_temporal_classifier.py`:

```python
def test_timm_backbone_frozen_default_has_no_trainable_params():
    bb = TimmBackbone(name="resnet18", pretrained=False)
    assert [p for p in bb.parameters() if p.requires_grad] == []


def test_timm_backbone_finetune_resnet18_unfreezes_only_layer4():
    bb = TimmBackbone(
        name="resnet18",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names, "expected some trainable params"
    assert all(".layer4." in n for n in trainable_names), trainable_names


def test_timm_backbone_finetune_resnet18_n2_unfreezes_layer3_and_layer4():
    bb = TimmBackbone(
        name="resnet18",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=2,
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all((".layer3." in n) or (".layer4." in n) for n in trainable_names)


def test_timm_backbone_finetune_train_mode_propagates_resnet18():
    bb = TimmBackbone(
        name="resnet18",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
    )
    bb.train()
    assert bb.backbone.layer4.training is True
    assert bb.backbone.layer1.training is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_temporal_classifier.py -v -k "finetune or frozen_default"`
Expected: FAIL — `TimmBackbone.__init__()` does not accept `finetune` / `finetune_last_n_blocks`.

- [ ] **Step 3: Implement the new signature and resnet unfreezing logic**

Replace the `TimmBackbone` class body in `src/bbox_tube_temporal/temporal_classifier.py` with:

```python
class TimmBackbone(nn.Module):
    """Wraps a pretrained timm model as a per-frame feature extractor.

    When ``finetune=False`` (default), all params are frozen and the
    inner model is forced into ``eval()`` mode regardless of the parent
    module's training flag; forward is wrapped in ``torch.no_grad()``.

    When ``finetune=True``, the last ``finetune_last_n_blocks`` blocks
    are unfrozen (family-specific resolution); everything else stays
    frozen, and ``.train()`` propagates normally so BatchNorm on
    unfrozen blocks updates.
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.feat_dim: int = self.backbone.num_features
        self.finetune = finetune
        self.name = name

        if not finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            return

        # Finetune path: freeze everything first, then unfreeze last N blocks.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._unfreeze_last_n_blocks(finetune_last_n_blocks)

    def _unfreeze_last_n_blocks(self, n: int) -> None:
        if n <= 0:
            return
        name = self.name
        if name.startswith("resnet"):
            stages = [
                getattr(self.backbone, f"layer{i}") for i in range(1, 5)
            ]
        else:
            stage_names = [
                n_ for n_, _ in self.backbone.named_children()
            ]
            raise NotImplementedError(
                f"finetune=True is not implemented for backbone family "
                f"{name!r}. Top-level children: {stage_names}. Add an "
                f"explicit unfreeze rule in TimmBackbone._unfreeze_last_n_blocks."
            )
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True

    def train(self, mode: bool = True) -> "TimmBackbone":
        super().train(mode)
        if not self.finetune:
            self.backbone.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if not self.finetune:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)
```

Leave the two existing head classes (`MeanPoolHead`, `GRUHead`) and `TemporalSmokeClassifier` unchanged for now — they still call `TimmBackbone(name=backbone, pretrained=pretrained)` which keeps the frozen default.

- [ ] **Step 4: Run tests to verify new tests pass AND prior tests still pass**

Run: `uv run pytest tests/test_temporal_classifier.py tests/test_lit_temporal.py -v`
Expected: PASS (all tests, including the new ones and the original frozen-equivalence tests).

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/temporal_classifier.py tests/test_temporal_classifier.py
git commit -m "feat(smokeynet-adapted): finetune flag on TimmBackbone for resnet family"
```

---

## Task 3: Add convnext support to `TimmBackbone`

**Files:**
- Modify: `src/bbox_tube_temporal/temporal_classifier.py`
- Modify: `tests/test_temporal_classifier.py`

- [ ] **Step 1: Write failing test for convnext_tiny**

Append to `tests/test_temporal_classifier.py`:

```python
def test_timm_backbone_finetune_convnext_tiny_unfreezes_only_last_stage():
    bb = TimmBackbone(
        name="convnext_tiny",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names
    # timm's convnext wraps blocks under `stages.<i>.*`; last stage is index 3
    assert all(".stages.3." in n for n in trainable_names), trainable_names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_temporal_classifier.py::test_timm_backbone_finetune_convnext_tiny_unfreezes_only_last_stage -v`
Expected: FAIL — hits the `NotImplementedError` branch in `_unfreeze_last_n_blocks`.

- [ ] **Step 3: Add convnext branch**

In `TimmBackbone._unfreeze_last_n_blocks`, replace the `else` branch so the method reads:

```python
    def _unfreeze_last_n_blocks(self, n: int) -> None:
        if n <= 0:
            return
        name = self.name
        if name.startswith("resnet"):
            stages = [
                getattr(self.backbone, f"layer{i}") for i in range(1, 5)
            ]
        elif name.startswith("convnext"):
            # timm's convnext exposes a `stages` ModuleList
            stages = list(self.backbone.stages)
        else:
            stage_names = [
                n_ for n_, _ in self.backbone.named_children()
            ]
            raise NotImplementedError(
                f"finetune=True is not implemented for backbone family "
                f"{name!r}. Top-level children: {stage_names}. Add an "
                f"explicit unfreeze rule in TimmBackbone._unfreeze_last_n_blocks."
            )
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True
```

- [ ] **Step 4: Run tests and verify green**

Run: `uv run pytest tests/test_temporal_classifier.py -v -k convnext`
Expected: PASS.

Also run the full file to confirm nothing regressed:
Run: `uv run pytest tests/test_temporal_classifier.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/temporal_classifier.py tests/test_temporal_classifier.py
git commit -m "feat(smokeynet-adapted): convnext support in TimmBackbone finetune path"
```

---

## Task 4: Raise `NotImplementedError` for unsupported families (explicit test)

**Files:**
- Modify: `tests/test_temporal_classifier.py`

The error-raising behavior already exists from Task 2; this task only adds the regression test so the message contract is guarded.

- [ ] **Step 1: Write test**

Append to `tests/test_temporal_classifier.py`:

```python
import pytest  # add near other imports if not already present


def test_timm_backbone_finetune_unsupported_family_raises():
    with pytest.raises(NotImplementedError) as exc:
        TimmBackbone(
            name="vit_small_patch16_224",
            pretrained=False,
            finetune=True,
            finetune_last_n_blocks=1,
        )
    msg = str(exc.value)
    assert "vit_small_patch16_224" in msg
    # Error message should mention the backbone's top-level children so the
    # operator knows which stage names are available.
    assert "children" in msg.lower()
```

(If `pytest` is already imported in the file, don't add a duplicate import.)

- [ ] **Step 2: Run test and verify it passes**

Run: `uv run pytest tests/test_temporal_classifier.py::test_timm_backbone_finetune_unsupported_family_raises -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_temporal_classifier.py
git commit -m "test(smokeynet-adapted): guard NotImplementedError for unsupported backbone families"
```

---

## Task 5: Frozen-equivalence regression test

**Files:**
- Modify: `tests/test_temporal_classifier.py`

Protects against future changes to the finetune path accidentally altering frozen behavior.

- [ ] **Step 1: Write test**

Append to `tests/test_temporal_classifier.py`:

```python
def test_timm_backbone_frozen_forward_matches_no_grad_eval():
    torch.manual_seed(0)
    bb = TimmBackbone(name="resnet18", pretrained=False, finetune=False)
    x = torch.randn(2, 3, 224, 224)
    # Put the module into train mode; frozen path must still force eval().
    bb.train()
    assert bb.backbone.training is False
    out_a = bb(x)
    # Independent reference: call the underlying timm model directly in eval/no_grad.
    bb.backbone.eval()
    with torch.no_grad():
        out_b = bb.backbone(x)
    assert torch.allclose(out_a, out_b, atol=1e-6)
```

- [ ] **Step 2: Run and verify**

Run: `uv run pytest tests/test_temporal_classifier.py::test_timm_backbone_frozen_forward_matches_no_grad_eval -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_temporal_classifier.py
git commit -m "test(smokeynet-adapted): frozen TimmBackbone equivalence regression"
```

---

## Task 6: Thread `finetune` through `TemporalSmokeClassifier`

**Files:**
- Modify: `src/bbox_tube_temporal/temporal_classifier.py`
- Modify: `tests/test_temporal_classifier.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_temporal_classifier.py`:

```python
def test_classifier_finetune_mode_exposes_backbone_params_as_trainable():
    clf = TemporalSmokeClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
        finetune=True,
        finetune_last_n_blocks=1,
    )
    trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
    assert any(".layer4." in n for n in trainable), trainable
    assert any(n.startswith("head.") for n in trainable)
    # Earlier backbone layers must still be frozen.
    assert not any(".layer1." in n for n in trainable)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_temporal_classifier.py::test_classifier_finetune_mode_exposes_backbone_params_as_trainable -v`
Expected: FAIL — `TemporalSmokeClassifier.__init__` does not accept `finetune`.

- [ ] **Step 3: Update `TemporalSmokeClassifier` signature**

Modify the `TemporalSmokeClassifier.__init__` in `src/bbox_tube_temporal/temporal_classifier.py`:

```python
    def __init__(
        self,
        backbone: str,
        arch: str,
        hidden_dim: int,
        pretrained: bool = True,
        num_layers: int = 1,
        bidirectional: bool = False,
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = TimmBackbone(
            name=backbone,
            pretrained=pretrained,
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
        )
        feat_dim = self.backbone.feat_dim
        if arch == "mean_pool":
            self.head: nn.Module = MeanPoolHead(
                feat_dim=feat_dim, hidden_dim=hidden_dim
            )
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
```

Also update `forward` so it no longer reshapes through `self.backbone(flat)` (this remains the same call shape — no change needed; patches path already works for both frozen and finetune, since `TimmBackbone.forward` handles both).

- [ ] **Step 4: Run full test file**

Run: `uv run pytest tests/test_temporal_classifier.py -v`
Expected: all PASS, including the new test and every pre-existing one.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/temporal_classifier.py tests/test_temporal_classifier.py
git commit -m "feat(smokeynet-adapted): TemporalSmokeClassifier accepts finetune flag"
```

---

## Task 7: Per-group optimizer in `LitTemporalClassifier`

**Files:**
- Modify: `src/bbox_tube_temporal/lit_temporal.py`
- Modify: `tests/test_lit_temporal.py`

- [ ] **Step 1: Write failing test for two-group optimizer**

Append to `tests/test_lit_temporal.py`:

```python
def test_lit_module_optimizer_uses_two_groups_when_finetune():
    lit = LitTemporalClassifier(
        backbone="resnet18",
        arch="gru",
        hidden_dim=32,
        learning_rate=1e-3,
        weight_decay=1e-2,
        pretrained=False,
        num_layers=1,
        bidirectional=False,
        finetune=True,
        finetune_last_n_blocks=1,
        backbone_lr=1e-5,
    )
    opt = lit.configure_optimizers()
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [1e-5, 1e-3]

    # Every trainable param must land in exactly one group.
    trainable = {
        id(p)
        for p in lit.model.parameters()
        if p.requires_grad
    }
    grouped = [id(p) for g in opt.param_groups for p in g["params"]]
    assert set(grouped) == trainable
    assert len(grouped) == len(trainable), "duplicate params across groups"


def test_lit_module_optimizer_stays_single_group_when_frozen():
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
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 1e-3
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_lit_temporal.py -v -k "two_groups or stays_single"`
Expected: FAIL on the two-groups test (`LitTemporalClassifier.__init__` does not accept `finetune`/`backbone_lr`).

- [ ] **Step 3: Update `LitTemporalClassifier`**

Replace the `LitTemporalClassifier.__init__` and `configure_optimizers` in `src/bbox_tube_temporal/lit_temporal.py` with:

```python
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
        finetune: bool = False,
        finetune_last_n_blocks: int = 0,
        backbone_lr: float | None = None,
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
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.finetune = finetune
        self.backbone_lr = backbone_lr
        self._val_preds: list[float] = []
        self._val_labels: list[float] = []
```

```python
    def configure_optimizers(self):
        if not self.finetune:
            head_params = [p for p in self.model.head.parameters() if p.requires_grad]
            return torch.optim.AdamW(
                head_params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        if self.backbone_lr is None:
            raise ValueError(
                "backbone_lr must be set when finetune=True"
            )

        backbone_params = [
            p for p in self.model.backbone.parameters() if p.requires_grad
        ]
        head_params = [p for p in self.model.head.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            [
                {"params": head_params, "lr": self.learning_rate,
                 "weight_decay": self.weight_decay},
                {"params": backbone_params, "lr": self.backbone_lr,
                 "weight_decay": self.weight_decay},
            ]
        )
```

- [ ] **Step 4: Run full test file**

Run: `uv run pytest tests/test_lit_temporal.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bbox_tube_temporal/lit_temporal.py tests/test_lit_temporal.py
git commit -m "feat(smokeynet-adapted): per-group optimizer for finetune mode"
```

---

## Task 8: Thread finetune params through `scripts/train.py`

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Extend config read**

In `scripts/train.py`, update the `LitTemporalClassifier(...)` construction block to pass the new knobs, reading them with safe defaults so existing `train_gru` / `train_mean_pool` configs keep working untouched:

```python
    lit = LitTemporalClassifier(
        backbone=cfg["backbone"],
        arch=cfg["arch"],
        hidden_dim=cfg["hidden_dim"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        pretrained=True,
        num_layers=cfg.get("num_layers", 1),
        bidirectional=cfg.get("bidirectional", False),
        finetune=cfg.get("finetune", False),
        finetune_last_n_blocks=cfg.get("finetune_last_n_blocks", 0),
        backbone_lr=cfg.get("backbone_lr"),
    )
```

- [ ] **Step 2: Run lint**

Run: `uv run ruff check scripts/train.py`
Expected: clean.

- [ ] **Step 3: Run smoke test — existing params still train end-to-end**

Run: `uv run pytest tests/test_lit_temporal.py tests/test_temporal_classifier.py -v`
Expected: all PASS.

(We do not run `uv run dvc repro train_gru` here; actual training is expensive and covered by the full pipeline at the end.)

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat(smokeynet-adapted): train.py passes finetune params to lit module"
```

---

## Task 9: Create `scripts/compare_variants.py` — FP@recall + summary table

**Files:**
- Create: `scripts/compare_variants.py`
- Create: `tests/test_compare_variants.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_compare_variants.py`:

```python
"""Tests for the variant comparison reporting script."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import compare_variants  # noqa: E402 - test-only path setup


def _preds(pairs: list[tuple[float, int]]) -> list[dict]:
    return [
        {
            "sequence_id": f"seq_{i}",
            "truth": t,
            "prob": p,
            "predicted": int(p > 0.5),
            "correct": (p > 0.5) == bool(t),
        }
        for i, (p, t) in enumerate(pairs)
    ]


def test_fp_at_target_recall_basic():
    # 2 pos + 3 neg, sorted desc:
    # prob 0.9 pos, 0.8 neg, 0.7 pos, 0.6 neg, 0.5 neg
    # @ recall=1.0 we accept through prob 0.7 -> TP=2, FP=1
    preds = _preds([(0.9, 1), (0.8, 0), (0.7, 1), (0.6, 0), (0.5, 0)])
    assert compare_variants.fp_at_recall(preds, 1.0) == 1
    # @ recall>=0.5 we stop at first pos -> FP=0
    assert compare_variants.fp_at_recall(preds, 0.5) == 0


def test_fp_at_target_recall_all_negatives_returns_none():
    preds = _preds([(0.9, 0), (0.5, 0)])
    assert compare_variants.fp_at_recall(preds, 0.9) is None


def test_summarize_writes_markdown_with_expected_columns(tmp_path: Path):
    v1 = tmp_path / "gru"
    v1.mkdir()
    (v1 / "predictions.json").write_text(
        json.dumps(_preds([(0.9, 1), (0.8, 0), (0.7, 1), (0.6, 0)]))
    )
    (v1 / "metrics.json").write_text(
        json.dumps(
            {
                "f1": 0.9,
                "pr_auc": 0.95,
                "roc_auc": 0.96,
            }
        )
    )
    v2 = tmp_path / "convnext"
    v2.mkdir()
    (v2 / "predictions.json").write_text(
        json.dumps(_preds([(0.95, 1), (0.7, 1), (0.5, 0)]))
    )
    (v2 / "metrics.json").write_text(
        json.dumps({"f1": 0.92, "pr_auc": 0.96, "roc_auc": 0.97})
    )

    output = tmp_path / "comparison.md"
    compare_variants.summarize(
        variant_dirs=[v1, v2], output_path=output
    )

    text = output.read_text()
    assert "| variant " in text
    assert "F1 @ 0.5" in text
    assert "FP @ recall 0.90" in text
    assert "FP @ recall 0.95" in text
    assert "FP @ recall 0.97" in text
    assert "FP @ recall 0.99" in text
    assert "| gru " in text
    assert "| convnext " in text


def test_summarize_raises_on_empty(tmp_path: Path):
    output = tmp_path / "comparison.md"
    with pytest.raises(ValueError, match="no variant"):
        compare_variants.summarize(variant_dirs=[], output_path=output)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_compare_variants.py -v`
Expected: FAIL — `compare_variants` module does not exist.

- [ ] **Step 3: Implement `scripts/compare_variants.py`**

Create `scripts/compare_variants.py`:

```python
"""Aggregate per-variant evaluate outputs into a Markdown comparison table.

For each ``<variant_dir>`` it reads ``predictions.json`` and ``metrics.json``
and emits a single Markdown table keyed by variant name (directory
basename). The table reports F1 at threshold 0.5, PR-AUC, ROC-AUC, and FP
count at target recalls 0.90 / 0.95 / 0.97 / 0.99.

Noise-floor interpretation is a prose rule documented in
``docs/specs/2026-04-14-performance-improvements-design.md``: a variant
must beat the baseline mean by more than the seed-to-seed spread on FP
count at target recall to count as signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGET_RECALLS = (0.90, 0.95, 0.97, 0.99)


def fp_at_recall(predictions: list[dict], target_recall: float) -> int | None:
    """Return the smallest FP count at which cumulative recall >= target.

    Sweeps probability thresholds from high to low and stops at the first
    threshold whose recall meets the target. Returns ``None`` when the
    prediction set has no positives or the target recall cannot be
    reached.
    """
    pairs = sorted(
        ((p["prob"], int(p["truth"])) for p in predictions),
        reverse=True,
    )
    n_pos = sum(1 for _, t in pairs if t == 1)
    if n_pos == 0:
        return None
    tp = 0
    fp = 0
    for _, truth in pairs:
        if truth == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        if recall >= target_recall:
            return fp
    return None


def summarize(variant_dirs: list[Path], output_path: Path) -> None:
    if not variant_dirs:
        raise ValueError("summarize requires at least one variant directory")

    headers = [
        "variant",
        "F1 @ 0.5",
        "PR-AUC",
        "ROC-AUC",
        *[f"FP @ recall {r:.2f}" for r in TARGET_RECALLS],
    ]
    rows: list[list[str]] = []
    for variant_dir in variant_dirs:
        name = variant_dir.name
        preds_path = variant_dir / "predictions.json"
        metrics_path = variant_dir / "metrics.json"
        predictions = json.loads(preds_path.read_text())
        metrics = json.loads(metrics_path.read_text())

        row: list[str] = [
            name,
            f"{metrics.get('f1', float('nan')):.3f}",
            f"{metrics.get('pr_auc', float('nan')):.3f}",
            f"{metrics.get('roc_auc', float('nan')):.3f}",
        ]
        for target in TARGET_RECALLS:
            fp = fp_at_recall(predictions, target)
            row.append("—" if fp is None else str(fp))
        rows.append(row)

    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    prose = (
        "# Variant comparison\n\n"
        "> A variant must beat the baseline mean by more than the seed-to-seed "
        "spread on FP count at target recall to count as signal. The "
        "`train_gru`, `train_gru_seed43`, and `train_gru_seed44` rows below "
        "provide that spread.\n\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prose + "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing predictions.json + metrics.json. Repeatable.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    summarize(variant_dirs=args.variant_dir, output_path=args.output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify green**

Run: `uv run pytest tests/test_compare_variants.py -v`
Expected: all PASS.

- [ ] **Step 5: Run lint**

Run: `uv run ruff check scripts/compare_variants.py tests/test_compare_variants.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/compare_variants.py tests/test_compare_variants.py
git commit -m "feat(smokeynet-adapted): compare_variants reporting script"
```

---

## Task 10: Add new variant sections to `params.yaml`

**Files:**
- Modify: `params.yaml`

- [ ] **Step 1: Append the five new variant blocks**

Append to `params.yaml` (after the existing `train_gru:` block):

```yaml
train_gru_seed43:
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
  seed: 43

train_gru_seed44:
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
  seed: 44

train_gru_convnext:
  arch: gru
  backbone: convnext_tiny
  finetune: false
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

train_gru_finetune:
  arch: gru
  backbone: resnet18
  finetune: true
  finetune_last_n_blocks: 1
  backbone_lr: 0.00001
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

train_gru_convnext_finetune:
  arch: gru
  backbone: convnext_tiny
  finetune: true
  finetune_last_n_blocks: 1
  backbone_lr: 0.00001
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

- [ ] **Step 2: Verify YAML parses**

Run: `uv run python -c "import yaml; yaml.safe_load(open('params.yaml').read())"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add params.yaml
git commit -m "chore(smokeynet-adapted): add variant params for backbone + finetune experiments"
```

---

## Task 11: Wire new train + evaluate stages into `dvc.yaml`

**Files:**
- Modify: `dvc.yaml`

- [ ] **Step 1: Append the new stages**

Append to `dvc.yaml` (after the existing `evaluate_gru:` foreach block and before any commented-out section). Keep the YAML format consistent with existing stages.

```yaml
  train_gru_seed43:
    cmd: >-
      uv run python scripts/train.py
      --arch gru
      --train-dir data/05_model_input/train
      --val-dir data/05_model_input/val
      --output-dir data/06_models/gru_seed43
      --params-path params.yaml
      --params-key train_gru_seed43
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - data/05_model_input/train
      - data/05_model_input/val
    params:
      - train_gru_seed43
    outs:
      - data/06_models/gru_seed43/best_checkpoint.pt
    plots:
      - data/06_models/gru_seed43/csv_logs/

  train_gru_seed44:
    cmd: >-
      uv run python scripts/train.py
      --arch gru
      --train-dir data/05_model_input/train
      --val-dir data/05_model_input/val
      --output-dir data/06_models/gru_seed44
      --params-path params.yaml
      --params-key train_gru_seed44
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - data/05_model_input/train
      - data/05_model_input/val
    params:
      - train_gru_seed44
    outs:
      - data/06_models/gru_seed44/best_checkpoint.pt
    plots:
      - data/06_models/gru_seed44/csv_logs/

  train_gru_convnext:
    cmd: >-
      uv run python scripts/train.py
      --arch gru
      --train-dir data/05_model_input/train
      --val-dir data/05_model_input/val
      --output-dir data/06_models/gru_convnext
      --params-path params.yaml
      --params-key train_gru_convnext
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - data/05_model_input/train
      - data/05_model_input/val
    params:
      - train_gru_convnext
    outs:
      - data/06_models/gru_convnext/best_checkpoint.pt
    plots:
      - data/06_models/gru_convnext/csv_logs/

  train_gru_finetune:
    cmd: >-
      uv run python scripts/train.py
      --arch gru
      --train-dir data/05_model_input/train
      --val-dir data/05_model_input/val
      --output-dir data/06_models/gru_finetune
      --params-path params.yaml
      --params-key train_gru_finetune
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - data/05_model_input/train
      - data/05_model_input/val
    params:
      - train_gru_finetune
    outs:
      - data/06_models/gru_finetune/best_checkpoint.pt
    plots:
      - data/06_models/gru_finetune/csv_logs/

  train_gru_convnext_finetune:
    cmd: >-
      uv run python scripts/train.py
      --arch gru
      --train-dir data/05_model_input/train
      --val-dir data/05_model_input/val
      --output-dir data/06_models/gru_convnext_finetune
      --params-path params.yaml
      --params-key train_gru_convnext_finetune
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - data/05_model_input/train
      - data/05_model_input/val
    params:
      - train_gru_convnext_finetune
    outs:
      - data/06_models/gru_convnext_finetune/best_checkpoint.pt
    plots:
      - data/06_models/gru_convnext_finetune/csv_logs/
```

- [ ] **Step 2: Append matching evaluate stages**

Append immediately after the train stages above:

```yaml
  evaluate_gru_seed43:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/evaluate.py
        --arch gru
        --data-dir data/05_model_input/${item}
        --checkpoint data/06_models/gru_seed43/best_checkpoint.pt
        --output-dir data/08_reporting/${item}/gru_seed43
        --params-path params.yaml
        --params-key train_gru_seed43
        --render-tubes-dir data/08_reporting/tubes/${item}
      deps:
        - scripts/evaluate.py
        - src/bbox_tube_temporal/temporal_classifier.py
        - src/bbox_tube_temporal/lit_temporal.py
        - src/bbox_tube_temporal/dataset.py
        - data/06_models/gru_seed43/best_checkpoint.pt
        - data/05_model_input/${item}
        - data/08_reporting/tubes/${item}
      params:
        - train_gru_seed43
      outs:
        - data/08_reporting/${item}/gru_seed43/errors
        - data/08_reporting/${item}/gru_seed43/predictions.json:
            cache: false
      metrics:
        - data/08_reporting/${item}/gru_seed43/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/gru_seed43/pr_curve.png
        - data/08_reporting/${item}/gru_seed43/roc_curve.png
        - data/08_reporting/${item}/gru_seed43/confusion_matrix.png
        - data/08_reporting/${item}/gru_seed43/confusion_matrix_normalized.png

  evaluate_gru_seed44:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/evaluate.py
        --arch gru
        --data-dir data/05_model_input/${item}
        --checkpoint data/06_models/gru_seed44/best_checkpoint.pt
        --output-dir data/08_reporting/${item}/gru_seed44
        --params-path params.yaml
        --params-key train_gru_seed44
        --render-tubes-dir data/08_reporting/tubes/${item}
      deps:
        - scripts/evaluate.py
        - src/bbox_tube_temporal/temporal_classifier.py
        - src/bbox_tube_temporal/lit_temporal.py
        - src/bbox_tube_temporal/dataset.py
        - data/06_models/gru_seed44/best_checkpoint.pt
        - data/05_model_input/${item}
        - data/08_reporting/tubes/${item}
      params:
        - train_gru_seed44
      outs:
        - data/08_reporting/${item}/gru_seed44/errors
        - data/08_reporting/${item}/gru_seed44/predictions.json:
            cache: false
      metrics:
        - data/08_reporting/${item}/gru_seed44/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/gru_seed44/pr_curve.png
        - data/08_reporting/${item}/gru_seed44/roc_curve.png
        - data/08_reporting/${item}/gru_seed44/confusion_matrix.png
        - data/08_reporting/${item}/gru_seed44/confusion_matrix_normalized.png

  evaluate_gru_convnext:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/evaluate.py
        --arch gru
        --data-dir data/05_model_input/${item}
        --checkpoint data/06_models/gru_convnext/best_checkpoint.pt
        --output-dir data/08_reporting/${item}/gru_convnext
        --params-path params.yaml
        --params-key train_gru_convnext
        --render-tubes-dir data/08_reporting/tubes/${item}
      deps:
        - scripts/evaluate.py
        - src/bbox_tube_temporal/temporal_classifier.py
        - src/bbox_tube_temporal/lit_temporal.py
        - src/bbox_tube_temporal/dataset.py
        - data/06_models/gru_convnext/best_checkpoint.pt
        - data/05_model_input/${item}
        - data/08_reporting/tubes/${item}
      params:
        - train_gru_convnext
      outs:
        - data/08_reporting/${item}/gru_convnext/errors
        - data/08_reporting/${item}/gru_convnext/predictions.json:
            cache: false
      metrics:
        - data/08_reporting/${item}/gru_convnext/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/gru_convnext/pr_curve.png
        - data/08_reporting/${item}/gru_convnext/roc_curve.png
        - data/08_reporting/${item}/gru_convnext/confusion_matrix.png
        - data/08_reporting/${item}/gru_convnext/confusion_matrix_normalized.png

  evaluate_gru_finetune:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/evaluate.py
        --arch gru
        --data-dir data/05_model_input/${item}
        --checkpoint data/06_models/gru_finetune/best_checkpoint.pt
        --output-dir data/08_reporting/${item}/gru_finetune
        --params-path params.yaml
        --params-key train_gru_finetune
        --render-tubes-dir data/08_reporting/tubes/${item}
      deps:
        - scripts/evaluate.py
        - src/bbox_tube_temporal/temporal_classifier.py
        - src/bbox_tube_temporal/lit_temporal.py
        - src/bbox_tube_temporal/dataset.py
        - data/06_models/gru_finetune/best_checkpoint.pt
        - data/05_model_input/${item}
        - data/08_reporting/tubes/${item}
      params:
        - train_gru_finetune
      outs:
        - data/08_reporting/${item}/gru_finetune/errors
        - data/08_reporting/${item}/gru_finetune/predictions.json:
            cache: false
      metrics:
        - data/08_reporting/${item}/gru_finetune/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/gru_finetune/pr_curve.png
        - data/08_reporting/${item}/gru_finetune/roc_curve.png
        - data/08_reporting/${item}/gru_finetune/confusion_matrix.png
        - data/08_reporting/${item}/gru_finetune/confusion_matrix_normalized.png

  evaluate_gru_convnext_finetune:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/evaluate.py
        --arch gru
        --data-dir data/05_model_input/${item}
        --checkpoint data/06_models/gru_convnext_finetune/best_checkpoint.pt
        --output-dir data/08_reporting/${item}/gru_convnext_finetune
        --params-path params.yaml
        --params-key train_gru_convnext_finetune
        --render-tubes-dir data/08_reporting/tubes/${item}
      deps:
        - scripts/evaluate.py
        - src/bbox_tube_temporal/temporal_classifier.py
        - src/bbox_tube_temporal/lit_temporal.py
        - src/bbox_tube_temporal/dataset.py
        - data/06_models/gru_convnext_finetune/best_checkpoint.pt
        - data/05_model_input/${item}
        - data/08_reporting/tubes/${item}
      params:
        - train_gru_convnext_finetune
      outs:
        - data/08_reporting/${item}/gru_convnext_finetune/errors
        - data/08_reporting/${item}/gru_convnext_finetune/predictions.json:
            cache: false
      metrics:
        - data/08_reporting/${item}/gru_convnext_finetune/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/gru_convnext_finetune/pr_curve.png
        - data/08_reporting/${item}/gru_convnext_finetune/roc_curve.png
        - data/08_reporting/${item}/gru_convnext_finetune/confusion_matrix.png
        - data/08_reporting/${item}/gru_convnext_finetune/confusion_matrix_normalized.png
```

- [ ] **Step 3: Append `compare_variants` stage (val only; train comparison not needed)**

```yaml
  compare_variants:
    cmd: >-
      uv run python scripts/compare_variants.py
      --variant-dir data/08_reporting/val/gru
      --variant-dir data/08_reporting/val/gru_seed43
      --variant-dir data/08_reporting/val/gru_seed44
      --variant-dir data/08_reporting/val/gru_convnext
      --variant-dir data/08_reporting/val/gru_finetune
      --variant-dir data/08_reporting/val/gru_convnext_finetune
      --output-path data/08_reporting/comparison.md
    deps:
      - scripts/compare_variants.py
      - data/08_reporting/val/gru/predictions.json
      - data/08_reporting/val/gru/metrics.json
      - data/08_reporting/val/gru_seed43/predictions.json
      - data/08_reporting/val/gru_seed43/metrics.json
      - data/08_reporting/val/gru_seed44/predictions.json
      - data/08_reporting/val/gru_seed44/metrics.json
      - data/08_reporting/val/gru_convnext/predictions.json
      - data/08_reporting/val/gru_convnext/metrics.json
      - data/08_reporting/val/gru_finetune/predictions.json
      - data/08_reporting/val/gru_finetune/metrics.json
      - data/08_reporting/val/gru_convnext_finetune/predictions.json
      - data/08_reporting/val/gru_convnext_finetune/metrics.json
    outs:
      - data/08_reporting/comparison.md:
          cache: false
```

- [ ] **Step 4: Verify DVC parses the file**

Run: `uv run dvc dag 2>&1 | head -40`
Expected: prints a DAG listing the new stages; no parse error. (Actual execution not required at this step.)

- [ ] **Step 5: Commit**

```bash
git add dvc.yaml
git commit -m "feat(smokeynet-adapted): dvc stages for backbone + finetune variants and comparison"
```

---

## Task 12: Full test + lint sweep before running experiments

**Files:** none (verification only)

- [ ] **Step 1: Lint**

Run: `make lint`
Expected: clean.

- [ ] **Step 2: Full test suite**

Run: `make test`
Expected: all PASS.

- [ ] **Step 3: DVC pipeline syntax check**

Run: `uv run dvc status` and `uv run dvc dag 2>&1 | tail -20`
Expected: lists all new stages; no parse error.

- [ ] **Step 4: Commit if anything drifted (no code change expected)**

If any formatter made changes:

```bash
git add -u
git commit -m "chore(smokeynet-adapted): lint fixups after variant wiring"
```

Otherwise skip. The branch is now ready for a human to kick off `uv run dvc repro train_gru_convnext train_gru_seed43 train_gru_seed44 train_gru_finetune`, review `comparison.md`, and conditionally run `train_gru_convnext_finetune` depending on Exp 1 + Exp 2 outcome (the "go/no-go" prose rule in the spec).

---

## Out of scope (deferred, flagged in spec)

- Transformer temporal head
- Hard-negative oversampling from the 4 recurring-FP camera sites
- Data augmentation

These can each become their own spec + plan cycle after this work lands.
