# ViT + Temporal Transformer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new variants to the `bbox-tube-temporal` experiment that swap the CNN backbone for a pretrained ViT and the GRU/mean-pool temporal head for a small transformer encoder.

**Architecture:** Factorized ViT-per-frame → learnable `[CLS]` + learned positional embeddings → 2-layer transformer encoder → `Linear` → scalar logit. Plugs into the existing `TemporalSmokeClassifier` / `LitTemporalClassifier` / `scripts/train.py` / DVC pipeline.

**Tech Stack:** PyTorch, PyTorch Lightning, `timm` (ViT-S/14 DINOv2 + ViT-S/16 IN21k), DVC.

**Spec:** `docs/specs/2026-04-15-vit-temporal-transformer-design.md`.

---

## File Structure

**Modify:**
- `src/bbox_tube_temporal/temporal_classifier.py` — add `TransformerHead`, extend `TimmBackbone` for ViT (new `global_pool` kwarg + ViT branch in `_unfreeze_last_n_blocks`), add `"transformer"` branch to `TemporalSmokeClassifier.__init__`.
- `src/bbox_tube_temporal/lit_temporal.py` — add transformer-head kwargs and pass them through.
- `scripts/train.py` — add `"transformer"` to `--arch` choices, pass new kwargs from cfg.
- `scripts/evaluate.py` — add `"transformer"` to `--arch` choices, pass new kwargs from cfg.
- `params.yaml` — three new `train_vit_*` sections.
- `dvc.yaml` — three new `train_vit_*` stages, three new `evaluate_vit_*` foreach stages, update `compare_variants` inputs.
- `tests/test_temporal_classifier.py` — new tests for `TransformerHead`, classifier forward shape for ViT arch, frozen/finetune param-trainability, positional-mask honored; rewrite `test_timm_backbone_finetune_unsupported_family_raises` to target a family that is still unsupported (ViT becomes supported).
- `tests/test_reproducibility.py` — parametrize or add a second test function exercising `arch="transformer"` with a ViT backbone.
- `tests/test_model_parity.py` — extend coverage to the transformer classifier.

**No new files.**

---

## Task 1: Extend `TimmBackbone` for ViT

**Files:**
- Modify: `src/bbox_tube_temporal/temporal_classifier.py:9-88`
- Modify: `tests/test_temporal_classifier.py:211-223` (rewrite the "unsupported family" test to use a non-ViT family)
- Modify: `tests/test_temporal_classifier.py` (append new tests)

### Background

The current `TimmBackbone` hardcodes `global_pool="avg"` (fine for CNNs; for ViTs this averages patch tokens). The spec calls for the CLS-token embedding, so we add a `global_pool` kwarg. `_unfreeze_last_n_blocks` currently raises for ViT; we add a ViT branch.

A pre-existing test asserts that constructing a finetune `TimmBackbone` for `vit_small_patch16_224` raises `NotImplementedError`. Once ViT is supported, that test is wrong. We rewrite it to target `efficientnet_b0`, which still has no unfreeze rule and thus preserves the "unsupported family error is informative" assertion for any *future* unsupported family.

### Steps

- [ ] **Step 1.1: Write failing test for ViT-avg CLS forward shape**

Append to `tests/test_temporal_classifier.py`:

```python
def test_timm_backbone_vit_token_pool_returns_cls_embedding():
    bb = TimmBackbone(
        name="vit_small_patch16_224",
        pretrained=False,
        global_pool="token",
    )
    x = torch.randn(2, 3, 224, 224)
    out = bb(x)
    assert out.shape == (2, bb.feat_dim)
    assert bb.feat_dim == 384
```

- [ ] **Step 1.2: Run and confirm it fails**

Run: `uv run pytest tests/test_temporal_classifier.py::test_timm_backbone_vit_token_pool_returns_cls_embedding -v`
Expected: FAIL (TypeError: unexpected keyword argument `global_pool`).

- [ ] **Step 1.3: Add `global_pool` kwarg to `TimmBackbone.__init__`**

In `src/bbox_tube_temporal/temporal_classifier.py`, replace the signature:

```python
def __init__(
    self,
    name: str,
    pretrained: bool = True,
    finetune: bool = False,
    finetune_last_n_blocks: int = 0,
    global_pool: str = "avg",
) -> None:
    super().__init__()
    self.backbone = timm.create_model(
        name, pretrained=pretrained, num_classes=0, global_pool=global_pool
    )
    self.feat_dim: int = self.backbone.num_features
    self.finetune = finetune
    self.name = name
    ...  # rest unchanged
```

- [ ] **Step 1.4: Re-run, confirm pass**

Run: `uv run pytest tests/test_temporal_classifier.py::test_timm_backbone_vit_token_pool_returns_cls_embedding -v`
Expected: PASS.

- [ ] **Step 1.5: Write failing test for ViT finetune unfreeze-last-N**

Append:

```python
def test_timm_backbone_finetune_vit_s16_unfreezes_only_last_block():
    bb = TimmBackbone(
        name="vit_small_patch16_224",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
        global_pool="token",
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names, "expected some trainable params"
    # timm's ViT wraps blocks under `blocks.<i>.*`; last block is index 11 for ViT-S.
    assert all(".blocks.11." in n for n in trainable_names), trainable_names


def test_timm_backbone_finetune_vit_s16_n2_unfreezes_last_two_blocks():
    bb = TimmBackbone(
        name="vit_small_patch16_224",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=2,
        global_pool="token",
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all(
        (".blocks.10." in n) or (".blocks.11." in n) for n in trainable_names
    ), trainable_names
```

- [ ] **Step 1.6: Run, confirm they fail**

Run: `uv run pytest tests/test_temporal_classifier.py -k vit_s16 -v`
Expected: FAIL (`NotImplementedError: finetune=True is not implemented for backbone family 'vit_small_patch16_224'`).

- [ ] **Step 1.7: Add ViT branch to `_unfreeze_last_n_blocks`**

In `src/bbox_tube_temporal/temporal_classifier.py`, extend `_unfreeze_last_n_blocks`:

```python
def _unfreeze_last_n_blocks(self, n: int) -> None:
    if n <= 0:
        return
    name = self.name
    if name.startswith("resnet"):
        stages = [getattr(self.backbone, f"layer{i}") for i in range(1, 5)]
    elif name.startswith("convnext"):
        stages = list(self.backbone.stages)
    elif name.startswith("vit_"):
        # timm's ViT exposes a `blocks` ModuleList of transformer blocks.
        stages = list(self.backbone.blocks)
    else:
        stage_names = [n_ for n_, _ in self.backbone.named_children()]
        raise NotImplementedError(
            f"finetune=True is not implemented for backbone family "
            f"{name!r}. Top-level children: {stage_names}. Add an "
            f"explicit unfreeze rule in TimmBackbone._unfreeze_last_n_blocks."
        )
    for stage in stages[-n:]:
        for p in stage.parameters():
            p.requires_grad = True
```

- [ ] **Step 1.8: Re-run, confirm pass**

Run: `uv run pytest tests/test_temporal_classifier.py -k vit_s16 -v`
Expected: PASS.

- [ ] **Step 1.9: Rewrite the unsupported-family test**

Replace `test_timm_backbone_finetune_unsupported_family_raises` in `tests/test_temporal_classifier.py`:

```python
def test_timm_backbone_finetune_unsupported_family_raises():
    # efficientnet_b0 has no explicit unfreeze rule; any future unsupported
    # family should raise the same informative error.
    with pytest.raises(NotImplementedError) as exc:
        TimmBackbone(
            name="efficientnet_b0",
            pretrained=False,
            finetune=True,
            finetune_last_n_blocks=1,
        )
    msg = str(exc.value)
    assert "efficientnet_b0" in msg
    assert "children" in msg.lower()
```

- [ ] **Step 1.10: Run full temporal_classifier test file**

Run: `uv run pytest tests/test_temporal_classifier.py -v`
Expected: all pass (original CNN tests + new ViT tests + rewritten unsupported-family test).

- [ ] **Step 1.11: Also exercise DINOv2 (patch-14) backbone name**

Append:

```python
def test_timm_backbone_vit_s14_dinov2_finetune_unfreezes_last_block():
    bb = TimmBackbone(
        name="vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        finetune=True,
        finetune_last_n_blocks=1,
        global_pool="token",
    )
    trainable_names = [n for n, p in bb.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all(".blocks.11." in n for n in trainable_names), trainable_names
```

Run: `uv run pytest tests/test_temporal_classifier.py::test_timm_backbone_vit_s14_dinov2_finetune_unfreezes_last_block -v`
Expected: PASS (same `vit_` prefix branch handles it).

- [ ] **Step 1.12: Commit**

```bash
git add src/bbox_tube_temporal/temporal_classifier.py tests/test_temporal_classifier.py
git commit -m "feat(bbox-tube-temporal): TimmBackbone supports ViT finetune + global_pool"
```

---

## Task 2: `TransformerHead` + classifier dispatch

**Files:**
- Modify: `src/bbox_tube_temporal/temporal_classifier.py` (add `TransformerHead`, new arch branch in `TemporalSmokeClassifier`)
- Modify: `tests/test_temporal_classifier.py` (new head + classifier tests)

### Design

```
feats: (B, T, D)  mask: (B, T) bool   # True = real frame
   |
   v
prepend learnable [CLS] token (1, D), broadcast to (B, 1, D)
cat -> (B, T+1, D)
+ learned positional embedding of shape (max_frames+1, D) slice [:T+1]
   |
   v
nn.TransformerEncoder(num_layers, d_model=D, nhead, ffn_dim=4*D, dropout, norm_first=True)
  with src_key_padding_mask of shape (B, T+1); CLS position is never masked.
   |
   v
take output at index 0 (CLS)  -> (B, D)
Linear(D, 1) -> squeeze(-1)   -> (B,)
```

### Steps

- [ ] **Step 2.1: Write failing test for `TransformerHead` output shape**

Append to `tests/test_temporal_classifier.py`:

```python
from bbox_tube_temporal.temporal_classifier import TransformerHead  # add at top


def test_transformer_head_returns_logits_per_batch():
    head = TransformerHead(
        feat_dim=384,
        num_layers=2,
        num_heads=6,
        ffn_dim=1536,
        dropout=0.0,
        max_frames=20,
    )
    feats = torch.randn(3, 20, 384)
    mask = torch.ones(3, 20, dtype=torch.bool)
    logits = head(feats, mask)
    assert logits.shape == (3,)
```

- [ ] **Step 2.2: Run, confirm ImportError**

Run: `uv run pytest tests/test_temporal_classifier.py::test_transformer_head_returns_logits_per_batch -v`
Expected: FAIL (ImportError).

- [ ] **Step 2.3: Implement `TransformerHead`**

Append to `src/bbox_tube_temporal/temporal_classifier.py` (after `GRUHead`):

```python
class TransformerHead(nn.Module):
    """Learnable [CLS] + learned positional embeddings + Transformer encoder.

    Returns one logit per tube. Padded positions are masked out of
    attention via ``src_key_padding_mask``.
    """

    def __init__(
        self,
        feat_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        max_frames: int,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # +1 for the prepended CLS position.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 1, feat_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(feat_dim, 1)
        self.max_frames = max_frames

    def forward(self, feats: Tensor, mask: Tensor) -> Tensor:
        b, t, _ = feats.shape
        if t > self.max_frames:
            raise ValueError(
                f"TransformerHead received T={t} frames but was configured "
                f"with max_frames={self.max_frames}"
            )
        cls = self.cls_token.expand(b, 1, -1)
        x = torch.cat([cls, feats], dim=1)  # (B, T+1, D)
        x = x + self.pos_embed[:, : t + 1, :]
        # True = pad (ignored) per torch convention; our input mask is True = real.
        cls_real = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
        real_mask = torch.cat([cls_real, mask], dim=1)
        key_padding_mask = ~real_mask  # (B, T+1), True = pad
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls_out = out[:, 0, :]
        return self.classifier(cls_out).squeeze(-1)
```

- [ ] **Step 2.4: Confirm the shape test passes**

Run: `uv run pytest tests/test_temporal_classifier.py::test_transformer_head_returns_logits_per_batch -v`
Expected: PASS.

- [ ] **Step 2.5: Write failing test for mask invariance**

Append:

```python
def test_transformer_head_respects_mask():
    torch.manual_seed(0)
    head = TransformerHead(
        feat_dim=16,
        num_layers=1,
        num_heads=2,
        ffn_dim=32,
        dropout=0.0,
        max_frames=20,
    )
    head.eval()
    real = torch.randn(2, 16)
    a = torch.zeros(20, 16)
    a[:2] = real
    b = a.clone()
    b[2:] = 1e3  # junk in padded positions
    feats = torch.stack([a, b])
    mask = torch.zeros(2, 20, dtype=torch.bool)
    mask[:, :2] = True
    logits = head(feats, mask)
    assert torch.allclose(logits[0], logits[1], atol=1e-4), logits
```

- [ ] **Step 2.6: Run, confirm pass (should pass because mask is honored)**

Run: `uv run pytest tests/test_temporal_classifier.py::test_transformer_head_respects_mask -v`
Expected: PASS.

- [ ] **Step 2.7: Write failing test for `arch="transformer"` in `TemporalSmokeClassifier`**

Append:

```python
def test_classifier_transformer_forward_shape_vit_backbone():
    clf = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        arch="transformer",
        hidden_dim=64,  # unused for transformer; kept for API symmetry
        pretrained=False,
        transformer_num_layers=2,
        transformer_num_heads=6,
        transformer_ffn_dim=1536,
        transformer_dropout=0.0,
        max_frames=20,
        global_pool="token",
    )
    patches = torch.randn(2, 5, 3, 224, 224)
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = clf(patches, mask)
    assert logits.shape == (2,)


def test_classifier_transformer_frozen_only_head_trainable():
    clf = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        arch="transformer",
        hidden_dim=64,
        pretrained=False,
        transformer_num_layers=2,
        transformer_num_heads=6,
        transformer_ffn_dim=1536,
        transformer_dropout=0.0,
        max_frames=20,
        global_pool="token",
    )
    trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
    assert all(n.startswith("head.") for n in trainable), trainable


def test_classifier_transformer_finetune_exposes_last_vit_block():
    clf = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        arch="transformer",
        hidden_dim=64,
        pretrained=False,
        transformer_num_layers=2,
        transformer_num_heads=6,
        transformer_ffn_dim=1536,
        transformer_dropout=0.0,
        max_frames=20,
        global_pool="token",
        finetune=True,
        finetune_last_n_blocks=1,
    )
    trainable = [n for n, p in clf.named_parameters() if p.requires_grad]
    assert any(".blocks.11." in n for n in trainable), trainable
    assert any(n.startswith("head.") for n in trainable)
    assert not any(".blocks.0." in n for n in trainable)
```

- [ ] **Step 2.8: Run, confirm TypeError**

Run: `uv run pytest tests/test_temporal_classifier.py -k classifier_transformer -v`
Expected: FAIL (TypeError: unexpected kwargs).

- [ ] **Step 2.9: Extend `TemporalSmokeClassifier.__init__`**

Replace `TemporalSmokeClassifier.__init__` in `src/bbox_tube_temporal/temporal_classifier.py`:

```python
class TemporalSmokeClassifier(nn.Module):
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
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 6,
        transformer_ffn_dim: int = 1536,
        transformer_dropout: float = 0.1,
        max_frames: int = 20,
        global_pool: str = "avg",
    ) -> None:
        super().__init__()
        self.backbone = TimmBackbone(
            name=backbone,
            pretrained=pretrained,
            finetune=finetune,
            finetune_last_n_blocks=finetune_last_n_blocks,
            global_pool=global_pool,
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
        elif arch == "transformer":
            self.head = TransformerHead(
                feat_dim=feat_dim,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                ffn_dim=transformer_ffn_dim,
                dropout=transformer_dropout,
                max_frames=max_frames,
            )
        else:
            raise ValueError(
                f"unknown arch: {arch!r} "
                f"(expected 'mean_pool', 'gru', or 'transformer')"
            )
        self.arch = arch
```

Update the existing `test_classifier_unknown_arch_raises` expected message check only if it used a literal string — re-read and adjust. The test currently uses `match="arch"` which still matches the new message. No change needed.

- [ ] **Step 2.10: Re-run, confirm pass**

Run: `uv run pytest tests/test_temporal_classifier.py -v`
Expected: all pass.

- [ ] **Step 2.11: Commit**

```bash
git add src/bbox_tube_temporal/temporal_classifier.py tests/test_temporal_classifier.py
git commit -m "feat(bbox-tube-temporal): TransformerHead + 'transformer' arch in classifier"
```

---

## Task 3: Lightning + scripts wiring

**Files:**
- Modify: `src/bbox_tube_temporal/lit_temporal.py`
- Modify: `scripts/train.py`
- Modify: `scripts/evaluate.py`
- Modify: `tests/test_lit_temporal.py` (add ViT-transformer forward test)

### Steps

- [ ] **Step 3.1: Write failing test exercising `LitTemporalClassifier` with `arch="transformer"`**

Read `tests/test_lit_temporal.py` to follow the existing style, then append:

```python
def test_lit_temporal_transformer_forward_shape():
    lit = LitTemporalClassifier(
        backbone="vit_small_patch16_224",
        arch="transformer",
        hidden_dim=64,
        learning_rate=1e-4,
        weight_decay=0.05,
        pretrained=False,
        transformer_num_layers=2,
        transformer_num_heads=6,
        transformer_ffn_dim=1536,
        transformer_dropout=0.0,
        max_frames=20,
        global_pool="token",
    )
    patches = torch.randn(2, 4, 3, 224, 224)
    mask = torch.ones(2, 4, dtype=torch.bool)
    out = lit(patches, mask)
    assert out.shape == (2,)
```

- [ ] **Step 3.2: Run, confirm TypeError**

Run: `uv run pytest tests/test_lit_temporal.py::test_lit_temporal_transformer_forward_shape -v`
Expected: FAIL (TypeError).

- [ ] **Step 3.3: Extend `LitTemporalClassifier.__init__`**

In `src/bbox_tube_temporal/lit_temporal.py`, add new kwargs and pass them through:

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
    transformer_num_layers: int = 2,
    transformer_num_heads: int = 6,
    transformer_ffn_dim: int = 1536,
    transformer_dropout: float = 0.1,
    max_frames: int = 20,
    global_pool: str = "avg",
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
        transformer_num_layers=transformer_num_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_ffn_dim=transformer_ffn_dim,
        transformer_dropout=transformer_dropout,
        max_frames=max_frames,
        global_pool=global_pool,
    )
    # rest of body unchanged
```

- [ ] **Step 3.4: Run, confirm pass**

Run: `uv run pytest tests/test_lit_temporal.py::test_lit_temporal_transformer_forward_shape -v`
Expected: PASS.

- [ ] **Step 3.5: Update `scripts/train.py` arch choices + kwargs**

In `scripts/train.py` replace:

```python
parser.add_argument("--arch", choices=["mean_pool", "gru"], required=True)
```

with:

```python
parser.add_argument(
    "--arch", choices=["mean_pool", "gru", "transformer"], required=True
)
```

And replace the `LitTemporalClassifier(...)` constructor with one that forwards the transformer kwargs from cfg:

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
    transformer_num_layers=cfg.get("transformer_num_layers", 2),
    transformer_num_heads=cfg.get("transformer_num_heads", 6),
    transformer_ffn_dim=cfg.get("transformer_ffn_dim", 1536),
    transformer_dropout=cfg.get("transformer_dropout", 0.1),
    max_frames=cfg.get("max_frames", 20),
    global_pool=cfg.get("global_pool", "avg"),
)
```

- [ ] **Step 3.6: Mirror the same change in `scripts/evaluate.py`**

In `scripts/evaluate.py:62`, replace:

```python
parser.add_argument("--arch", choices=["mean_pool", "gru"], required=True)
```

with:

```python
parser.add_argument(
    "--arch", choices=["mean_pool", "gru", "transformer"], required=True
)
```

And update the `LitTemporalClassifier(...)` construction there (around line 79) to pass the same new kwargs with the same `cfg.get(...)` defaults as Step 3.5.

- [ ] **Step 3.7: Add cosine + warmup scheduler (ViT variants only)**

The spec calls for a cosine schedule with 5 % warmup for ViT variants. Gate it behind flags so existing baselines are unaffected.

In `src/bbox_tube_temporal/lit_temporal.py`, add kwargs `use_cosine_warmup: bool = False` and `warmup_frac: float = 0.05` to `__init__`, store on `self`, then replace `configure_optimizers`:

```python
def configure_optimizers(self):
    if not self.finetune:
        head_params = [p for p in self.model.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            head_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
    else:
        if self.backbone_lr is None:
            raise ValueError("backbone_lr must be set when finetune=True")
        backbone_params = [
            p for p in self.model.backbone.parameters() if p.requires_grad
        ]
        head_params = [p for p in self.model.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": head_params,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": backbone_params,
                    "lr": self.backbone_lr,
                    "weight_decay": self.weight_decay,
                },
            ]
        )

    if not self.use_cosine_warmup:
        return optimizer

    total_steps = int(self.trainer.estimated_stepping_batches)
    warmup_steps = max(1, int(total_steps * self.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }
```

Add `import math` at the top of `lit_temporal.py` if not already present.

Extend `scripts/train.py` (and the evaluate-script Lightning construction, though the scheduler is harmless at eval time) to forward the new kwargs:

```python
use_cosine_warmup=cfg.get("use_cosine_warmup", False),
warmup_frac=cfg.get("warmup_frac", 0.05),
```

Write a quick smoke test in `tests/test_lit_temporal.py`:

```python
def test_lit_temporal_cosine_warmup_returns_scheduler_dict(tmp_path):
    lit = LitTemporalClassifier(
        backbone="resnet18",
        arch="mean_pool",
        hidden_dim=16,
        learning_rate=1e-3,
        weight_decay=1e-2,
        pretrained=False,
        use_cosine_warmup=True,
        warmup_frac=0.1,
    )
    # configure_optimizers reads self.trainer; attach a minimal stub.
    stub = MagicMock()
    stub.estimated_stepping_batches = 100
    lit.trainer = stub
    out = lit.configure_optimizers()
    assert isinstance(out, dict)
    assert "optimizer" in out
    assert "lr_scheduler" in out
    assert out["lr_scheduler"]["interval"] == "step"
```

Add `from unittest.mock import MagicMock` at the top of the file if not already present.

Run: `uv run pytest tests/test_lit_temporal.py::test_lit_temporal_cosine_warmup_returns_scheduler_dict -v`
Expected: PASS.

- [ ] **Step 3.8: Run the full unit test suite to make sure nothing regressed**

Run: `uv run pytest tests/ -v -x --ignore=tests/test_reproducibility.py --ignore=tests/test_model_parity.py`
Expected: all pass. (We exclude slow tests for this quick check; they're exercised in Task 5.)

- [ ] **Step 3.9: Commit**

```bash
git add src/bbox_tube_temporal/lit_temporal.py scripts/train.py scripts/evaluate.py tests/test_lit_temporal.py
git commit -m "feat(bbox-tube-temporal): wire transformer arch + cosine warmup through Lightning + scripts"
```

---

## Task 4: `params.yaml` + `dvc.yaml`

**Files:**
- Modify: `params.yaml`
- Modify: `dvc.yaml`

### Steps

- [ ] **Step 4.1: Add three new variant sections to `params.yaml`**

Append after `train_gru_convnext_base_finetune` and before `augment`:

```yaml
_vit_defaults: &vit_defaults
  <<: *train_defaults
  arch: transformer
  batch_size: 16
  learning_rate: 0.0001
  weight_decay: 0.05
  max_epochs: 30
  global_pool: token
  transformer_num_layers: 2
  transformer_num_heads: 6
  transformer_ffn_dim: 1536
  transformer_dropout: 0.1
  use_cosine_warmup: true
  warmup_frac: 0.05

train_vit_dinov2_frozen:
  <<: *vit_defaults
  backbone: vit_small_patch14_dinov2.lvd142m
  finetune: false

train_vit_dinov2_finetune:
  <<: *vit_defaults
  backbone: vit_small_patch14_dinov2.lvd142m
  finetune: true
  finetune_last_n_blocks: 1
  backbone_lr: 0.00001

train_vit_in21k_finetune:
  <<: *vit_defaults
  backbone: vit_small_patch16_224.augreg_in21k_ft_in1k
  finetune: true
  finetune_last_n_blocks: 1
  backbone_lr: 0.00001
```

- [ ] **Step 4.2: Verify timm resolves the model names**

Run: `uv run python -c "import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False); timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=False); print('ok')"`
Expected: prints `ok`. If either raises `RuntimeError: Unknown model`, use `uv run python -c "import timm; print([n for n in timm.list_models() if 'dinov2' in n.lower()])"` to find the correct timm model name and update `params.yaml` accordingly before proceeding.

- [ ] **Step 4.3: Add three train + three evaluate stages to `dvc.yaml`**

Insert after `train_gru_convnext_base_finetune` block in `dvc.yaml`:

```yaml
  train_vit_dinov2_frozen:
    cmd: >-
      uv run python scripts/train.py
      --arch transformer
      --train-dir data/05_model_input/train
      --val-dir data/05_model_input/val
      --output-dir data/06_models/vit_dinov2_frozen
      --params-path params.yaml
      --params-key train_vit_dinov2_frozen
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/augment.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - src/bbox_tube_temporal/training_plots.py
      - data/05_model_input/train
      - data/05_model_input/val
    params:
      - train_vit_dinov2_frozen
      - augment
    outs:
      - data/06_models/vit_dinov2_frozen/best_checkpoint.pt
      - data/06_models/vit_dinov2_frozen/csv_logs/
    plots:
      - data/06_models/vit_dinov2_frozen/plots/training_curves.png

  train_vit_dinov2_finetune:
    cmd: >-
      uv run python scripts/train.py
      --arch transformer
      --train-dir data/05_model_input/train
      --val-dir data/05_model_input/val
      --output-dir data/06_models/vit_dinov2_finetune
      --params-path params.yaml
      --params-key train_vit_dinov2_finetune
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/augment.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - src/bbox_tube_temporal/training_plots.py
      - data/05_model_input/train
      - data/05_model_input/val
    params:
      - train_vit_dinov2_finetune
      - augment
    outs:
      - data/06_models/vit_dinov2_finetune/best_checkpoint.pt
      - data/06_models/vit_dinov2_finetune/csv_logs/
    plots:
      - data/06_models/vit_dinov2_finetune/plots/training_curves.png

  train_vit_in21k_finetune:
    cmd: >-
      uv run python scripts/train.py
      --arch transformer
      --train-dir data/05_model_input/train
      --val-dir data/05_model_input/val
      --output-dir data/06_models/vit_in21k_finetune
      --params-path params.yaml
      --params-key train_vit_in21k_finetune
    deps:
      - scripts/train.py
      - src/bbox_tube_temporal/augment.py
      - src/bbox_tube_temporal/dataset.py
      - src/bbox_tube_temporal/temporal_classifier.py
      - src/bbox_tube_temporal/lit_temporal.py
      - src/bbox_tube_temporal/training_plots.py
      - data/05_model_input/train
      - data/05_model_input/val
    params:
      - train_vit_in21k_finetune
      - augment
    outs:
      - data/06_models/vit_in21k_finetune/best_checkpoint.pt
      - data/06_models/vit_in21k_finetune/csv_logs/
    plots:
      - data/06_models/vit_in21k_finetune/plots/training_curves.png
```

And insert three matching `evaluate_vit_*` stages before the `package` stage. Each follows the `evaluate_gru_convnext_finetune` template with the per-variant name substituted:

```yaml
  evaluate_vit_dinov2_frozen:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/evaluate.py
        --arch transformer
        --data-dir data/05_model_input/${item}
        --checkpoint data/06_models/vit_dinov2_frozen/best_checkpoint.pt
        --output-dir data/08_reporting/${item}/vit_dinov2_frozen
        --params-path params.yaml
        --params-key train_vit_dinov2_frozen
        --render-tubes-dir data/08_reporting/tubes/${item}
      deps:
        - scripts/evaluate.py
        - src/bbox_tube_temporal/temporal_classifier.py
        - src/bbox_tube_temporal/lit_temporal.py
        - src/bbox_tube_temporal/dataset.py
        - data/06_models/vit_dinov2_frozen/best_checkpoint.pt
        - data/05_model_input/${item}
        - data/08_reporting/tubes/${item}
      params:
        - train_vit_dinov2_frozen
      outs:
        - data/08_reporting/${item}/vit_dinov2_frozen/errors
        - data/08_reporting/${item}/vit_dinov2_frozen/predictions.json:
            cache: false
      metrics:
        - data/08_reporting/${item}/vit_dinov2_frozen/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/vit_dinov2_frozen/pr_curve.png
        - data/08_reporting/${item}/vit_dinov2_frozen/roc_curve.png
        - data/08_reporting/${item}/vit_dinov2_frozen/confusion_matrix.png
        - data/08_reporting/${item}/vit_dinov2_frozen/confusion_matrix_normalized.png

  evaluate_vit_dinov2_finetune:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/evaluate.py
        --arch transformer
        --data-dir data/05_model_input/${item}
        --checkpoint data/06_models/vit_dinov2_finetune/best_checkpoint.pt
        --output-dir data/08_reporting/${item}/vit_dinov2_finetune
        --params-path params.yaml
        --params-key train_vit_dinov2_finetune
        --render-tubes-dir data/08_reporting/tubes/${item}
      deps:
        - scripts/evaluate.py
        - src/bbox_tube_temporal/temporal_classifier.py
        - src/bbox_tube_temporal/lit_temporal.py
        - src/bbox_tube_temporal/dataset.py
        - data/06_models/vit_dinov2_finetune/best_checkpoint.pt
        - data/05_model_input/${item}
        - data/08_reporting/tubes/${item}
      params:
        - train_vit_dinov2_finetune
      outs:
        - data/08_reporting/${item}/vit_dinov2_finetune/errors
        - data/08_reporting/${item}/vit_dinov2_finetune/predictions.json:
            cache: false
      metrics:
        - data/08_reporting/${item}/vit_dinov2_finetune/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/vit_dinov2_finetune/pr_curve.png
        - data/08_reporting/${item}/vit_dinov2_finetune/roc_curve.png
        - data/08_reporting/${item}/vit_dinov2_finetune/confusion_matrix.png
        - data/08_reporting/${item}/vit_dinov2_finetune/confusion_matrix_normalized.png

  evaluate_vit_in21k_finetune:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/evaluate.py
        --arch transformer
        --data-dir data/05_model_input/${item}
        --checkpoint data/06_models/vit_in21k_finetune/best_checkpoint.pt
        --output-dir data/08_reporting/${item}/vit_in21k_finetune
        --params-path params.yaml
        --params-key train_vit_in21k_finetune
        --render-tubes-dir data/08_reporting/tubes/${item}
      deps:
        - scripts/evaluate.py
        - src/bbox_tube_temporal/temporal_classifier.py
        - src/bbox_tube_temporal/lit_temporal.py
        - src/bbox_tube_temporal/dataset.py
        - data/06_models/vit_in21k_finetune/best_checkpoint.pt
        - data/05_model_input/${item}
        - data/08_reporting/tubes/${item}
      params:
        - train_vit_in21k_finetune
      outs:
        - data/08_reporting/${item}/vit_in21k_finetune/errors
        - data/08_reporting/${item}/vit_in21k_finetune/predictions.json:
            cache: false
      metrics:
        - data/08_reporting/${item}/vit_in21k_finetune/metrics.json:
            cache: false
      plots:
        - data/08_reporting/${item}/vit_in21k_finetune/pr_curve.png
        - data/08_reporting/${item}/vit_in21k_finetune/roc_curve.png
        - data/08_reporting/${item}/vit_in21k_finetune/confusion_matrix.png
        - data/08_reporting/${item}/vit_in21k_finetune/confusion_matrix_normalized.png
```

- [ ] **Step 4.4: Extend `compare_variants` stage inputs**

In `dvc.yaml`, update the `compare_variants` stage's `cmd`, `deps`, and (implicit) metric reading so the three new variants are included. Replace the existing stage block with:

```yaml
  compare_variants:
    cmd: >-
      uv run python scripts/compare_variants.py
      --variant-dir data/08_reporting/val/gru
      --variant-dir data/08_reporting/val/gru_convnext
      --variant-dir data/08_reporting/val/gru_finetune
      --variant-dir data/08_reporting/val/gru_convnext_finetune
      --variant-dir data/08_reporting/val/gru_convnext_base_finetune
      --variant-dir data/08_reporting/val/vit_dinov2_frozen
      --variant-dir data/08_reporting/val/vit_dinov2_finetune
      --variant-dir data/08_reporting/val/vit_in21k_finetune
      --output-path data/08_reporting/comparison.md
    deps:
      - scripts/compare_variants.py
      - data/08_reporting/val/gru/predictions.json
      - data/08_reporting/val/gru/metrics.json
      - data/08_reporting/val/gru_convnext/predictions.json
      - data/08_reporting/val/gru_convnext/metrics.json
      - data/08_reporting/val/gru_finetune/predictions.json
      - data/08_reporting/val/gru_finetune/metrics.json
      - data/08_reporting/val/gru_convnext_finetune/predictions.json
      - data/08_reporting/val/gru_convnext_finetune/metrics.json
      - data/08_reporting/val/gru_convnext_base_finetune/predictions.json
      - data/08_reporting/val/gru_convnext_base_finetune/metrics.json
      - data/08_reporting/val/vit_dinov2_frozen/predictions.json
      - data/08_reporting/val/vit_dinov2_frozen/metrics.json
      - data/08_reporting/val/vit_dinov2_finetune/predictions.json
      - data/08_reporting/val/vit_dinov2_finetune/metrics.json
      - data/08_reporting/val/vit_in21k_finetune/predictions.json
      - data/08_reporting/val/vit_in21k_finetune/metrics.json
    outs:
      - data/08_reporting/comparison.md:
          cache: false
```

- [ ] **Step 4.5: Validate `dvc.yaml` parses and stages resolve**

Run: `uv run dvc stage list`
Expected: lists `train_vit_dinov2_frozen`, `train_vit_dinov2_finetune`, `train_vit_in21k_finetune`, `evaluate_vit_dinov2_frozen@train`, `evaluate_vit_dinov2_frozen@val`, ... (six new eval entries). Any parse error points to a YAML typo — fix inline.

- [ ] **Step 4.6: Commit**

```bash
git add params.yaml dvc.yaml
git commit -m "chore(bbox-tube-temporal): DVC stages + params for ViT transformer variants"
```

---

## Task 5: Reproducibility + parity tests

**Files:**
- Modify: `tests/test_reproducibility.py`
- Modify: `tests/test_model_parity.py`

### Steps

- [ ] **Step 5.1: Add a ViT-transformer reproducibility test**

In `tests/test_reproducibility.py`, keep the existing function and add a new one that reuses `_make_split` and `_fit_once` but with the ViT backbone + `arch="transformer"`:

```python
def _fit_once_transformer(seed: int, train_dir: Path, val_dir: Path, log_dir: Path) -> dict:
    L.seed_everything(seed, workers=True)

    train_ds = TubePatchDataset(train_dir, max_frames=5)
    val_ds = TubePatchDataset(val_dir, max_frames=5)
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True,
        num_workers=2, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False,
        num_workers=2, persistent_workers=True,
    )

    lit = LitTemporalClassifier(
        backbone="vit_small_patch16_224",
        arch="transformer",
        hidden_dim=16,
        learning_rate=1e-4,
        weight_decay=5e-2,
        pretrained=False,
        transformer_num_layers=1,
        transformer_num_heads=2,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
        max_frames=5,
        global_pool="token",
    )

    trainer = L.Trainer(
        max_epochs=2, accelerator="cpu", devices=1, deterministic=True,
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        log_every_n_steps=1, default_root_dir=log_dir,
    )
    trainer.fit(lit, train_loader, val_loader)
    return {k: v.detach().clone() for k, v in lit.state_dict().items()}


def test_transformer_training_is_bitwise_reproducible_with_fixed_seed(
    tmp_path: Path,
) -> None:
    train_dir = _make_split(
        tmp_path, "train",
        [("a", 1, 5), ("b", 0, 4), ("c", 1, 3), ("d", 0, 5)],
    )
    val_dir = _make_split(tmp_path, "val", [("e", 1, 4), ("f", 0, 3)])

    run1 = _fit_once_transformer(SEED, train_dir, val_dir, tmp_path / "run1")
    run2 = _fit_once_transformer(SEED, train_dir, val_dir, tmp_path / "run2")
    run_other = _fit_once_transformer(
        OTHER_SEED, train_dir, val_dir, tmp_path / "run_other"
    )

    assert run1.keys() == run2.keys() == run_other.keys()
    for key in run1:
        assert torch.equal(run1[key], run2[key]), (
            f"Same-seed transformer runs diverged at {key!r}"
        )
    differing = [key for key in run1 if not torch.equal(run1[key], run_other[key])]
    assert differing, "Different-seed run produced identical transformer weights"
```

- [ ] **Step 5.2: Run just the new test**

Run: `uv run pytest tests/test_reproducibility.py::test_transformer_training_is_bitwise_reproducible_with_fixed_seed -v`
Expected: PASS. If it fails on same-seed equality, investigate deterministic-algorithm coverage for `nn.TransformerEncoderLayer` (the plan assumes `deterministic=True` already covers it — same mechanism as GRU).

- [ ] **Step 5.3: Extend `test_model_parity.py` with a transformer case**

The existing file at `tests/test_model_parity.py` declares a module-scope CFG, a `classifier` fixture, and a single `test_parity_logit_matches` function. Follow that pattern: add a second CFG, a second fixture, and a second test. Append to `tests/test_model_parity.py`:

```python
CFG_TRANSFORMER: dict = {
    "infer": {"confidence_threshold": 0.01, "iou_nms": 0.2, "image_size": 224},
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
        "patch_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "classifier": {
        "backbone": "vit_small_patch16_224",
        "arch": "transformer",
        "hidden_dim": 32,
        "max_frames": 5,
        "pretrained": False,
        "global_pool": "token",
        "transformer_num_layers": 1,
        "transformer_num_heads": 2,
        "transformer_ffn_dim": 64,
        "transformer_dropout": 0.0,
    },
    "decision": {
        "aggregation": "max_logit",
        "threshold": 0.0,
        "target_recall": 0.95,
        "trigger_rule": "end_of_winner",
    },
}


@pytest.fixture(scope="module")
def transformer_classifier() -> TemporalSmokeClassifier:
    torch.manual_seed(0)
    model = TemporalSmokeClassifier(
        backbone="vit_small_patch16_224",
        arch="transformer",
        hidden_dim=32,
        pretrained=False,
        global_pool="token",
        transformer_num_layers=1,
        transformer_num_heads=2,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
        max_frames=5,
    )
    model.eval()
    return model


def _offline_logit_with_cfg(
    classifier: TemporalSmokeClassifier, cfg: dict
) -> float:
    """Variant of _offline_logit that reads patch_size/normalization from cfg."""
    fdets = load_frame_detections(FIXTURE)
    tubes = build_tubes(fdets, iou_threshold=0.2, max_misses=2)
    tube = select_longest_tube(tubes)
    assert tube is not None
    interpolate_gaps(tube)

    mi = cfg["model_input"]
    t_max = cfg["classifier"]["max_frames"]
    patches = torch.zeros(t_max, 3, mi["patch_size"], mi["patch_size"])
    mask = torch.zeros(t_max, dtype=torch.bool)
    frame_paths = sorted((FIXTURE / "images").glob("*.jpg"))
    mean = torch.tensor(mi["normalization"]["mean"]).view(3, 1, 1)
    std = torch.tensor(mi["normalization"]["std"]).view(3, 1, 1)
    for slot, entry in enumerate(tube.entries[:t_max]):
        det = entry.detection
        assert det is not None
        img = np.array(Image.open(frame_paths[entry.frame_idx]).convert("RGB"))
        h_img, w_img, _ = img.shape
        cx, cy, w, h = expand_bbox(
            det.cx, det.cy, det.w, det.h, mi["context_factor"]
        )
        box = norm_bbox_to_pixel_square(cx, cy, w, h, w_img, h_img)
        p = crop_and_resize(img, box, mi["patch_size"])
        pt = to_tensor(Image.fromarray(p))
        patches[slot] = (pt - mean) / std
        mask[slot] = True

    with torch.no_grad():
        logit = classifier(patches.unsqueeze(0), mask.unsqueeze(0))
    return float(logit.item())


def test_parity_logit_matches_transformer(
    transformer_classifier: TemporalSmokeClassifier,
) -> None:
    offline = _offline_logit_with_cfg(transformer_classifier, CFG_TRANSFORMER)

    frames = [
        Frame(frame_id=p.stem, image_path=p, timestamp=None)
        for p in sorted((FIXTURE / "images").glob("*.jpg"))
    ]
    yolo = _fake_yolo_from_gt(FIXTURE)
    model = BboxTubeTemporalModel(
        yolo_model=yolo, classifier=transformer_classifier, config=CFG_TRANSFORMER
    )
    out = model.predict(frames=frames)

    assert out.details["num_tubes_kept"] >= 1
    online = max(out.details["tube_logits"])

    assert online == pytest.approx(offline, abs=1e-5)
```

If `BboxTubeTemporalModel.predict()` reads additional transformer-specific config keys (e.g., `max_frames`) from a different location than `cfg["classifier"]`, add them to `CFG_TRANSFORMER` alongside the existing keys — the assertion tells you exactly what's missing.

- [ ] **Step 5.4: Run the parity tests**

Run: `uv run pytest tests/test_model_parity.py -v`
Expected: both `test_parity_logit_matches` and `test_parity_logit_matches_transformer` pass.

- [ ] **Step 5.5: Commit**

```bash
git add tests/test_reproducibility.py tests/test_model_parity.py
git commit -m "test(bbox-tube-temporal): reproducibility + parity coverage for ViT transformer"
```

---

## Task 6: End-to-end `dvc repro` smoke + formatting

**Files:**
- No code changes; run full pipeline + lint.

### Steps

- [ ] **Step 6.1: Lint and format**

Run: `make lint`
Expected: no errors. If `ruff` flags anything, fix inline, then `make format`.

- [ ] **Step 6.2: Full pytest**

Run: `make test`
Expected: all pass.

- [ ] **Step 6.3: Train one variant end-to-end via DVC**

Pick the cheapest variant first (`train_vit_dinov2_frozen` — frozen backbone, only head trains):

Run: `uv run dvc repro train_vit_dinov2_frozen`
Expected: produces `data/06_models/vit_dinov2_frozen/best_checkpoint.pt` and `plots/training_curves.png`. If DINOv2 weights download fails, re-run (HF can be flaky) or fix the model name as per Step 4.2.

- [ ] **Step 6.4: Evaluate it on val**

Run: `uv run dvc repro evaluate_vit_dinov2_frozen@val`
Expected: writes `data/08_reporting/val/vit_dinov2_frozen/{metrics.json,predictions.json,pr_curve.png,...}`.

- [ ] **Step 6.5: Spot-check val F1**

Run: `uv run dvc metrics show`
Expected: `vit_dinov2_frozen` val F1 appears. Compare against `gru` baseline — spec's primary success criterion is that at least one ViT variant beats `gru`. A single-run frozen variant that already beats `gru` is a strong early signal; if it's below `gru`, continue with finetune variants before concluding.

- [ ] **Step 6.6: Train + evaluate the two finetune variants**

Run: `uv run dvc repro train_vit_dinov2_finetune evaluate_vit_dinov2_finetune@val`
Run: `uv run dvc repro train_vit_in21k_finetune evaluate_vit_in21k_finetune@val`
Expected: both produce checkpoints + metrics. If `1e-4` head LR causes loss-spike / NaN, drop `train_vit_*_finetune.backbone_lr` to `3e-6` and `learning_rate` to `3e-5` as the spec's tuning knob, then re-run.

- [ ] **Step 6.7: Regenerate comparison report**

Run: `uv run dvc repro compare_variants`
Expected: `data/08_reporting/comparison.md` now includes rows for the three new variants.

- [ ] **Step 6.8: Commit the DVC lock + any metrics**

```bash
git add dvc.lock
git commit -m "chore(bbox-tube-temporal): reproduce ViT transformer variants"
```
