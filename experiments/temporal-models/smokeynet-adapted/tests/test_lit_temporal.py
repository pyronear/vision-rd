"""Tests for the basic temporal classifier Lightning module."""

import torch

from smokeynet_adapted.lit_temporal import LitTemporalClassifier


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
    trainable = {id(p) for p in lit.model.parameters() if p.requires_grad}
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
