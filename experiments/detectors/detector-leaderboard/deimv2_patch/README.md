# DEIMv2 repo patch (for reproducibility)

`deimv2_repo/` is an external clone (gitignored). To reproduce the tiled-training
experiment, after cloning DEIMv2 apply this patch:

1. Copy `smoke_aug.py` into `deimv2_repo/engine/data/transforms/`.
2. In `deimv2_repo/engine/data/transforms/__init__.py`, add:
   `from .smoke_aug import FPInject, RandomGaussianBlur`
3. Ensure `base/deimv2.yml` keeps `copyblend_prob: 0.5` (stock) — the tiled config
   overrides `copyblend_type: 'copy'` + `with_expand`.

`smoke_aug.py` registers `RandomGaussianBlur` (p-gated blur) and `FPInject`
(hard-negative fp-crop paste, adds no box), and sets the `file_system` torch
multiprocessing sharing strategy (avoids DataLoader SIGBUS on the large tiled set).
