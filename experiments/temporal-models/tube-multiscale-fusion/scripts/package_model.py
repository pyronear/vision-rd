"""Package a trained tube-multiscale-fusion checkpoint into a portable .zip.

Bundles the Lightning checkpoint with the exact subset of ``params.yaml``
needed to instantiate the model (``global_branch``, the variant's
``local_branch`` section, ``fusion``, and the variant's training block).

Example:
    uv run python scripts/package_model.py \\
        --checkpoint data/06_models/dinov2_multiscale/best_checkpoint.pt \\
        --params-path params.yaml \\
        --params-key train_dinov2_multiscale \\
        --output data/06_models/dinov2_multiscale/model_package.zip
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

PACKAGE_VERSION = "1.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, required=True)
    parser.add_argument("--params-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    full_params = yaml.safe_load(args.params_path.read_text())
    if args.params_key not in full_params:
        raise KeyError(f"params key not found in {args.params_path}: {args.params_key}")
    train_cfg = full_params[args.params_key]
    local_section = train_cfg.get("local_branch_section", "local_branch")
    global_section = train_cfg.get("global_branch_section", "global_branch")
    fusion_section = train_cfg.get("fusion_section", "fusion")

    bundled_params = {
        global_section: full_params[global_section],
        local_section: full_params[local_section],
        fusion_section: full_params[fusion_section],
        args.params_key: train_cfg,
    }

    manifest = {
        "package_version": PACKAGE_VERSION,
        "experiment": "tube-multiscale-fusion",
        "params_key": args.params_key,
        "sections": {
            "global_branch": global_section,
            "local_branch": local_section,
            "fusion": fusion_section,
        },
        "checkpoint": {
            "filename": "checkpoint.pt",
            "sha256": _sha256(args.checkpoint),
            "size_bytes": args.checkpoint.stat().st_size,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        shutil.copy2(args.checkpoint, td_path / "checkpoint.pt")
        (td_path / "params.yaml").write_text(
            yaml.safe_dump(bundled_params, sort_keys=False)
        )
        (td_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (td_path / "README.md").write_text(_load_readme(args.params_key))

        with zipfile.ZipFile(
            args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            for name in ("manifest.json", "params.yaml", "checkpoint.pt", "README.md"):
                zf.write(td_path / name, arcname=name)

    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


def _load_readme(params_key: str) -> str:
    return f"""# tube-multiscale-fusion model package

This archive contains a trained `tube-multiscale-fusion` smoke classifier.

Contents:
- `checkpoint.pt`: Lightning checkpoint
  (`LitTubeMultiscaleClassifier.load_from_checkpoint`)
- `params.yaml`: the exact `global_branch`, `local_branch`, `fusion`, and
  training-variant sections used to train this checkpoint
- `manifest.json`: package metadata (version, SHA-256, params key)

## Loading

```python
import zipfile, tempfile, yaml
from pathlib import Path
from tube_multiscale_fusion.lit_module import LitTubeMultiscaleClassifier

with zipfile.ZipFile("model_package.zip") as zf, tempfile.TemporaryDirectory() as td:
    zf.extractall(td)
    ckpt = Path(td) / "checkpoint.pt"
    lit = LitTubeMultiscaleClassifier.load_from_checkpoint(str(ckpt), pretrained=False)
    lit.eval()
```

## Inference contract

- Input: `(B, T=16, 3, 224, 224)` ImageNet-normalized RGB tensor plus a
  `(B, 16)` boolean mask (True = real frame, False = padded).
- Output: `(B,)` logits; apply `torch.sigmoid` to get smoke probability.

Trained variant: `{params_key}`. See `params.yaml` inside this archive for the
full hyperparameter set.
"""


if __name__ == "__main__":
    main()
