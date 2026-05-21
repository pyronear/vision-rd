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
