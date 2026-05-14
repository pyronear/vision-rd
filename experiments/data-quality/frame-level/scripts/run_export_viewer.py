"""Launch the read-only export viewer.

Discovers exported (model, split) pairs under ``data/10_export/`` and serves
a browser viewer for the YOLO labels + images at
``data/01_raw/datasets/<split>/images/``.

Usage::

    uv run --group audit-app python scripts/run_export_viewer.py
"""

import argparse
from pathlib import Path

import uvicorn

from data_quality_frame_level.export_viewer.main import create_app, discover_contexts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    contexts = discover_contexts(args.repo_root)
    if not contexts:
        print(f"No exports found under {args.repo_root / 'data/10_export'}")
    else:
        for c in contexts:
            print(f"  {c.model} / {c.split}  ({c.labels_dir})")
    app = create_app(contexts=contexts)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
