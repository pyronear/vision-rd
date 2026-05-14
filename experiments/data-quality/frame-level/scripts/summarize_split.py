"""Print a one-block summary of an export for a (model, split).

Reads ``data/10_export/<model>/<split>/{manifest.json,pending.json}`` and
prints reviewer-facing counts (reviewed, changed, added/removed/modified,
pending unclear, contributors). Used by ``scripts/audit_publish.sh`` to
show what a reviewer is about to publish.

Exits 0 in all non-error cases. Prints a warning line if ``manifest.json``
is older than ``review.json`` (i.e. the app's Export button wasn't clicked
after the latest edits).
"""

import argparse
import json
import sys
from pathlib import Path


def _fmt(label: str, value: str) -> str:
    return f"    {label:<22}{value}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    args = parser.parse_args()

    review = (
        args.repo_root / "data" / "09_review" / args.model / args.split / "review.json"
    )
    export_dir = args.repo_root / "data" / "10_export" / args.model / args.split
    manifest_path = export_dir / "manifest.json"
    pending_path = export_dir / "pending.json"

    if not manifest_path.is_file():
        print("    (no manifest — click Export in the app first)")
        return 0

    manifest = json.loads(manifest_path.read_text())
    pending = (
        json.loads(pending_path.read_text()).get("pending", [])
        if pending_path.is_file()
        else []
    )
    t = manifest["totals"]
    contributors = manifest.get("contributors") or []

    print(_fmt("reviewed:", str(t["reviewed"])))
    changed_detail = (
        f"{t['changed']}   "
        f"(+{t['added']} added, -{t['removed']} removed, ~{t['modified']} modified)"
    )
    print(_fmt("changed:", changed_detail))
    print(_fmt("pending (unclear):", str(len(pending))))
    print(_fmt("contributors:", ", ".join(contributors) if contributors else "(none)"))

    if review.is_file() and review.stat().st_mtime > manifest_path.stat().st_mtime:
        print(
            "    !! review.json is newer than manifest.json — "
            "click Export in the app to refresh"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
