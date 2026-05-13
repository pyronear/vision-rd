"""Parse the pyro-dataset revision from the imported ``.dvc`` files.

Each ``data/01_raw/datasets/<split>/{images,labels}.dvc`` records the
upstream ``rev`` of pyro-dataset it was imported from. We surface that
in the audit app header so reviewers know which dataset version they
are looking at.
"""

from pathlib import Path

import yaml

_DVC_FILES = ("images.dvc", "labels.dvc")


def _read_rev(dvc_path: Path) -> str | None:
    payload = yaml.safe_load(dvc_path.read_text())
    for dep in payload.get("deps", []) or []:
        repo = dep.get("repo") or {}
        rev = repo.get("rev")
        if rev:
            return str(rev)
    return None


def read_dataset_version(datasets_root: Path) -> str | None:
    """Return the pyro-dataset rev pinned across all split ``.dvc`` files.

    Walks ``datasets_root/<split>/{images,labels}.dvc`` and reads the
    ``deps[0].repo.rev`` field of each. Returns:

    - ``None`` when ``datasets_root`` is missing or no ``.dvc`` file
      records a rev (e.g. fresh checkout, local-only data).
    - A single rev string (e.g. ``"v4.0.0"``) when all files agree.
    - ``"mixed: <r1>, <r2>, ..."`` when files disagree — a signal that
      the imports drifted and should be reconciled.
    """
    if not datasets_root.is_dir():
        return None
    revs: set[str] = set()
    for split_dir in sorted(p for p in datasets_root.iterdir() if p.is_dir()):
        for name in _DVC_FILES:
            dvc_path = split_dir / name
            if not dvc_path.is_file():
                continue
            rev = _read_rev(dvc_path)
            if rev is not None:
                revs.add(rev)
    if not revs:
        return None
    if len(revs) == 1:
        return next(iter(revs))
    return "mixed: " + ", ".join(sorted(revs))
