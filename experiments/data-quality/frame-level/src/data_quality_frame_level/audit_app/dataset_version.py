"""Parse the pyro-dataset revision from ``SOURCE.json``.

``data/01_raw/datasets/SOURCE.json`` is written by
``scripts/refresh_datasets.py`` and records which pyro-dataset version
the raw splits were last imported from. The audit-app header surfaces
that version so reviewers know which dataset they are looking at.
"""

import json
from pathlib import Path


def read_dataset_version(datasets_root: Path) -> str | None:
    """Return the pyro-dataset version recorded in ``SOURCE.json``.

    Returns ``None`` when ``datasets_root`` is missing, ``SOURCE.json``
    is missing, the JSON is malformed, or the ``pyro_dataset_version``
    field is absent. Otherwise returns the version string verbatim.
    """
    if not datasets_root.is_dir():
        return None
    source_path = datasets_root / "SOURCE.json"
    if not source_path.is_file():
        return None
    try:
        payload = json.loads(source_path.read_text())
    except json.JSONDecodeError:
        return None
    version = payload.get("pyro_dataset_version")
    if not isinstance(version, str):
        return None
    return version
