"""Import pyro-annotator sequences (human-labeled zip export) into the store.

The label comes from the folder path (``smoke/<subtype>``, ``fp/<subtype>``,
``unlabeled``). Frames already live in the zip, so images are copied (not
downloaded). Camera/org/timestamps are enriched per sequence via the admin
platform API, so these sequences sit in the same org -> camera navigation as the
alert-API source. Enrichment requires admin creds.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from . import platform_api
from .store import FrameRef, SequenceMeta, slug, write_meta

log = logging.getLogger(__name__)


def parse_label(klass: str, subtype: str | None) -> tuple[str, str | None]:
    """Map a (class, subtype) folder pair to the tri-state label + detail."""
    if klass == "smoke":
        return "smoke", subtype
    if klass == "fp":
        return "fp", subtype
    if klass == "unlabeled":
        return "unknown", None
    raise ValueError(f"unknown class folder: {klass!r}")


def iter_zip_sequences(
    src: Path,
) -> Iterator[tuple[str, str | None, int, Path]]:
    """Yield (class, subtype, seq_id, seq_dir) for each seq_<id>/ with images/.

    Layout: ``<class>/<subtype>/seq_<id>`` (smoke, fp) or ``<class>/seq_<id>``
    (unlabeled). macOS ``__MACOSX`` entries are skipped.
    """
    for images_dir in sorted(src.rglob("images")):
        seq_dir = images_dir.parent
        rel = seq_dir.relative_to(src).parts
        if "__MACOSX" in rel or not seq_dir.name.startswith("seq_"):
            continue
        klass = rel[0]
        subtype = rel[1] if len(rel) == 3 else None
        seq_id = int(seq_dir.name[len("seq_") :])
        yield klass, subtype, seq_id, seq_dir
