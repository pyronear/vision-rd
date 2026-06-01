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
