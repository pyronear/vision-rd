"""Stem parsing and temporal sequence grouping for the review app.

Stems in pyro-dataset are
``<source>_<camera>_<sequence_id>_<timestamp>`` where ``<timestamp>``
is ISO-8601 with hyphen-replaced colons (e.g. ``2024-02-17T17-36-57``).
``<source>`` may contain hyphens (``awf-axis``, ``pyronear-force-06``)
but never underscores. Splitting on the last ``_`` reliably yields
``(prefix, timestamp)``.

A single ``prefix`` (camera + dataset's recorded ``sequence_id``) may
recur across many separate detection events spanning days or weeks. We
treat each contiguous block of frames within ``max_gap_seconds`` of
each other as one *temporal sequence*; later blocks under the same
prefix become ``prefix#1``, ``prefix#2``, … in chronological order.
"""

from collections.abc import Iterable
from datetime import datetime

_TS_FORMAT = "%Y-%m-%dT%H-%M-%S"


def parse_stem(stem: str) -> tuple[str, str]:
    """Return ``(prefix, timestamp)`` for a pyro-dataset stem."""
    prefix, timestamp = stem.rsplit("_", 1)
    return prefix, timestamp


def parse_timestamp(ts: str) -> datetime:
    """Parse a pyro-dataset timestamp string like ``2024-02-17T17-36-57``."""
    return datetime.strptime(ts, _TS_FORMAT)


def assign_temporal_sequences(
    stems: Iterable[str], *, max_gap_seconds: int = 180
) -> dict[str, str]:
    """Map each stem to a temporal-sequence identifier.

    Two stems sharing a prefix belong to the same temporal sequence iff
    they appear in the same contiguous block — i.e., no gap larger than
    ``max_gap_seconds`` between consecutive frames in the block.
    Successive blocks under the same prefix get ``#0``, ``#1``, ... in
    chronological order.
    """
    by_prefix: dict[str, list[tuple[datetime, str]]] = {}
    for stem in stems:
        prefix, ts_raw = parse_stem(stem)
        by_prefix.setdefault(prefix, []).append((parse_timestamp(ts_raw), stem))
    out: dict[str, str] = {}
    for prefix, items in by_prefix.items():
        items.sort()
        cluster_idx = 0
        prev_ts: datetime | None = None
        for ts, stem in items:
            if prev_ts is not None and (ts - prev_ts).total_seconds() > max_gap_seconds:
                cluster_idx += 1
            out[stem] = f"{prefix}#{cluster_idx}"
            prev_ts = ts
    return out
