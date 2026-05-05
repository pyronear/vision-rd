"""Stem parsing and sequence grouping for the review app.

Stems in pyro-dataset are
``<source>_<camera>_<sequence_id>_<timestamp>`` where ``<timestamp>``
is ISO-8601 with hyphen-replaced colons (e.g. ``2024-02-17T17-36-57``).
``<source>`` may contain hyphens (``awf-axis``, ``pyronear-force-06``)
but never underscores. Splitting on the last ``_`` reliably yields
``(sequence_id, timestamp)``.
"""


def parse_stem(stem: str) -> tuple[str, str]:
    """Return ``(sequence_id, timestamp)`` for a pyro-dataset stem."""
    sequence_id, timestamp = stem.rsplit("_", 1)
    return sequence_id, timestamp
