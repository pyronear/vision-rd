"""Persist and restore FiftyOne sample tags across sessions.

Reviewers tag samples in the FiftyOne app (e.g. ``label:add-smoke``,
``reviewer:arthur``) to record per-image decisions. Those tags live in
the local FiftyOne mongo store, which is machine-local and not
shareable. This module provides the pure serialization boundary:

- :func:`payload_from_stem_tags` → turn a ``{stem: [tags]}`` map into a
  JSON-ready dict that can be written to disk and tracked by DVC.
- :func:`stem_tags_from_payload` → inverse for re-applying on startup.
- :func:`merge_tags` → union two tag lists (used when re-importing on
  top of an existing mongo dataset so we don't clobber fresh tags).

The FiftyOne-touching read/write helpers live in
:mod:`scripts/export_review` and :mod:`scripts/launch_fiftyone` — this
module stays pure so it's trivially unit-testable without a live mongo.
"""

from __future__ import annotations

PAYLOAD_VERSION = 1

# Controlled vocabulary for sample tags in the FiftyOne review workflow.
# Populated on ``Dataset.tags`` by :func:`fiftyone_build.build_dataset`
# so the tag popover autocompletes these as the reviewer types.
REVIEW_VOCAB: tuple[str, ...] = (
    "label:add-smoke",   # YOLO found real smoke; GT has no bbox. Add to annotations.
    "label:remove-gt",   # GT bbox is not smoke (cloud/dust/glare). Remove.
    "label:fix-bbox",    # Smoke is present but bbox is mispositioned. Reposition.
    "label:ok",          # Flag is a genuine model error, not a label issue.
    "status:unclear",    # Ambiguous — revisit or ask a second reviewer.
)


def payload_from_stem_tags(dataset_name: str, stem_tags: dict[str, list[str]]) -> dict:
    """Build a JSON-serializable payload from a ``{stem: [tags]}`` map.

    Stems whose tag list is empty are dropped — only carry decisions
    that were actually made. Tag lists are sorted for stable diffs.
    """
    filtered = {
        stem: sorted(set(tags)) for stem, tags in sorted(stem_tags.items()) if tags
    }
    return {
        "version": PAYLOAD_VERSION,
        "dataset_name": dataset_name,
        "tags_by_stem": filtered,
    }


def stem_tags_from_payload(payload: dict) -> dict[str, list[str]]:
    """Extract the ``{stem: [tags]}`` map from a persisted payload."""
    return dict(payload.get("tags_by_stem", {}))


def merge_tags(existing: list[str], incoming: list[str]) -> list[str]:
    """Union two tag lists without duplicates, sorted.

    Called per-sample when re-importing a tags file into a dataset that
    already carries tags (e.g. a reviewer started tagging before running
    ``make review-pull``). New mongo-side tags are preserved.
    """
    return sorted(set(existing) | set(incoming))
