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

import difflib
from collections.abc import Iterable

PAYLOAD_VERSION = 1

REVIEWER_TAG_PREFIX = "reviewer:"
LABEL_TAG_PREFIX = "label:"

# Controlled vocabulary for sample tags in the FiftyOne review workflow.
# Seeded on a neutral sample by :func:`fiftyone_build._seed_tag_vocab`
# so the tag popover autocompletes these as the reviewer types.
REVIEW_VOCAB: tuple[str, ...] = (
    "label:add-smoke",   # YOLO found real smoke; GT has no bbox. Add to annotations.
    "label:remove-gt",   # GT bbox is not smoke (cloud/dust/glare). Remove.
    "label:fix-bbox",    # Smoke is present but bbox is mispositioned. Reposition.
    "label:ok",          # Flag is a genuine model error, not a label issue.
    "status:unclear",    # Ambiguous — revisit or ask a second reviewer.
)

# Marker tag applied alongside REVIEW_VOCAB on the vocab-seed sample.
# Presence of this tag means "this sample's tags are the autocomplete
# seed, not a review decision" — export/import should skip it.
VOCAB_SEED_TAG = "system:vocab-seed"


def is_vocab_seed(tags: list[str]) -> bool:
    """True iff ``tags`` carry the vocab-seed marker."""
    return VOCAB_SEED_TAG in tags


def is_valid_tag(tag: str) -> bool:
    """True iff ``tag`` is an accepted review tag.

    Accepts exact :data:`REVIEW_VOCAB` entries, the
    :data:`VOCAB_SEED_TAG` marker, and any ``reviewer:<handle>``
    attribution (free-form handle after the prefix). Everything else —
    including typos, wrong case, and trailing whitespace — is rejected.
    """
    return (
        tag in REVIEW_VOCAB
        or tag == VOCAB_SEED_TAG
        or tag.startswith(REVIEWER_TAG_PREFIX)
    )


def invalid_tags(tags: Iterable[str]) -> list[str]:
    """Return the subset of ``tags`` rejected by :func:`is_valid_tag`."""
    return [t for t in tags if not is_valid_tag(t)]


def suggest_tag(tag: str) -> str | None:
    """Return the closest :data:`REVIEW_VOCAB` entry to ``tag``, or None.

    Uses stdlib :func:`difflib.get_close_matches` with a relatively
    lenient cutoff so reviewers get a hint for typos like
    ``lable:add-smoke`` → ``label:add-smoke``. Returns ``None`` when
    no candidate is within the cutoff (input is too far from any
    vocab entry to guess confidently).
    """
    matches = difflib.get_close_matches(tag, REVIEW_VOCAB, n=1, cutoff=0.7)
    return matches[0] if matches else None


def scan_invalid(stem_tags: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return ``{stem: [bad_tags]}`` for every stem with invalid tags.

    Pure wrapper around :func:`invalid_tags` over a stem→tags mapping;
    stems whose tags are fully valid are omitted from the result.
    Used by both :mod:`scripts.export_review` and
    :mod:`scripts.validate_review` to share the validation body.
    """
    return {
        stem: bad for stem, tags in stem_tags.items() if (bad := invalid_tags(tags))
    }


def is_reviewed(tags: Iterable[str]) -> bool:
    """True iff ``tags`` include at least one ``label:`` decision tag.

    Samples tagged only with ``status:unclear`` or ``reviewer:<handle>``
    are considered in-progress, not reviewed. This is deliberate: a
    reviewer lists someone's name on a sample they've *started* looking
    at, but 'reviewed' means they've made a label decision.
    """
    return any(t.startswith(LABEL_TAG_PREFIX) for t in tags)


def count_reviewed(stem_tags: dict[str, list[str]]) -> tuple[int, int]:
    """Return ``(reviewed, total)`` counts over a stem→tags map."""
    reviewed = sum(1 for tags in stem_tags.values() if is_reviewed(tags))
    return reviewed, len(stem_tags)


def format_progress_line(
    name: str, reviewed: int, total: int, bar_width: int = 20
) -> str:
    """Format a single progress line with counts, percentage, and ASCII bar."""
    ratio = reviewed / total if total else 0.0
    filled = int(round(ratio * bar_width))
    bar = "#" * filled + "-" * (bar_width - filled)
    return (
        f"{name} {total:>4} samples: {reviewed:>4} reviewed "
        f"({ratio * 100:>4.1f}%) [{bar}]"
    )


def format_invalid_report(
    dataset_name: str, stem_to_invalid: dict[str, list[str]]
) -> str:
    """Format a human-readable report of invalid tags with suggestions."""
    total = sum(len(v) for v in stem_to_invalid.values())
    header = (
        f"[{dataset_name}] {total} invalid tag(s) "
        f"across {len(stem_to_invalid)} sample(s):"
    )
    lines = [header]
    for stem in sorted(stem_to_invalid):
        for tag in sorted(stem_to_invalid[stem]):
            suggestion = suggest_tag(tag)
            hint = f"  (did you mean '{suggestion}'?)" if suggestion else ""
            lines.append(f"  {stem}: '{tag}'{hint}")
    return "\n".join(lines)


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
