"""Regression tests over the real working-set sequences.

These pin the candidate's tube counts on the sequences reviewed in the lab, so a
change to ``candidate.py`` that alters a confirmed-good result is caught. They
need the DVC-tracked detections cache (``make cache``); when it is absent (e.g.
CI without the data) the whole module is skipped.

When you intentionally change the algorithm, re-verify the affected sequences in
the app and update the snapshots below.
"""

from pathlib import Path

import pytest

from tube_builder_lab.cache import detections_present, load_cached
from tube_builder_lab.candidate import build_tubes_candidate
from tube_builder_lab.pipeline import (
    current_builder,
    detections_to_display_tubes,
    load_pipeline_config,
)

DETECTIONS = Path("data/05_model_input/detections")
PIPELINE_CONFIG = Path("data/05_model_input/pipeline_config.yaml")

pytestmark = pytest.mark.skipif(
    not PIPELINE_CONFIG.exists(),
    reason="needs the DVC-tracked detections cache (run `make cache`)",
)

# Known-fragmented targets -> expected candidate tube count (all reduced).
TARGET_EXPECTED = {
    "platform_43096": 1,
    "platform_42466": 1,
    "platform_41304": 1,
    "platform_41319": 1,
    "platform_41310": 3,
    "platform_41289": 3,
    "platform_41209": 1,
    "platform_40800": 1,
    "platform_41616": 1,
    "platform_41786": 1,
    "platform_42562": 1,
    "platform_42538": 1,
    "platform_41887": 1,
    "platform_41541": 1,
    "platform_43206": 2,
}

# Control sequences reviewed + confirmed good in the lab -> expected count.
VERIFIED_CONTROL_EXPECTED = {
    "platform_40686": 1,
    "platform_41168": 4,
    "platform_41314": 2,
    "platform_41356": 4,
    "platform_41380": 1,
    "platform_41443": 2,
    "platform_41502": 1,
    "platform_42931": 1,
    "platform_42936": 2,
    "platform_42945": 1,
    "platform_43020": 4,
    "platform_43230": 4,
    "platform_43266": 2,
    "platform_43267": 1,
    "platform_43317": 3,
    "platform_43359": 1,
    "platform_43441": 1,
}

REVIEWED_EXPECTED = {**TARGET_EXPECTED, **VERIFIED_CONTROL_EXPECTED}


def _candidate_count(key: str, cfg) -> int:
    fds = load_cached(DETECTIONS, key)
    return len(
        detections_to_display_tubes(fds, build_tubes_candidate, cfg, truncate=False)
    )


def _current_count(key: str, cfg) -> int:
    fds = load_cached(DETECTIONS, key)
    return len(
        detections_to_display_tubes(fds, current_builder(cfg), cfg, truncate=False)
    )


@pytest.mark.parametrize("key, expected", sorted(REVIEWED_EXPECTED.items()))
def test_reviewed_sequence_candidate_count(key: str, expected: int):
    """The candidate's tube count on a reviewed sequence matches the snapshot."""
    if not detections_present(DETECTIONS, key):
        pytest.skip(f"no cached detections for {key}")
    assert _candidate_count(key, load_pipeline_config(PIPELINE_CONFIG)) == expected


@pytest.mark.parametrize("key", sorted(TARGET_EXPECTED))
def test_target_is_reduced(key: str):
    """Every known-fragmented target ends with fewer tubes than the current builder."""
    if not detections_present(DETECTIONS, key):
        pytest.skip(f"no cached detections for {key}")
    cfg = load_pipeline_config(PIPELINE_CONFIG)
    assert _candidate_count(key, cfg) < _current_count(key, cfg)
