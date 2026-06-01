# Merge-aware tubes + retrain `vit_dinov2_finetune` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Propagate the lab's `merge_colocated_tubes` post-build merge pass into `lib/bbox-tube-temporal` as an optional, config-gated step (backward compatible — missing config keys = skip merge), then retrain `vit_dinov2_finetune` on the new tubes and compare against the current packaged model.

**Architecture:** The merge becomes a first-class lib function with explicit threshold kwargs (no module constants). The inference pipeline `predict` calls a new `build_tubes_for_inference` orchestrator that does build → (optional merge with filter sandwich) → filter+interpolate; missing merge keys preserve the legacy single-filter path. The experiment's `scripts/build_tubes.py` (training tube prep) inserts the merge between `build_tubes` and `select_longest_tube`; `params.yaml` gains three merge keys; `package_model.py` copies them into the bundled `config.yaml`. The DVC graph reruns the affected stages.

**Tech Stack:** Python 3.11+, pytest, ruff, PyYAML, PyTorch + Lightning, DVC, uv. No new dependencies.

Spec: [`docs/specs/2026-06-01-merge-aware-tubes-and-retrain.md`](../specs/2026-06-01-merge-aware-tubes-and-retrain.md)

**Faithful values from PR #72 (lab's tuned defaults):**
- `merge_iomin = 0.3`, `merge_prox_factor = 1.0`, `merge_max_gap = 10`

> **Operator notes:** stage files explicitly (never `git add -A`); no Claude/Anthropic co-author trailers in commits; do not run `dvc pull` (operator syncs manually). Run lib commands from `lib/bbox-tube-temporal/` and experiment commands from `experiments/temporal-models/bbox-tube-temporal/`.

---

## Task 1: Add `merge_colocated_tubes` (and helpers) to the lib

**Files:**
- Modify: `lib/bbox-tube-temporal/src/bbox_tube_temporal/tubes.py`
- Test: `lib/bbox-tube-temporal/tests/test_tubes.py`

- [ ] **Step 1: Append the failing merge tests** to `lib/bbox-tube-temporal/tests/test_tubes.py` (after the existing `TestTubeFromRecord` class)

```python
# ── merge_colocated_tubes ────────────────────────────────────────────────


def _merge_tube(tid: int, frame_boxes):
    """frame_boxes: list[(frame_idx, (cx, cy, w, h))]."""
    entries = [
        TubeEntry(frame_idx=f, detection=_det(cx, cy, w, h))
        for f, (cx, cy, w, h) in frame_boxes
    ]
    return Tube(
        tube_id=tid,
        entries=entries,
        start_frame=frame_boxes[0][0],
        end_frame=frame_boxes[-1][0],
    )


class TestMergeColocatedTubes:
    KWARGS = dict(merge_iomin=0.3, merge_prox_factor=1.0, merge_max_gap=10)

    def test_empty_input(self):
        from bbox_tube_temporal.tubes import merge_colocated_tubes
        assert merge_colocated_tubes([], **self.KWARGS) == []

    def test_single_tube_passthrough(self):
        from bbox_tube_temporal.tubes import merge_colocated_tubes
        t = _merge_tube(0, [(0, (0.5, 0.5, 0.05, 0.05)), (1, (0.5, 0.5, 0.05, 0.05))])
        out = merge_colocated_tubes([t], **self.KWARGS)
        assert len(out) == 1
        assert (out[0].start_frame, out[0].end_frame) == (0, 1)

    def test_merge_contained_overlapping(self):
        """Small box inside big box on shared frames -> one tube."""
        from bbox_tube_temporal.tubes import merge_colocated_tubes
        big = _merge_tube(0, [(0, (0.5, 0.5, 0.2, 0.2)), (1, (0.5, 0.5, 0.2, 0.2))])
        small = _merge_tube(1, [(1, (0.5, 0.5, 0.02, 0.02))])
        out = merge_colocated_tubes([big, small], **self.KWARGS)
        assert len(out) == 1
        assert (out[0].start_frame, out[0].end_frame) == (0, 1)

    def test_keep_distinct_far_apart(self):
        from bbox_tube_temporal.tubes import merge_colocated_tubes
        left = _merge_tube(0, [(0, (0.2, 0.5, 0.05, 0.05)), (1, (0.2, 0.5, 0.05, 0.05))])
        right = _merge_tube(1, [(0, (0.8, 0.5, 0.05, 0.05)), (1, (0.8, 0.5, 0.05, 0.05))])
        out = merge_colocated_tubes([left, right], **self.KWARGS)
        assert len(out) == 2

    def test_bridge_gap_within_window(self):
        from bbox_tube_temporal.tubes import merge_colocated_tubes
        a = _merge_tube(0, [(0, (0.5, 0.5, 0.05, 0.05)), (2, (0.5, 0.5, 0.05, 0.05))])
        b = _merge_tube(1, [(8, (0.5, 0.5, 0.05, 0.05)), (10, (0.5, 0.5, 0.05, 0.05))])
        out = merge_colocated_tubes([a, b], **self.KWARGS)
        assert len(out) == 1
        assert (out[0].start_frame, out[0].end_frame) == (0, 10)
        assert any(e.detection is None for e in out[0].entries)

    def test_do_not_bridge_beyond_window(self):
        from bbox_tube_temporal.tubes import merge_colocated_tubes
        a = _merge_tube(0, [(0, (0.5, 0.5, 0.05, 0.05)), (2, (0.5, 0.5, 0.05, 0.05))])
        b = _merge_tube(1, [(16, (0.5, 0.5, 0.05, 0.05)), (18, (0.5, 0.5, 0.05, 0.05))])
        out = merge_colocated_tubes([a, b], **self.KWARGS)
        assert len(out) == 2

    def test_transitive_merge(self):
        from bbox_tube_temporal.tubes import merge_colocated_tubes
        a = _merge_tube(0, [(0, (0.5, 0.5, 0.05, 0.05))])
        b = _merge_tube(1, [(3, (0.5, 0.5, 0.05, 0.05))])
        c = _merge_tube(2, [(6, (0.5, 0.5, 0.05, 0.05))])
        out = merge_colocated_tubes([a, b, c], **self.KWARGS)
        assert len(out) == 1

    def test_proximity_is_scale_relative(self):
        """Tiny boxes ~1.75 box-sizes apart MUST NOT merge (no teleport)."""
        from bbox_tube_temporal.tubes import merge_colocated_tubes
        a = _merge_tube(0, [(0, (0.50, 0.5, 0.02, 0.02)), (1, (0.50, 0.5, 0.02, 0.02))])
        near = _merge_tube(1, [(2, (0.515, 0.5, 0.02, 0.02)), (3, (0.515, 0.5, 0.02, 0.02))])
        assert len(merge_colocated_tubes([a, near], **self.KWARGS)) == 1
        far = _merge_tube(1, [(2, (0.535, 0.5, 0.02, 0.02)), (3, (0.535, 0.5, 0.02, 0.02))])
        assert len(merge_colocated_tubes([a, far], **self.KWARGS)) == 2

    def test_combine_tiebreak_is_order_invariant(self):
        """Equal-area ties resolve to the same box regardless of input order."""
        from bbox_tube_temporal.tubes import merge_colocated_tubes
        a = _merge_tube(0, [(0, (0.50, 0.5, 0.02, 0.02)), (1, (0.50, 0.5, 0.02, 0.02))])
        b = _merge_tube(1, [(1, (0.51, 0.5, 0.02, 0.02)), (2, (0.51, 0.5, 0.02, 0.02))])

        def box_at_1(tubes):
            [t] = tubes
            return next(e.detection for e in t.entries if e.frame_idx == 1)

        assert box_at_1(merge_colocated_tubes([a, b], **self.KWARGS)).cx == pytest.approx(0.51)
        assert box_at_1(merge_colocated_tubes([b, a], **self.KWARGS)).cx == pytest.approx(0.51)
```

- [ ] **Step 2: Run the new tests; expect failure**

```bash
cd lib/bbox-tube-temporal
uv run pytest tests/test_tubes.py::TestMergeColocatedTubes -v
```
Expected: `ImportError: cannot import name 'merge_colocated_tubes' from 'bbox_tube_temporal.tubes'`.

- [ ] **Step 3: Append the implementation** to `lib/bbox-tube-temporal/src/bbox_tube_temporal/tubes.py` (after `tube_from_record`)

```python
# ── post-hoc co-located merge ───────────────────────────────────────────────
#
# Fragments-of-the-same-plume merge that runs after build_tubes. Two tubes are
# linked when temporally close AND, at the frames where they are nearest in
# time, the smaller is mostly inside the larger (IoMin) OR their centers are
# within ~merge_prox_factor box-sizes (scale-relative, so tiny boxes don't
# chain across many widths). Merging is connected-components under that
# relation; per-frame collisions on overlap keep the higher-ranked box
# (largest area, then highest confidence, then larger cx — deterministic).

from collections.abc import Callable

Observed = list[tuple[int, Detection]]


def merge_colocated_tubes(
    tubes: list[Tube],
    *,
    merge_iomin: float,
    merge_prox_factor: float,
    merge_max_gap: int,
) -> list[Tube]:
    """Merge tubes that are fragments of the same plume.

    Args:
        tubes: candidate tubes (typically from :func:`build_tubes` after the
            inference filter).
        merge_iomin: containment threshold — link if the smaller box's
            intersection-over-min-area with the larger is at least this.
        merge_prox_factor: proximity factor — link if centers are within this
            many box-sizes (scale-relative).
        merge_max_gap: temporal gap, in frames, allowed between two observed
            spans for them to still be considered the same plume.

    Returns:
        Merged tubes, sorted by ``(start_frame, end_frame)`` with sequentially
        re-assigned ``tube_id``s. Gap entries (``detection=None``) are inserted
        for unobserved frames between observed ones.
    """
    observed = [_observed(t) for t in tubes]

    def related(i: int, j: int) -> bool:
        return _same_plume(
            observed[i], observed[j],
            merge_iomin=merge_iomin,
            merge_prox_factor=merge_prox_factor,
            merge_max_gap=merge_max_gap,
        )

    components = _connected_components(len(tubes), related)
    merged = [_combine(c, observed) for c in components]
    merged = [t for t in merged if t is not None]
    merged.sort(key=lambda t: (t.start_frame, t.end_frame))
    for tube_id, tube in enumerate(merged):
        tube.tube_id = tube_id
    return merged


# ── the "same plume" relation ───────────────────────────────────────────────


def _same_plume(
    a: Observed,
    b: Observed,
    *,
    merge_iomin: float,
    merge_prox_factor: float,
    merge_max_gap: int,
) -> bool:
    if not a or not b:
        return False
    if _time_gap(a, b) > merge_max_gap:
        return False
    det_a, det_b = _closest_in_time(a, b)
    return _same_box(
        det_a, det_b,
        merge_iomin=merge_iomin,
        merge_prox_factor=merge_prox_factor,
    )


def _time_gap(a: Observed, b: Observed) -> int:
    """Frames separating the two observed spans (0 when they overlap)."""
    a_start, a_end = a[0][0], a[-1][0]
    b_start, b_end = b[0][0], b[-1][0]
    if b_start > a_end:
        return b_start - a_end
    if a_start > b_end:
        return a_start - b_end
    return 0


def _closest_in_time(a: Observed, b: Observed) -> tuple[Detection, Detection]:
    best = min(
        ((abs(fa - fb), da, db) for fa, da in a for fb, db in b),
        key=lambda candidate: candidate[0],
    )
    return best[1], best[2]


def _same_box(
    a: Detection,
    b: Detection,
    *,
    merge_iomin: float,
    merge_prox_factor: float,
) -> bool:
    if _iou_min(a, b) >= merge_iomin:
        return True
    box_size = max(a.w, a.h, b.w, b.h)
    return _center_distance(a, b) <= merge_prox_factor * box_size


def _iou_min(a: Detection, b: Detection) -> float:
    """Intersection over the smaller box's area (containment)."""
    ax1, ay1, ax2, ay2 = a.cx - a.w / 2, a.cy - a.h / 2, a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1, bx2, by2 = b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    smaller = min(a.w * a.h, b.w * b.h)
    return inter / smaller if smaller > 0 else 0.0


def _center_distance(a: Detection, b: Detection) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


# ── grouping and rebuilding ─────────────────────────────────────────────────


def _connected_components(
    n: int, related: Callable[[int, int], bool]
) -> list[list[int]]:
    """Partition indices ``range(n)`` into components linked by ``related``."""
    parent = list(range(n))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if related(i, j):
                parent[root(i)] = root(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def _combine(members: list[int], observed: list[Observed]) -> Tube | None:
    """Fuse a component's fragments into one tube (largest box wins on overlap)."""
    best_by_frame: dict[int, Detection] = {}
    for m in members:
        for frame_idx, det in observed[m]:
            best = best_by_frame.get(frame_idx)
            if best is None or _box_rank(det) > _box_rank(best):
                best_by_frame[frame_idx] = det
    if not best_by_frame:
        return None
    start, end = min(best_by_frame), max(best_by_frame)
    entries = [
        TubeEntry(frame_idx=f, detection=best_by_frame.get(f))
        for f in range(start, end + 1)
    ]
    return Tube(tube_id=0, entries=entries, start_frame=start, end_frame=end)


def _observed(tube: Tube) -> Observed:
    return [(e.frame_idx, e.detection) for e in tube.entries if e.detection is not None]


def _area(det: Detection) -> float:
    return det.w * det.h


def _box_rank(det: Detection) -> tuple[float, float, float]:
    """Pick order among boxes on the same frame: larger area, then higher
    confidence, then larger cx — a total order, so the choice is independent of
    the input tube order (no silent nondeterminism on equal-area ties)."""
    return (_area(det), det.confidence, det.cx)
```

- [ ] **Step 4: Run the merge tests; expect pass**

```bash
cd lib/bbox-tube-temporal
uv run pytest tests/test_tubes.py::TestMergeColocatedTubes -v
```
Expected: 9 passed.

- [ ] **Step 5: Run all lib tests**

```bash
cd lib/bbox-tube-temporal
uv run pytest tests/ -v
```
Expected: all existing + 9 new pass (no regressions).

- [ ] **Step 6: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add lib/bbox-tube-temporal/src/bbox_tube_temporal/tubes.py lib/bbox-tube-temporal/tests/test_tubes.py
git commit -m "feat(bbox-tube-temporal): merge_colocated_tubes — post-hoc co-located fragment merge"
```

---

## Task 2: Export `merge_colocated_tubes` from the package

**Files:**
- Modify: `lib/bbox-tube-temporal/src/bbox_tube_temporal/__init__.py`
- Test: `lib/bbox-tube-temporal/tests/test_tubes.py` (add one import test)

- [ ] **Step 1: Add failing top-level import test** — append to `tests/test_tubes.py`

```python
def test_merge_colocated_tubes_is_top_level_export():
    from bbox_tube_temporal import merge_colocated_tubes  # noqa: F401
```

- [ ] **Step 2: Run; expect failure**

```bash
cd lib/bbox-tube-temporal
uv run pytest tests/test_tubes.py::test_merge_colocated_tubes_is_top_level_export -v
```
Expected: `ImportError: cannot import name 'merge_colocated_tubes' from 'bbox_tube_temporal'`.

- [ ] **Step 3: Replace `__init__.py`** with the explicit re-export

```python
from .tubes import build_tubes, merge_colocated_tubes

__all__ = ["build_tubes", "merge_colocated_tubes"]
```

- [ ] **Step 4: Run; expect pass**

```bash
cd lib/bbox-tube-temporal
uv run pytest tests/test_tubes.py::test_merge_colocated_tubes_is_top_level_export -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add lib/bbox-tube-temporal/src/bbox_tube_temporal/__init__.py lib/bbox-tube-temporal/tests/test_tubes.py
git commit -m "feat(bbox-tube-temporal): export merge_colocated_tubes from package"
```

---

## Task 3: `build_tubes_for_inference` orchestrator in `inference.py`

This isolates the build → (optional merge) → filter pipeline so `model.predict` becomes a thin caller and the orchestration is unit-testable.

**Files:**
- Modify: `lib/bbox-tube-temporal/src/bbox_tube_temporal/inference.py`
- Test: `lib/bbox-tube-temporal/tests/test_inference_units.py`

- [ ] **Step 1: Append the failing test** to `tests/test_inference_units.py`

```python
class TestBuildTubesForInference:
    def _fd(self, idx, dets):
        from bbox_tube_temporal.types import FrameDetections
        return FrameDetections(frame_idx=idx, frame_id=str(idx), timestamp=None, detections=dets)

    def _d(self, cx, cy, w=0.05, h=0.05):
        from bbox_tube_temporal.types import Detection
        return Detection(class_id=0, cx=cx, cy=cy, w=w, h=h, confidence=0.9)

    def test_without_merge_keys_matches_legacy(self):
        """No merge keys -> behaves like the legacy build + filter_and_interpolate path."""
        from bbox_tube_temporal.inference import (
            build_tubes_for_inference,
            filter_and_interpolate_tubes,
        )
        from bbox_tube_temporal.tubes import build_tubes
        fds = [self._fd(i, [self._d(0.5, 0.5)]) for i in range(5)]
        legacy = filter_and_interpolate_tubes(
            build_tubes(fds, iou_threshold=0.2, max_misses=2),
            min_tube_length=2, min_detected_entries=2, interpolate_gaps=True,
        )
        new = build_tubes_for_inference(
            fds, iou_threshold=0.2, max_misses=2,
            min_tube_length=2, min_detected_entries=2, interpolate_gaps=True,
            merge_iomin=None, merge_prox_factor=None, merge_max_gap=None,
        )
        assert len(new) == len(legacy)
        assert [(t.start_frame, t.end_frame, len(t.entries)) for t in new] == \
               [(t.start_frame, t.end_frame, len(t.entries)) for t in legacy]

    def test_with_merge_keys_fuses_fragmented_plume(self):
        """A growing plume that build_tubes splits at the IoU jump merges back to one."""
        from bbox_tube_temporal.inference import build_tubes_for_inference
        fds = [
            self._fd(0, [self._d(0.5, 0.5, 0.02, 0.02)]),
            self._fd(1, [self._d(0.5, 0.5, 0.02, 0.02)]),
            self._fd(2, [self._d(0.5, 0.5, 0.2, 0.2)]),
            self._fd(3, [self._d(0.5, 0.5, 0.2, 0.2)]),
        ]
        tubes = build_tubes_for_inference(
            fds, iou_threshold=0.2, max_misses=2,
            min_tube_length=2, min_detected_entries=2, interpolate_gaps=True,
            merge_iomin=0.3, merge_prox_factor=1.0, merge_max_gap=10,
        )
        assert len(tubes) == 1

    def test_partial_merge_keys_treated_as_no_merge(self):
        """If any merge key is None, merging is skipped (backward-compat guard)."""
        from bbox_tube_temporal.inference import build_tubes_for_inference
        fds = [
            self._fd(0, [self._d(0.5, 0.5, 0.02, 0.02)]),
            self._fd(1, [self._d(0.5, 0.5, 0.02, 0.02)]),
            self._fd(2, [self._d(0.5, 0.5, 0.2, 0.2)]),
            self._fd(3, [self._d(0.5, 0.5, 0.2, 0.2)]),
        ]
        tubes = build_tubes_for_inference(
            fds, iou_threshold=0.2, max_misses=2,
            min_tube_length=2, min_detected_entries=2, interpolate_gaps=True,
            merge_iomin=0.3, merge_prox_factor=None, merge_max_gap=10,
        )
        assert len(tubes) == 2  # would be 1 if merge ran
```

- [ ] **Step 2: Run; expect failure**

```bash
cd lib/bbox-tube-temporal
uv run pytest tests/test_inference_units.py::TestBuildTubesForInference -v
```
Expected: `ImportError: cannot import name 'build_tubes_for_inference'`.

- [ ] **Step 3: Add `build_tubes_for_inference`** at the end of `lib/bbox-tube-temporal/src/bbox_tube_temporal/inference.py`

```python
def build_tubes_for_inference(
    frame_detections: list[FrameDetections],
    *,
    iou_threshold: float,
    max_misses: int,
    min_tube_length: int,
    min_detected_entries: int,
    interpolate_gaps: bool,
    merge_iomin: float | None,
    merge_prox_factor: float | None,
    merge_max_gap: int | None,
) -> list[Tube]:
    """Run the full inference tube pipeline.

    When all three ``merge_*`` thresholds are provided, the merge pass runs
    between two filter steps (filter-before-merge prevents resurrecting
    sub-threshold noise; the second filter+interpolate prepares the tubes for
    the classifier). If any ``merge_*`` is ``None`` the merge is skipped and
    the pipeline collapses to the legacy ``build_tubes`` → single
    ``filter_and_interpolate_tubes`` path (bit-for-bit backward compatible).
    """
    from .tubes import build_tubes, merge_colocated_tubes  # noqa: PLC0415

    tubes = build_tubes(
        frame_detections, iou_threshold=iou_threshold, max_misses=max_misses
    )
    merge_active = (
        merge_iomin is not None
        and merge_prox_factor is not None
        and merge_max_gap is not None
    )
    if merge_active:
        tubes = filter_and_interpolate_tubes(
            tubes,
            min_tube_length=min_tube_length,
            min_detected_entries=min_detected_entries,
            interpolate_gaps=False,
        )
        tubes = merge_colocated_tubes(
            tubes,
            merge_iomin=merge_iomin,
            merge_prox_factor=merge_prox_factor,
            merge_max_gap=merge_max_gap,
        )
    return filter_and_interpolate_tubes(
        tubes,
        min_tube_length=min_tube_length,
        min_detected_entries=min_detected_entries,
        interpolate_gaps=interpolate_gaps,
    )
```

Lift the in-function imports to the top of `inference.py` instead — at the top of the file, replace the existing import of `.tubes` (it imports `interpolate_gaps as _interpolate_gaps`) with:

```python
from .tubes import build_tubes, interpolate_gaps as _interpolate_gaps, merge_colocated_tubes
```

and remove the `# noqa: PLC0415` import block from inside `build_tubes_for_inference`. (The PLC0415 ruff rule forbids function-level imports.)

- [ ] **Step 4: Run; expect pass**

```bash
cd lib/bbox-tube-temporal
uv run pytest tests/test_inference_units.py::TestBuildTubesForInference -v
```
Expected: 3 passed.

- [ ] **Step 5: Run all lib tests**

```bash
cd lib/bbox-tube-temporal
uv run pytest tests/ -v
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add lib/bbox-tube-temporal/src/bbox_tube_temporal/inference.py lib/bbox-tube-temporal/tests/test_inference_units.py
git commit -m "feat(bbox-tube-temporal): build_tubes_for_inference orchestrator (optional merge)"
```

---

## Task 4: Wire the orchestrator into `model.py::predict`

**Files:**
- Modify: `lib/bbox-tube-temporal/src/bbox_tube_temporal/model.py:184-203`
- Test: `lib/bbox-tube-temporal/tests/test_model_edge_cases.py` or `tests/test_model_parity.py`

- [ ] **Step 1: Add a parity test** — append to `tests/test_model_edge_cases.py`

```python
def test_predict_unchanged_when_config_has_no_merge_keys(tmp_path):
    """A config without merge keys must produce the same predict() output as
    before this feature (no silent behavior change on existing model packages).
    """
    # Build two pipelines from the same inputs and compare their kept tubes.
    from bbox_tube_temporal.inference import (
        build_tubes_for_inference,
        filter_and_interpolate_tubes,
    )
    from bbox_tube_temporal.tubes import build_tubes
    from bbox_tube_temporal.types import Detection, FrameDetections

    fds = [
        FrameDetections(
            frame_idx=i, frame_id=str(i), timestamp=None,
            detections=[Detection(class_id=0, cx=0.5, cy=0.5, w=0.1, h=0.1, confidence=0.8)],
        )
        for i in range(5)
    ]
    legacy_kept = filter_and_interpolate_tubes(
        build_tubes(fds, iou_threshold=0.2, max_misses=2),
        min_tube_length=2, min_detected_entries=2, interpolate_gaps=True,
    )
    new_kept = build_tubes_for_inference(
        fds, iou_threshold=0.2, max_misses=2,
        min_tube_length=2, min_detected_entries=2, interpolate_gaps=True,
        merge_iomin=None, merge_prox_factor=None, merge_max_gap=None,
    )
    assert [(t.start_frame, t.end_frame, len(t.entries)) for t in new_kept] == \
           [(t.start_frame, t.end_frame, len(t.entries)) for t in legacy_kept]
```

- [ ] **Step 2: Run the test as-is; expect pass** (the orchestrator already preserves legacy behavior from Task 3)

```bash
cd lib/bbox-tube-temporal
uv run pytest tests/test_model_edge_cases.py::test_predict_unchanged_when_config_has_no_merge_keys -v
```
Expected: PASS.

- [ ] **Step 3: Replace the tube-pipeline block in `model.py::predict`**

In `lib/bbox-tube-temporal/src/bbox_tube_temporal/model.py`, find the block (around lines 184-203):

```python
        frame_dets = run_yolo_on_frames(
            self._yolo,
            truncated,
            confidence_threshold=infer["confidence_threshold"],
            iou_nms=infer["iou_nms"],
            image_size=infer["image_size"],
            device=self._device,
        )

        candidate_tubes = build_tubes(
            frame_dets,
            iou_threshold=tubes_cfg["iou_threshold"],
            max_misses=tubes_cfg["max_misses"],
        )
        kept = filter_and_interpolate_tubes(
            candidate_tubes,
            min_tube_length=tubes_cfg["infer_min_tube_length"],
            min_detected_entries=tubes_cfg["min_detected_entries"],
            interpolate_gaps=tubes_cfg["interpolate_gaps"],
        )
```

Replace with:

```python
        frame_dets = run_yolo_on_frames(
            self._yolo,
            truncated,
            confidence_threshold=infer["confidence_threshold"],
            iou_nms=infer["iou_nms"],
            image_size=infer["image_size"],
            device=self._device,
        )

        # Pre-merge (raw) candidates count, for the details JSON.
        candidate_tubes = build_tubes(
            frame_dets,
            iou_threshold=tubes_cfg["iou_threshold"],
            max_misses=tubes_cfg["max_misses"],
        )
        kept = build_tubes_for_inference(
            frame_dets,
            iou_threshold=tubes_cfg["iou_threshold"],
            max_misses=tubes_cfg["max_misses"],
            min_tube_length=tubes_cfg["infer_min_tube_length"],
            min_detected_entries=tubes_cfg["min_detected_entries"],
            interpolate_gaps=tubes_cfg["interpolate_gaps"],
            merge_iomin=tubes_cfg.get("merge_iomin"),
            merge_prox_factor=tubes_cfg.get("merge_prox_factor"),
            merge_max_gap=tubes_cfg.get("merge_max_gap"),
        )
```

`candidate_tubes` stays (used by `_make_details(num_candidates=len(candidate_tubes))` later). The `kept` line now goes through the new orchestrator.

Also update the imports at the top of `model.py` — find:

```python
from .inference import (
    crop_tube_patches,
    filter_and_interpolate_tubes,
    find_first_crossing_trigger,
    pad_frames_symmetrically,
    pad_frames_uniform,
    run_yolo_on_frames,
    score_tubes,
)
```

Replace with:

```python
from .inference import (
    build_tubes_for_inference,
    crop_tube_patches,
    filter_and_interpolate_tubes,
    find_first_crossing_trigger,
    pad_frames_symmetrically,
    pad_frames_uniform,
    run_yolo_on_frames,
    score_tubes,
)
```

(`filter_and_interpolate_tubes` is no longer called directly by `predict`, but other paths may use it — leave it imported to keep this change surgical. Ruff F401 will flag if it's truly unused; remove it then.)

- [ ] **Step 4: Run lib tests; expect pass + verify F401**

```bash
cd lib/bbox-tube-temporal
uv run pytest tests/ -v
uv run ruff check .
```
Expected: all tests pass; ruff clean (remove `filter_and_interpolate_tubes` from the import in `model.py` if F401 flags it).

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add lib/bbox-tube-temporal/src/bbox_tube_temporal/model.py lib/bbox-tube-temporal/tests/test_model_edge_cases.py
git commit -m "feat(bbox-tube-temporal): predict uses build_tubes_for_inference (optional merge)"
```

---

## Task 5: Lib full-suite verification

- [ ] **Step 1: Lint + format + test the whole lib**

```bash
cd lib/bbox-tube-temporal
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
```
Expected: ruff clean; all existing tests + 13 new (9 merge + 1 export + 3 orchestrator + 1 parity) PASS.

- [ ] **Step 2 (optional cleanup commit)** — only if `ruff format` reflowed anything:

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add lib/bbox-tube-temporal/
git commit -m "style(bbox-tube-temporal): ruff format"
```

---

## Task 6: Experiment `params.yaml` — add merge thresholds

**Files:**
- Modify: `experiments/temporal-models/bbox-tube-temporal/params.yaml:16-18`

- [ ] **Step 1: Extend the `tubes:` block**

In `experiments/temporal-models/bbox-tube-temporal/params.yaml`, find:

```yaml
tubes:
  iou_threshold: 0.2
  max_misses: 2
```

Replace with:

```yaml
tubes:
  iou_threshold: 0.2
  max_misses: 2
  # Post-build merge thresholds (from PR #72). When all three are present, the
  # build_tubes stage and the packaged model both run the merge. Set any to
  # `null` to disable the merge (reverts to legacy behavior).
  merge_iomin: 0.3
  merge_prox_factor: 1.0
  merge_max_gap: 10
```

- [ ] **Step 2: Sanity-load**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run python -c "import yaml; print(yaml.safe_load(open('params.yaml'))['tubes'])"
```
Expected: a dict with `iou_threshold`, `max_misses`, `merge_iomin: 0.3`, `merge_prox_factor: 1.0`, `merge_max_gap: 10`.

- [ ] **Step 3: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/temporal-models/bbox-tube-temporal/params.yaml
git commit -m "feat(bbox-tube-temporal): add merge thresholds to params.yaml::tubes"
```

---

## Task 7: Apply merge in `scripts/build_tubes.py` (training tubes)

**Files:**
- Modify: `experiments/temporal-models/bbox-tube-temporal/scripts/build_tubes.py`
- Test: `experiments/temporal-models/bbox-tube-temporal/tests/test_build_tubes_script.py` (new)

- [ ] **Step 1: Create the failing test** — `tests/test_build_tubes_script.py`

```python
"""Unit tests for the build_tubes.py CLI's per-sequence processing."""

import sys
from pathlib import Path

import pytest

# scripts/ is not a package; load the module by path.
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_tubes.py"


@pytest.fixture(scope="module")
def script_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_tubes_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_tubes_script"] = module
    spec.loader.exec_module(module)
    return module


def _write_wf_sequence(root: Path, name: str, lines_per_frame: list[list[str]]) -> Path:
    """Create a tiny 'wildfire' GT sequence under root/wildfire/<name>/labels."""
    seq = root / "wildfire" / name
    (seq / "labels").mkdir(parents=True)
    for i, lines in enumerate(lines_per_frame):
        (seq / "labels" / f"{name}_2024-01-01T00-00-{i:02d}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
    return seq


def test_merge_keys_fuse_a_growing_plume(tmp_path, script_module):
    """Two frames of a tiny box then two of a big box -> the merge unites them
    into one tube longer than what the no-merge path would select.
    """
    # 5-col GT format: "class cx cy w h" (no confidence)
    seq = _write_wf_sequence(
        tmp_path,
        "seq01",
        lines_per_frame=[
            ["0 0.5 0.5 0.02 0.02"],
            ["0 0.5 0.5 0.02 0.02"],
            ["0 0.5 0.5 0.20 0.20"],
            ["0 0.5 0.5 0.20 0.20"],
        ],
    )
    record_with, _ = script_module._process_sequence(
        seq, split="train",
        iou_threshold=0.2, max_misses=2,
        min_tube_length=2, min_detected_entries=2,
        merge_iomin=0.3, merge_prox_factor=1.0, merge_max_gap=10,
    )
    record_without, _ = script_module._process_sequence(
        seq, split="train",
        iou_threshold=0.2, max_misses=2,
        min_tube_length=2, min_detected_entries=2,
        merge_iomin=None, merge_prox_factor=None, merge_max_gap=None,
    )
    assert record_with is not None and record_without is not None
    with_tube = record_with["tube"]
    without_tube = record_without["tube"]
    # Merge yields a single tube spanning all 4 frames; legacy picks one of the
    # two fragments (length 2).
    assert with_tube["end_frame"] - with_tube["start_frame"] + 1 == 4
    assert without_tube["end_frame"] - without_tube["start_frame"] + 1 == 2
```

- [ ] **Step 2: Run; expect failure**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run pytest tests/test_build_tubes_script.py -v
```
Expected: failure — `_process_sequence` does not accept `merge_iomin` / `merge_prox_factor` / `merge_max_gap`.

- [ ] **Step 3: Update `scripts/build_tubes.py`** — change three things

(a) imports at the top — extend to import the merge function:

```python
from bbox_tube_temporal.tubes import (
    build_tubes,
    interpolate_gaps,
    merge_colocated_tubes,
    select_longest_tube,
)
```

(b) `_process_sequence` signature + body — insert the optional merge between `build_tubes` and `select_longest_tube`:

```python
def _process_sequence(
    seq_dir: Path,
    *,
    split: str,
    iou_threshold: float,
    max_misses: int,
    min_tube_length: int,
    min_detected_entries: int,
    merge_iomin: float | None,
    merge_prox_factor: float | None,
    merge_max_gap: int | None,
) -> tuple[dict | None, str | None]:
    """Process a single sequence.

    Returns ``(record_or_None, drop_reason_or_None)``.
    """
    is_wf = is_wf_sequence(seq_dir)
    label = "smoke" if is_wf else "fp"
    source = "gt" if is_wf else "yolo"

    if not (seq_dir / "labels").is_dir():
        return None, "no_labels_dir"

    fdets = load_frame_detections(seq_dir)
    if not fdets:
        return None, "no_frames"

    total_dets = sum(len(fd.detections) for fd in fdets)
    if total_dets < min_detected_entries:
        return None, "no_detections"

    tubes = build_tubes(fdets, iou_threshold=iou_threshold, max_misses=max_misses)
    if not tubes:
        return None, "no_tubes"

    # Fuse fragments of the same plume before picking the longest, so the
    # selection sees the merged plume rather than one of its pieces.
    if (
        merge_iomin is not None
        and merge_prox_factor is not None
        and merge_max_gap is not None
    ):
        tubes = merge_colocated_tubes(
            tubes,
            merge_iomin=merge_iomin,
            merge_prox_factor=merge_prox_factor,
            merge_max_gap=merge_max_gap,
        )

    selected = select_longest_tube(tubes)
    assert selected is not None  # tubes is non-empty

    length = selected.end_frame - selected.start_frame + 1
    if length < min_tube_length:
        return None, "too_short"

    n_observed = sum(1 for e in selected.entries if e.detection is not None)
    if n_observed < min_detected_entries:
        return None, "too_few_observed"

    interpolate_gaps(selected)

    frame_id_by_idx = {fd.frame_idx: fd.frame_id for fd in fdets}
    record = _serialize_tube(
        sequence_id=seq_dir.name,
        split=split,
        label=label,
        source=source,
        num_frames=len(fdets),
        tube=selected,
        frame_id_by_idx=frame_id_by_idx,
    )
    return record, None
```

(c) `main()` — extend the argparser and thread the new args:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--max-misses", type=int, default=2)
    parser.add_argument("--min-tube-length", type=int, default=4)
    parser.add_argument("--min-detected-entries", type=int, default=2)
    parser.add_argument(
        "--merge-iomin", type=float, default=None,
        help="Post-build merge: containment threshold. Omit to disable merge.",
    )
    parser.add_argument(
        "--merge-prox-factor", type=float, default=None,
        help="Post-build merge: centers-within-N box-sizes factor. Omit to disable merge.",
    )
    parser.add_argument(
        "--merge-max-gap", type=int, default=None,
        help="Post-build merge: max frames between observed spans. Omit to disable merge.",
    )
    args = parser.parse_args()

    split = args.input_dir.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = list_sequences(args.input_dir)
    written = 0
    by_label: dict[str, int] = {"smoke": 0, "fp": 0}
    dropped: list[DropRecord] = []

    for seq_dir in seq_dirs:
        record, reason = _process_sequence(
            seq_dir,
            split=split,
            iou_threshold=args.iou_threshold,
            max_misses=args.max_misses,
            min_tube_length=args.min_tube_length,
            min_detected_entries=args.min_detected_entries,
            merge_iomin=args.merge_iomin,
            merge_prox_factor=args.merge_prox_factor,
            merge_max_gap=args.merge_max_gap,
        )
        if reason is not None:
            dropped.append(DropRecord(sequence_id=seq_dir.name, reason=reason))
            continue

        out_path = args.output_dir / f"{seq_dir.name}.json"
        out_path.write_text(json.dumps(record, indent=2))
        written += 1
        by_label[record["label"]] += 1

    summary = {
        "split": split,
        "total_sequences": len(seq_dirs),
        "tubes_written": written,
        "by_label": by_label,
        "dropped": [
            {"sequence_id": d.sequence_id, "reason": d.reason} for d in dropped
        ],
    }
    (args.output_dir / "_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"[{split}] wrote {written}/{len(seq_dirs)} tubes "
        f"(smoke={by_label['smoke']}, fp={by_label['fp']}, "
        f"dropped={len(dropped)})"
    )
```

- [ ] **Step 4: Run; expect pass**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run pytest tests/test_build_tubes_script.py -v
```
Expected: PASS.

- [ ] **Step 5: Run all experiment tests**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run pytest tests/ -v
```
Expected: all PASS (no regressions in the existing 20+ tests).

- [ ] **Step 6: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/temporal-models/bbox-tube-temporal/scripts/build_tubes.py experiments/temporal-models/bbox-tube-temporal/tests/test_build_tubes_script.py
git commit -m "feat(bbox-tube-temporal): apply merge in build_tubes script before select_longest"
```

---

## Task 8: DVC `build_tubes` stage — pass merge args + params

**Files:**
- Modify: `experiments/temporal-models/bbox-tube-temporal/dvc.yaml:34-57`

- [ ] **Step 1: Update the stage**

In `experiments/temporal-models/bbox-tube-temporal/dvc.yaml`, find the `build_tubes` stage and replace it with:

```yaml
  build_tubes:
    foreach:
      - train
      - val
    do:
      cmd: >-
        uv run python scripts/build_tubes.py
        --input-dir data/01_raw/datasets/${item}
        --output-dir data/03_primary/tubes/${item}
        --iou-threshold ${tubes.iou_threshold}
        --max-misses ${tubes.max_misses}
        --min-tube-length ${build_tubes.min_tube_length}
        --min-detected-entries ${build_tubes.min_detected_entries}
        --merge-iomin ${tubes.merge_iomin}
        --merge-prox-factor ${tubes.merge_prox_factor}
        --merge-max-gap ${tubes.merge_max_gap}
      deps:
        - scripts/build_tubes.py
        - src/bbox_tube_temporal/tubes.py
        - src/bbox_tube_temporal/data.py
        - src/bbox_tube_temporal/types.py
        - data/01_raw/datasets/${item}
      params:
        - tubes
        - build_tubes
      outs:
        - data/03_primary/tubes/${item}
```

(Two changes: three new CLI args, no other change. The `params: [tubes, build_tubes]` already covers the new `tubes.merge_*` keys, so DVC will re-run when they change.)

- [ ] **Step 2: Validate the dvc.yaml**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run dvc dag --quiet || uv run dvc status -q
```
Expected: no parser errors. (Both commands tolerate missing data; we just want the YAML to parse.)

- [ ] **Step 3: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/temporal-models/bbox-tube-temporal/dvc.yaml
git commit -m "build(bbox-tube-temporal): thread merge args through dvc build_tubes stage"
```

---

## Task 9: Embed merge keys in the packaged model config

**Files:**
- Modify: `experiments/temporal-models/bbox-tube-temporal/scripts/package_model.py:94-114`

- [ ] **Step 1: Update `_build_config`** — extend the `tubes` block to include the merge keys when present in `params.yaml`

In `experiments/temporal-models/bbox-tube-temporal/scripts/package_model.py`, find the `_build_config` function body and replace the `"tubes"` dict literal (lines ~96-103) with:

```python
        "tubes": _tubes_config(all_params),
```

Then add this helper above `_build_config`:

```python
def _tubes_config(all_params: dict) -> dict:
    """Build the packaged ``config["tubes"]`` block.

    Includes the post-build merge thresholds only when all three are present in
    ``params.yaml::tubes``; missing/null keys mean "no merge" and are omitted
    from the packaged config (preserves backward-compat semantics for older
    consumers that don't know about merge keys).
    """
    tubes_params = all_params["tubes"]
    build_tubes_params = all_params["build_tubes"]
    cfg: dict = {
        "iou_threshold": tubes_params["iou_threshold"],
        "max_misses": tubes_params["max_misses"],
        "min_tube_length": build_tubes_params["min_tube_length"],
        "infer_min_tube_length": all_params["package"]["infer_min_tube_length"],
        "min_detected_entries": build_tubes_params["min_detected_entries"],
        "interpolate_gaps": True,
    }
    merge_iomin = tubes_params.get("merge_iomin")
    merge_prox_factor = tubes_params.get("merge_prox_factor")
    merge_max_gap = tubes_params.get("merge_max_gap")
    if all(v is not None for v in (merge_iomin, merge_prox_factor, merge_max_gap)):
        cfg["merge_iomin"] = float(merge_iomin)
        cfg["merge_prox_factor"] = float(merge_prox_factor)
        cfg["merge_max_gap"] = int(merge_max_gap)
    return cfg
```

- [ ] **Step 2: Verify the new helper is exercised** — add a quick smoke test to `tests/test_package_predict.py` (or create `tests/test_package_model_config.py`):

```python
def test_tubes_config_includes_merge_keys_when_present():
    import sys
    from pathlib import Path
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "package_model.py"
    spec = importlib.util.spec_from_file_location("package_model_script", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["package_model_script"] = mod
    spec.loader.exec_module(mod)

    all_params = {
        "tubes": {
            "iou_threshold": 0.2, "max_misses": 2,
            "merge_iomin": 0.3, "merge_prox_factor": 1.0, "merge_max_gap": 10,
        },
        "build_tubes": {"min_tube_length": 4, "min_detected_entries": 2},
        "package": {"infer_min_tube_length": 2},
    }
    cfg = mod._tubes_config(all_params)
    assert cfg["merge_iomin"] == 0.3
    assert cfg["merge_prox_factor"] == 1.0
    assert cfg["merge_max_gap"] == 10


def test_tubes_config_omits_merge_keys_when_any_missing():
    import sys
    from pathlib import Path
    import importlib.util
    script = Path(__file__).resolve().parents[1] / "scripts" / "package_model.py"
    spec = importlib.util.spec_from_file_location("package_model_script_v2", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["package_model_script_v2"] = mod
    spec.loader.exec_module(mod)

    all_params = {
        "tubes": {"iou_threshold": 0.2, "max_misses": 2, "merge_iomin": 0.3},
        "build_tubes": {"min_tube_length": 4, "min_detected_entries": 2},
        "package": {"infer_min_tube_length": 2},
    }
    cfg = mod._tubes_config(all_params)
    assert "merge_iomin" not in cfg
    assert "merge_prox_factor" not in cfg
    assert "merge_max_gap" not in cfg
```

(Save as `tests/test_package_model_script.py` — new file.)

- [ ] **Step 3: Run; expect pass**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run pytest tests/test_package_model_script.py -v
```
Expected: 2 passed.

- [ ] **Step 4: Run all experiment tests + lint**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format --check .
```
Expected: all green, ruff clean.

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/temporal-models/bbox-tube-temporal/scripts/package_model.py experiments/temporal-models/bbox-tube-temporal/tests/test_package_model_script.py
git commit -m "feat(bbox-tube-temporal): embed merge thresholds in packaged config when present"
```

---

## Task 10: Operator retrain + evaluation

> Steps 1–6 are **operator-run** (need DVC data + GPU + several hours). The
> agent has produced the code changes; the operator runs the pipeline and
> reports the results. Do not run `dvc pull`.

- [ ] **Step 1 (operator): pull the training data**

```bash
cd experiments/temporal-models/bbox-tube-temporal
# pull data/01_raw/datasets/{train,val}, data/01_raw/models/best.pt, etc.
uv run dvc pull
```
Expected: tracked data resolved; `data/01_raw/datasets/{train,val}` and `data/01_raw/models/best.pt` populated.

- [ ] **Step 2 (operator): rebuild the tube training data with the merge**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run dvc repro build_tubes
```
Expected: `data/03_primary/tubes/{train,val}` regenerated. Compare counts vs. before via the `_summary.json` written by the script.

- [ ] **Step 3 (operator): rebuild model input crops + retrain `vit_dinov2_finetune`**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run dvc repro build_model_input
uv run dvc repro train_vit_dinov2_finetune
```
Expected: new `data/06_models/vit_dinov2_finetune/best_checkpoint.pt`; the existing CSV logs + training-curve plot updated.

- [ ] **Step 4 (operator): evaluate, analyze, package**

```bash
cd experiments/temporal-models/bbox-tube-temporal
uv run dvc repro evaluate_vit_dinov2_finetune
uv run dvc repro analyze_variant@vit_dinov2_finetune
uv run dvc repro package@vit_dinov2_finetune
uv run dvc repro evaluate_packaged@vit_dinov2_finetune-train
uv run dvc repro evaluate_packaged@vit_dinov2_finetune-val
```
Expected: new `data/06_models/vit_dinov2_finetune/model.zip` whose bundled `config.yaml::tubes` contains `merge_iomin: 0.3`, `merge_prox_factor: 1.0`, `merge_max_gap: 10`. New metrics in `data/08_reporting/{train,val}/{vit_dinov2_finetune,packaged/vit_dinov2_finetune}/metrics.json`.

- [ ] **Step 5 (operator): verify the bundled config**

```bash
python3 - <<'PY'
import zipfile, yaml
cfg = yaml.safe_load(zipfile.ZipFile("data/06_models/vit_dinov2_finetune/model.zip").read("config.yaml"))
print(cfg["tubes"])
assert cfg["tubes"]["merge_iomin"] == 0.3
assert cfg["tubes"]["merge_prox_factor"] == 1.0
assert cfg["tubes"]["merge_max_gap"] == 10
print("OK: merge keys present in packaged config")
PY
```
Expected: prints the tubes config and "OK".

- [ ] **Step 6 (operator): compare metrics and DVC push**

Read `data/08_reporting/val/packaged/vit_dinov2_finetune/metrics.json` and compare against the same path on `main` (the pre-merge baseline). Capture a one-line summary (precision/recall/F1) in the PR description.

```bash
uv run dvc push
```

- [ ] **Step 7: Commit data pointers + open PR**

```bash
cd /mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/vision-rd
git add experiments/temporal-models/bbox-tube-temporal/dvc.lock
git commit -m "build(bbox-tube-temporal): retrain vit_dinov2_finetune on merge-aware tubes"
git push -u origin arthur/feat-bbox-tube-merge-retrain
gh pr create --base main --head arthur/feat-bbox-tube-merge-retrain --title "..." --body "..."
```

---

## Self-Review

**Spec coverage**

- Lib: add `merge_colocated_tubes` + helpers (spec §"Lib changes::tubes.py") → Task 1. ✓
- Lib: top-level export (spec §"Lib changes::tubes.py") → Task 2. ✓
- Lib: `model.py::predict` new pipeline order via orchestrator (spec §"Lib changes::model.py::predict") → Tasks 3 + 4. ✓
- Lib: optional config keys, backward compat (spec §"Packaged config schema") → covered by `build_tubes_for_inference` and the parity test. ✓
- Lib tests mirroring the lab's merge tests (spec §"Tests") → Task 1 (9 merge tests) + Task 4 (parity). ✓
- Experiment `params.yaml` (spec §"Experiment changes::params.yaml") → Task 6. ✓
- Experiment `scripts/build_tubes.py` insert merge (spec §"scripts/build_tubes.py") → Task 7. ✓
- Experiment `dvc.yaml` build_tubes (spec §"dvc.yaml") → Task 8. ✓
- Experiment `scripts/package_model.py` embed merge keys (spec §"package_model.py") → Task 9. ✓
- Retrain + evaluation (spec §"Retrain + evaluation protocol") → Task 10. ✓
- Success criteria (lib tests pass, retrain runs, parity preserved, merge keys present in package) → asserted across Tasks 5, 9 step 2, 10 step 5. ✓

**Placeholder scan:** every step has either exact code, an exact command + expected output, or an operator instruction with concrete files/paths. No TBD/TODO; no "similar to Task N" without repeating code; no vague "handle edge cases". The single deliberately-blank placeholder is the gh PR title/body in Task 10 step 7 — that's a real "operator fills in based on the metrics they just observed". ✓

**Type consistency:** the function `merge_colocated_tubes(tubes, *, merge_iomin, merge_prox_factor, merge_max_gap)` keeps the same signature in every task (1, 3, 7, 9). `build_tubes_for_inference` has the same signature in Tasks 3 and 4. The packaged config key names (`merge_iomin`, `merge_prox_factor`, `merge_max_gap`) match between Tasks 4 (read), 6 (params), 8 (CLI args), and 9 (config emit). ✓
