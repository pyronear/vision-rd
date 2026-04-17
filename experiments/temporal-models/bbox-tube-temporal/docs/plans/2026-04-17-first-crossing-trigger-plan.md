# First-crossing trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `winner.end_frame`-based trigger in `BboxTubeTemporalModel.predict` with a first-crossing trigger, so `trigger_frame_index` reflects the earliest frame the model would have fired. Keep `is_positive` bit-identical under both `max_logit` and `logistic` aggregations.

**Architecture:** New helper `find_first_crossing_trigger` in `inference.py` replaces `pick_winner_and_trigger`. For each tube whose full-length decision is positive (D2 guard), serially score prefixes of increasing length and record the first crossing. Winner is the qualifying tube whose first-crossing `frame_idx` is earliest (tie-break on smallest `tube_id`).

**Tech Stack:** Python 3.11, PyTorch, pytest. Helper lives alongside existing `pick_winner_and_trigger`; callers: only `BboxTubeTemporalModel.predict`.

**Spec reference:** `docs/specs/2026-04-17-first-crossing-trigger-design.md`.

**Working directory:** all commands run from `experiments/temporal-models/bbox-tube-temporal/`.

---

## File Structure

**Modified:**
- `src/bbox_tube_temporal/inference.py` — add `find_first_crossing_trigger`; delete `pick_winner_and_trigger`.
- `src/bbox_tube_temporal/model.py` — swap the call site (line ~211) to the new helper; add `per_tube_first_crossing` to `details{}`.
- `tests/test_inference_units.py` — replace `TestPickWinnerAndTrigger` with `TestFindFirstCrossingTrigger`; drop the stale import.
- `tests/test_model_edge_cases.py` — one existing test asserts `trigger_frame_index is None`; no behavioral change under D2. Add one positive-case assertion on a known fixture.

**Untouched:** `tubes.py`, `temporal_classifier.py`, `package.py` loader, `protocol_eval.py`, training pipeline, calibration, DVC stage graph, `params.yaml`, archive `config.yaml` schema.

---

### Task 1: Add `find_first_crossing_trigger` with `max_logit` aggregation

**Files:**
- Modify: `src/bbox_tube_temporal/inference.py`
- Test: `tests/test_inference_units.py`

The helper handles `max_logit` only in this task; `logistic` path raises `NotImplementedError` and is added in Task 2.

- [ ] **Step 1: Write failing tests for max_logit path**

Append to `tests/test_inference_units.py` after the existing `TestPickWinnerAndTrigger` class:

```python
class TestFindFirstCrossingTrigger:
    """Unit tests for find_first_crossing_trigger (max_logit mode)."""

    def _make_patches_and_masks(
        self, tubes: list[Tube], max_frames: int = 8
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Build dummy patches/masks for tubes; mask reflects tube length."""
        patches: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for t in tubes:
            n = len(t.entries)
            patches.append(torch.zeros(max_frames, 3, 8, 8))
            m = torch.zeros(max_frames, dtype=torch.bool)
            m[:n] = True
            masks.append(m)
        return patches, masks

    def test_empty_tubes_returns_negative(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        res = find_first_crossing_trigger(
            classifier=MagicMock(),
            tubes=[],
            patches_per_tube=[],
            masks_per_tube=[],
            full_logits=torch.zeros(0),
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert res == (False, None, None, {})

    def test_no_qualifying_tubes_returns_negative(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        tubes = [_tube(1, [(0, _det()), (1, _det())])]
        patches, masks = self._make_patches_and_masks(tubes)
        # Full-tube logit below threshold → no qualifying tube.
        full_logits = torch.tensor([-1.0])
        classifier = MagicMock()

        res = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert res == (False, None, None, {})
        # Classifier should not be called at all when no tube qualifies.
        classifier.assert_not_called()

    def test_single_qualifying_tube_crosses_at_min_prefix(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        tubes = [
            _tube(7, [(3, _det()), (4, _det()), (5, _det()), (6, _det())]),
        ]
        patches, masks = self._make_patches_and_masks(tubes)
        # Full-tube logit ≥ threshold (qualifies). Prefix scorer returns
        # >= threshold starting at L=2 (the min).
        full_logits = torch.tensor([1.0])
        # Stub returns 2.0 for any prefix call so crossing happens at L=2.
        classifier = MagicMock(return_value=torch.tensor([2.0]))

        is_positive, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert is_positive is True
        assert winner == 7
        # tube.entries[L-1].frame_idx at L=2 → entry idx 1 → frame_idx = 4
        assert trigger == 4
        assert diag == {7: {"crossing_frame": 4, "prefix_length": 2}}
        # Classifier called exactly once (only L=2 needed — early exit).
        assert classifier.call_count == 1

    def test_earliest_crossing_wins_across_tubes(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        # Tube A starts at frame 5; Tube B starts at frame 0. Both qualify.
        # Both cross at L=2 — B's crossing frame (1) is earlier than A's (6).
        tube_a = _tube(1, [(5, _det()), (6, _det()), (7, _det())])
        tube_b = _tube(2, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube_a, tube_b]
        patches, masks = self._make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0, 1.0])
        classifier = MagicMock(return_value=torch.tensor([2.0]))

        is_positive, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert is_positive is True
        assert winner == 2  # tube B wins on earlier frame
        assert trigger == 1
        assert diag == {
            1: {"crossing_frame": 6, "prefix_length": 2},
            2: {"crossing_frame": 1, "prefix_length": 2},
        }

    def test_tie_on_crossing_frame_breaks_on_smallest_tube_id(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        # Both tubes share the same entry frame indices and cross at L=2
        # → tie on crossing_frame=1. Smallest tube_id (5) should win over 9.
        tube_a = _tube(9, [(0, _det()), (1, _det()), (2, _det())])
        tube_b = _tube(5, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube_a, tube_b]
        patches, masks = self._make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0, 1.0])
        classifier = MagicMock(return_value=torch.tensor([2.0]))

        _, _, winner, _ = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert winner == 5

    def test_d2_guard_ignores_non_qualifying_tube(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        # Tube 1 qualifies (full logit ≥ 0); Tube 2 does not (full logit < 0).
        # Even if the stubbed classifier would cross on a prefix of tube 2
        # earlier, it must be ignored under D2.
        tube_q = _tube(1, [(4, _det()), (5, _det()), (6, _det())])
        tube_skip = _tube(2, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube_q, tube_skip]
        patches, masks = self._make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0, -1.0])
        classifier = MagicMock(return_value=torch.tensor([2.0]))

        _, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert winner == 1
        # tube 1's first crossing is at L=2 → frame_idx 5.
        assert trigger == 5
        # Only tube 1 recorded; tube 2 never scored.
        assert diag == {1: {"crossing_frame": 5, "prefix_length": 2}}

    def test_loop_walks_prefix_lengths_until_crossing(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        tube = _tube(3, [(0, _det()), (1, _det()), (2, _det()), (3, _det())])
        tubes = [tube]
        patches, masks = self._make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0])  # full-tube qualifies
        # Classifier returns below-threshold at L=2, above at L=3.
        # Since at L = full_len the helper reuses full_logits (no call),
        # the sequence of classifier calls is: [L=2, L=3].
        classifier = MagicMock(side_effect=[torch.tensor([-0.5]), torch.tensor([0.5])])

        _, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert winner == 3
        assert trigger == 2  # entry[L-1].frame_idx at L=3 → frame_idx 2
        assert diag == {3: {"crossing_frame": 2, "prefix_length": 3}}
        assert classifier.call_count == 2

    def test_full_length_crossing_reuses_full_logits(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        tube = _tube(1, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube]
        patches, masks = self._make_patches_and_masks(tubes)
        full_logits = torch.tensor([1.0])
        # Classifier returns below threshold at L=2.
        # At L=3 = full_len, helper must reuse full_logits[0] instead of
        # calling the classifier. So classifier is called exactly once
        # (for L=2).
        classifier = MagicMock(return_value=torch.tensor([-0.5]))

        _, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="max_logit",
            threshold=0.0,
            min_prefix_length=2,
        )
        assert winner == 1
        assert trigger == 2
        assert diag == {1: {"crossing_frame": 2, "prefix_length": 3}}
        assert classifier.call_count == 1  # L=2 only; L=3 reused full_logits

    def test_unknown_aggregation_raises(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        tubes = [_tube(1, [(0, _det()), (1, _det())])]
        patches, masks = self._make_patches_and_masks(tubes)
        with pytest.raises(ValueError, match="aggregation"):
            find_first_crossing_trigger(
                classifier=MagicMock(),
                tubes=tubes,
                patches_per_tube=patches,
                masks_per_tube=masks,
                full_logits=torch.tensor([1.0]),
                aggregation="bogus",
                threshold=0.0,
                min_prefix_length=2,
            )
```

Also add this import at the top of the file, keeping existing imports:

```python
from bbox_tube_temporal.types import Tube  # already imported below via _tube helper
```

Verify `Tube` is already imported — check line ~20 where `_tube` helper lives. If not imported already, add it to the existing `from bbox_tube_temporal.types import` line.

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
uv run pytest tests/test_inference_units.py::TestFindFirstCrossingTrigger -v
```

Expected: all tests FAIL with `ImportError: cannot import name 'find_first_crossing_trigger'` (it doesn't exist yet).

- [ ] **Step 3: Implement `find_first_crossing_trigger` — max_logit branch only**

Add to `src/bbox_tube_temporal/inference.py`, directly below `pick_winner_and_trigger`:

```python
def find_first_crossing_trigger(
    *,
    classifier: Any,
    tubes: list[Tube],
    patches_per_tube: list[torch.Tensor],
    masks_per_tube: list[torch.Tensor],
    full_logits: torch.Tensor,
    aggregation: str = "max_logit",
    threshold: float,
    calibrator: LogisticCalibrator | None = None,
    logistic_threshold: float = 0.5,
    min_prefix_length: int,
) -> tuple[bool, int | None, int | None, dict]:
    """Find the earliest frame at which the aggregation rule would have fired.

    Under the full-tube guard (D2), only tubes whose *full-length*
    decision is positive qualify. For each qualifying tube, prefixes of
    length ``L = min_prefix_length .. len(tube.entries)`` are scored
    serially; the first L whose decision is positive gives that tube's
    first-crossing ``frame_idx = tube.entries[L-1].frame_idx``. The
    sequence-level trigger is the qualifying tube with the earliest
    first-crossing frame (tie-break: smallest ``tube_id``).

    ``is_positive`` is preserved bit-for-bit versus
    :func:`pick_winner_and_trigger`: the sequence is positive iff any
    tube's full-length decision is positive.

    Micro-optimisation: at ``L = len(tube.entries)`` the prefix logit is
    exactly ``full_logits[i]`` — we reuse it rather than re-running the
    classifier.

    Args:
        classifier: Callable ``(patches[1,T,3,H,W], mask[1,T]) -> logits[1]``.
        tubes: Kept tubes, aligned with ``full_logits``.
        patches_per_tube: One ``[max_frames, 3, H, W]`` tensor per tube.
        masks_per_tube: One ``[max_frames]`` bool tensor per tube.
        full_logits: Output of :func:`score_tubes` on the full tubes.
        aggregation: ``"max_logit"`` or ``"logistic"``.
        threshold: Raw logit threshold (``max_logit`` only).
        calibrator: Required when ``aggregation == "logistic"``.
        logistic_threshold: Probability threshold (``logistic`` only).
        min_prefix_length: Smallest prefix length to score. Must equal
            the inference-time ``infer_min_tube_length``.

    Returns:
        Tuple ``(is_positive, trigger_frame_index, winner_tube_id,
        per_tube_first_crossing)``. The trailing dict maps
        ``tube_id -> {"crossing_frame": int, "prefix_length": int}`` for
        every qualifying tube.

    Raises:
        ValueError: unknown ``aggregation`` or ``"logistic"`` without a
            calibrator.
    """
    if not tubes:
        return False, None, None, {}

    if aggregation == "max_logit":
        def decides_positive(logit: float, _tube_prefix: Tube, _n_tubes: int) -> bool:
            return logit >= threshold
    elif aggregation == "logistic":
        raise NotImplementedError("logistic aggregation wired in Task 2")
    else:
        raise ValueError(f"unknown aggregation: {aggregation!r}")

    n_tubes = len(tubes)

    # D2 guard: restrict to tubes whose full-length decision is positive.
    qualifying_indices: list[int] = [
        i for i, tube in enumerate(tubes)
        if decides_positive(float(full_logits[i].item()), tube, n_tubes)
    ]
    if not qualifying_indices:
        return False, None, None, {}

    per_tube: dict[int, dict] = {}
    for i in qualifying_indices:
        tube = tubes[i]
        full_len = len(tube.entries)
        assert full_len >= min_prefix_length, (
            f"tube {tube.tube_id} len {full_len} < min_prefix_length {min_prefix_length}"
        )

        patches_i = patches_per_tube[i]
        mask_i = masks_per_tube[i]

        crossed = False
        for L in range(min_prefix_length, full_len + 1):
            if L == full_len:
                prefix_logit = float(full_logits[i].item())
            else:
                prefix_mask = mask_i.clone()
                prefix_mask[L:] = False
                with torch.no_grad():
                    out = classifier(patches_i.unsqueeze(0), prefix_mask.unsqueeze(0))
                prefix_logit = float(out[0].item())

            prefix_entries = tube.entries[:L]
            prefix_tube = Tube(
                tube_id=tube.tube_id,
                entries=prefix_entries,
                start_frame=prefix_entries[0].frame_idx,
                end_frame=prefix_entries[-1].frame_idx,
            )

            if decides_positive(prefix_logit, prefix_tube, n_tubes):
                per_tube[tube.tube_id] = {
                    "crossing_frame": prefix_entries[-1].frame_idx,
                    "prefix_length": L,
                }
                crossed = True
                break

        assert crossed, (
            f"D2 invariant violated: qualifying tube {tube.tube_id} "
            f"produced no crossing prefix"
        )

    winner_tube_id = min(
        per_tube,
        key=lambda tid: (per_tube[tid]["crossing_frame"], tid),
    )
    trigger = per_tube[winner_tube_id]["crossing_frame"]
    return True, trigger, winner_tube_id, per_tube
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_inference_units.py::TestFindFirstCrossingTrigger -v
```

Expected: all tests added in Step 1 PASS.

- [ ] **Step 5: Run full unit-test suite for the module to make sure nothing regressed**

```bash
uv run pytest tests/test_inference_units.py -v
```

Expected: all previously-passing tests still PASS (we haven't touched `pick_winner_and_trigger` yet).

- [ ] **Step 6: Lint**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Expected: no errors. If formatting errors, run `uv run ruff format src/ tests/` then re-check.

- [ ] **Step 7: Commit**

```bash
git add src/bbox_tube_temporal/inference.py tests/test_inference_units.py
git commit -m "feat(bbox-tube-temporal): add find_first_crossing_trigger (max_logit)"
```

---

### Task 2: Extend `find_first_crossing_trigger` with `logistic` aggregation

**Files:**
- Modify: `src/bbox_tube_temporal/inference.py` (replace the `NotImplementedError` branch)
- Test: `tests/test_inference_units.py` (append logistic-mode cases)

- [ ] **Step 1: Write failing tests for logistic path**

Append inside `class TestFindFirstCrossingTrigger` in `tests/test_inference_units.py`:

```python
    def test_logistic_mode_fires_at_first_crossing(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        # One tube of length 3. Calibrator is linear on logit; coef=2.0.
        # Full-tube logit = 2.0 → z=4.0 → sigmoid ≈ 0.98 ≥ 0.5 → qualifies.
        # Prefix logit at L=2 is 0.5 → z=1.0 → sigmoid ≈ 0.73 ≥ 0.5
        # → crosses at L=2.
        tube = _tube(4, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube]
        patches, masks = self._make_patches_and_masks(tubes)
        full_logits = torch.tensor([2.0])
        classifier = MagicMock(return_value=torch.tensor([0.5]))

        cal = LogisticCalibrator(
            features=["logit", "log_len", "mean_conf", "n_tubes"],
            coefficients=np.array([2.0, 0.0, 0.0, 0.0]),
            intercept=0.0,
            sanity_checks=[],
        )

        is_positive, trigger, winner, diag = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="logistic",
            threshold=0.0,  # ignored in logistic branch
            calibrator=cal,
            logistic_threshold=0.5,
            min_prefix_length=2,
        )
        assert is_positive is True
        assert winner == 4
        assert trigger == 1  # L=2 → entry[1].frame_idx = 1
        assert diag == {4: {"crossing_frame": 1, "prefix_length": 2}}

    def test_logistic_mode_no_qualifying_tube(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        tube = _tube(1, [(0, _det()), (1, _det()), (2, _det())])
        tubes = [tube]
        patches, masks = self._make_patches_and_masks(tubes)
        full_logits = torch.tensor([-2.0])  # z=-4 → sigmoid < 0.5
        classifier = MagicMock()

        cal = LogisticCalibrator(
            features=["logit", "log_len", "mean_conf", "n_tubes"],
            coefficients=np.array([2.0, 0.0, 0.0, 0.0]),
            intercept=0.0,
            sanity_checks=[],
        )

        res = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="logistic",
            threshold=0.0,
            calibrator=cal,
            logistic_threshold=0.5,
            min_prefix_length=2,
        )
        assert res == (False, None, None, {})
        classifier.assert_not_called()

    def test_logistic_mode_requires_calibrator(self) -> None:
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        tubes = [_tube(1, [(0, _det()), (1, _det())])]
        patches, masks = self._make_patches_and_masks(tubes)
        with pytest.raises(ValueError, match="calibrator"):
            find_first_crossing_trigger(
                classifier=MagicMock(),
                tubes=tubes,
                patches_per_tube=patches,
                masks_per_tube=masks,
                full_logits=torch.tensor([1.0]),
                aggregation="logistic",
                threshold=0.0,
                calibrator=None,
                logistic_threshold=0.5,
                min_prefix_length=2,
            )

    def test_logistic_log_len_affects_prefix_decision(self) -> None:
        """Ensure the predicate sees the prefix length via log_len, not just the logit."""
        from bbox_tube_temporal.inference import find_first_crossing_trigger

        # Tube of length 4. Calibrator has coef only on log_len (index 1):
        #   z = -2.0 * log_len
        # → at L=2: z = -2 * log(2) ≈ -1.386 → sigmoid ≈ 0.200 (< 0.5, NOT crossing)
        # → at L=3: z = -2 * log(3) ≈ -2.197 → sigmoid ≈ 0.100 (NOT crossing)
        # → at L=4: z = -2 * log(4) ≈ -2.773 → sigmoid ≈ 0.059 (NOT crossing)
        # So even though the classifier returns any logit, this tube never
        # qualifies at full length → no trigger.
        tube = _tube(1, [(0, _det()), (1, _det()), (2, _det()), (3, _det())])
        tubes = [tube]
        patches, masks = self._make_patches_and_masks(tubes)
        full_logits = torch.tensor([10.0])  # logit irrelevant (coef=0)
        classifier = MagicMock(return_value=torch.tensor([10.0]))

        cal = LogisticCalibrator(
            features=["logit", "log_len", "mean_conf", "n_tubes"],
            coefficients=np.array([0.0, -2.0, 0.0, 0.0]),
            intercept=0.0,
            sanity_checks=[],
        )

        res = find_first_crossing_trigger(
            classifier=classifier,
            tubes=tubes,
            patches_per_tube=patches,
            masks_per_tube=masks,
            full_logits=full_logits,
            aggregation="logistic",
            threshold=0.0,
            calibrator=cal,
            logistic_threshold=0.5,
            min_prefix_length=2,
        )
        assert res == (False, None, None, {})
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
uv run pytest tests/test_inference_units.py::TestFindFirstCrossingTrigger -v -k logistic
```

Expected: tests FAIL with `NotImplementedError: logistic aggregation wired in Task 2` or `ValueError` in the `requires_calibrator` test (depending on order of validation).

- [ ] **Step 3: Replace the `NotImplementedError` branch with the logistic predicate**

In `src/bbox_tube_temporal/inference.py`, find the branch:

```python
    elif aggregation == "logistic":
        raise NotImplementedError("logistic aggregation wired in Task 2")
```

Replace with:

```python
    elif aggregation == "logistic":
        if calibrator is None:
            raise ValueError("aggregation='logistic' requires a fitted calibrator")

        def decides_positive(logit: float, tube_prefix: Tube, n_tubes: int) -> bool:
            tube_dict = {
                "logit": logit,
                "start_frame": tube_prefix.start_frame,
                "end_frame": tube_prefix.end_frame,
                "entries": [
                    {
                        "confidence": (
                            e.detection.confidence
                            if e.detection is not None
                            else None
                        )
                    }
                    for e in tube_prefix.entries
                ],
            }
            features = extract_features(tube_dict, n_tubes=n_tubes)
            return bool(calibrator.predict_proba(features) >= logistic_threshold)
```

Note: `extract_features` is already imported at the top of `inference.py` (see existing `pick_winner_and_trigger`).

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest tests/test_inference_units.py::TestFindFirstCrossingTrigger -v
```

Expected: all tests (max_logit + logistic) PASS.

- [ ] **Step 5: Lint**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/bbox_tube_temporal/inference.py tests/test_inference_units.py
git commit -m "feat(bbox-tube-temporal): support logistic aggregation in find_first_crossing_trigger"
```

---

### Task 3: Wire `find_first_crossing_trigger` into `model.py`

**Files:**
- Modify: `src/bbox_tube_temporal/model.py`
- Test: `tests/test_model_edge_cases.py`

Replace the `pick_winner_and_trigger` call at line ~211 with the new helper. Capture the `per_tube_first_crossing` diagnostic into `details{}`.

- [ ] **Step 1: Write a failing positive-path assertion in the model tests**

Read `tests/test_model_edge_cases.py` first to understand how the test builds `BboxTubeTemporalModel` and runs `predict()`. Look for an existing positive-path test (a test that produces `is_positive=True`). If one exists, add a new assertion to it. If no positive-path test exists yet, add a new test modeled on the existing negative-path tests but with a stubbed classifier returning a logit that qualifies, and assert:

```python
assert out.is_positive is True
assert out.trigger_frame_index is not None
# The new trigger must never exceed the winner tube's end_frame
# (monotone-decrease guarantee vs. the old end_frame rule).
winner_id = out.details["winner_tube_id"]
winner_tube = next(t for t in out.details["kept_tubes"] if t["tube_id"] == winner_id)
assert out.trigger_frame_index <= winner_tube["end_frame"]
# per_tube_first_crossing should be populated for the winner.
assert winner_id in out.details["per_tube_first_crossing"]
```

Name the test `test_first_crossing_trigger_never_exceeds_end_frame`.

Exact scaffolding depends on how the existing tests build the model — mirror the closest existing positive-path or negative-path fixture. If there's no positive-path fixture, defer adding this test to Task 3 Step 8 below (after the swap lands) and rely on the unit tests from Task 1/2 as the TDD anchor for this task.

- [ ] **Step 2: Run new test to confirm it fails**

```bash
uv run pytest tests/test_model_edge_cases.py::test_first_crossing_trigger_never_exceeds_end_frame -v
```

Expected: FAIL with `KeyError: 'per_tube_first_crossing'` (details dict doesn't yet contain the new key).

If the test can't be written (no positive-path fixture), skip steps 1–2 and proceed.

- [ ] **Step 3: Update the import in `model.py`**

In `src/bbox_tube_temporal/model.py`, change:

```python
from .inference import (
    crop_tube_patches,
    filter_and_interpolate_tubes,
    pad_frames_symmetrically,
    pad_frames_uniform,
    pick_winner_and_trigger,
    run_yolo_on_frames,
    score_tubes,
)
```

to:

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

- [ ] **Step 4: Replace the call site in `predict()`**

Locate the block (around line 210–218):

```python
        aggregation = dec.get("aggregation", "max_logit")
        is_positive, trigger, winner_id = pick_winner_and_trigger(
            tubes=kept,
            logits=logits,
            threshold=float(dec["threshold"]),
            aggregation=aggregation,
            calibrator=self._calibrator,
            logistic_threshold=float(dec.get("logistic_threshold", 0.5)),
        )
```

Replace with:

```python
        aggregation = dec.get("aggregation", "max_logit")
        is_positive, trigger, winner_id, per_tube_first_crossing = (
            find_first_crossing_trigger(
                classifier=self._classifier,
                tubes=kept,
                patches_per_tube=patches_per_tube,
                masks_per_tube=masks_per_tube,
                full_logits=logits,
                aggregation=aggregation,
                threshold=float(dec["threshold"]),
                calibrator=self._calibrator,
                logistic_threshold=float(dec.get("logistic_threshold", 0.5)),
                min_prefix_length=tubes_cfg["infer_min_tube_length"],
            )
        )
```

- [ ] **Step 5: Add `per_tube_first_crossing` to `details` in the positive-path return**

Locate the `details={...}` dict in the positive-path `TemporalModelOutput(...)` return (around line 258). Add one entry:

```python
            details={
                "num_frames": original_len,
                "num_truncated": n_truncated,
                "num_padded": n_padded,
                "num_detections_per_frame": num_dets_per_frame,
                "num_tubes_total": len(candidate_tubes),
                "num_tubes_kept": len(kept),
                "tube_logits": logits_list,
                "winner_tube_id": winner_id,
                "winner_tube_entries": winner_entries,
                "kept_tubes": kept_tubes,
                "threshold": float(dec["threshold"]),
                "per_tube_first_crossing": per_tube_first_crossing,  # NEW
            },
```

Leave the two early-return branches (empty tubes, no kept tubes) unchanged — their `details` don't include `per_tube_first_crossing` because no qualifying tubes exist. For consistency, optionally add `"per_tube_first_crossing": {}` to those two blocks as well.

- [ ] **Step 6: Run model edge-case tests**

```bash
uv run pytest tests/test_model_edge_cases.py -v
```

Expected: all PASS. The existing `assert out.trigger_frame_index is None` checks at lines 121, 136 remain correct (they're negative-path tests). If any fail, read the failure and reconcile against the new helper's semantics.

- [ ] **Step 7: Run full unit-test suite**

```bash
uv run pytest tests/ -v
```

Expected: everything passes except tests that still import `pick_winner_and_trigger` (those are handled in Task 4).

- [ ] **Step 8: Run the model parity test**

```bash
uv run pytest tests/test_model_parity.py -v
```

Expected: PASS. The parity test asserts equal logits between the offline path and `predict()`, unaffected by the trigger rule change.

- [ ] **Step 9: Lint**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

- [ ] **Step 10: Commit**

```bash
git add src/bbox_tube_temporal/model.py tests/test_model_edge_cases.py
git commit -m "feat(bbox-tube-temporal): use first-crossing trigger in predict()"
```

---

### Task 4: Delete `pick_winner_and_trigger` and its unit tests

With `model.py` fully migrated, the old helper has no callers. Remove it.

**Files:**
- Modify: `src/bbox_tube_temporal/inference.py` (delete function)
- Modify: `tests/test_inference_units.py` (delete `TestPickWinnerAndTrigger`, fix imports)

- [ ] **Step 1: Confirm no other callers remain**

```bash
uv run grep -rn "pick_winner_and_trigger" src/ tests/ scripts/ || true
```

Expected: only matches inside `src/bbox_tube_temporal/inference.py` (the definition) and `tests/test_inference_units.py` (`TestPickWinnerAndTrigger` + the import). If anything else shows up, stop and update those callers first.

- [ ] **Step 2: Remove the import from `tests/test_inference_units.py`**

Delete `pick_winner_and_trigger` from the import block:

```python
from bbox_tube_temporal.inference import (
    crop_tube_patches,
    filter_and_interpolate_tubes,
    pick_winner_and_trigger,   # <-- delete this line
    run_yolo_on_frames,
    score_tubes,
)
```

- [ ] **Step 3: Delete the `TestPickWinnerAndTrigger` class**

Remove the entire class (currently at lines ~283–392) from `tests/test_inference_units.py`.

- [ ] **Step 4: Delete `pick_winner_and_trigger` from `inference.py`**

Remove the entire function definition (currently at lines 242–310) from `src/bbox_tube_temporal/inference.py`.

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS. If any fail with `ImportError` or `NameError` referencing `pick_winner_and_trigger`, there's a leftover caller — find it and fix.

- [ ] **Step 6: Lint**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

- [ ] **Step 7: Commit**

```bash
git add src/bbox_tube_temporal/inference.py tests/test_inference_units.py
git commit -m "refactor(bbox-tube-temporal): remove obsolete pick_winner_and_trigger"
```

---

### Task 5: Rebuild packaged models + capture validation numbers

Not a code change, but the spec requires before/after TTD numbers in the PR description. This task runs the DVC pipeline and records the outcome.

**Files:** none committed. Numbers go into the PR description.

- [ ] **Step 1: Capture current (pre-change-in-output) TTD numbers for both variants**

If an un-rebuilt `evaluate_packaged` metrics file still reflects the old trigger (check `git status` and the `dvc.lock` hash for the `evaluate_packaged_*` stages), read them first:

```bash
uv run cat data/08_reporting/val/packaged/gru_convnext_finetune/metrics.json
uv run cat data/08_reporting/val/packaged/vit_dinov2_finetune/metrics.json
uv run cat data/08_reporting/train/packaged/gru_convnext_finetune/metrics.json
uv run cat data/08_reporting/train/packaged/vit_dinov2_finetune/metrics.json
```

Record `mean_ttd_seconds`, `median_ttd_seconds`, `precision`, `recall`, `f1` from each. These are the "before" numbers.

If the reports on disk were already regenerated mid-development (e.g., `metrics.json` is dirty in `git status`), find the pre-change numbers in git history on the branch point:

```bash
git log --oneline -- data/08_reporting/val/packaged/gru_convnext_finetune/metrics.json
git show <last-commit-before-branch>:experiments/temporal-models/bbox-tube-temporal/data/08_reporting/val/packaged/gru_convnext_finetune/metrics.json
```

- [ ] **Step 2: Rebuild `model.zip` and re-run evaluation for both variants**

```bash
uv run dvc repro package_gru_convnext_finetune evaluate_packaged_gru_convnext_finetune
uv run dvc repro package_vit_dinov2_finetune evaluate_packaged_vit_dinov2_finetune
```

Expected: stages run green. `dvc.lock` updates. `data/06_models/<variant>/model.zip` and `data/08_reporting/{train,val}/packaged/<variant>/metrics.json` get new contents.

(Exact stage names depend on `dvc.yaml`. If the above don't exist, run `uv run dvc stage list` and adjust.)

- [ ] **Step 3: Capture post-change TTD numbers**

```bash
uv run cat data/08_reporting/val/packaged/gru_convnext_finetune/metrics.json
uv run cat data/08_reporting/val/packaged/vit_dinov2_finetune/metrics.json
uv run cat data/08_reporting/train/packaged/gru_convnext_finetune/metrics.json
uv run cat data/08_reporting/train/packaged/vit_dinov2_finetune/metrics.json
```

Record the new `mean_ttd_seconds`, `median_ttd_seconds`, `precision`, `recall`, `f1`.

- [ ] **Step 4: Sanity-check the numbers**

Expected:

- `mean_ttd_seconds` dropped substantially vs. pre-change (spec target: from ~515s to something much smaller).
- `precision`, `recall`, `f1` unchanged vs. pre-change (D2 guard guarantees `is_positive` is bit-identical; any drift means a bug).
- `median_ttd_seconds` also dropped (or at minimum did not increase).

If precision/recall moved: STOP. D2 guard is broken somewhere. Debug before continuing.

- [ ] **Step 5: Commit the regenerated artefacts**

```bash
git add dvc.lock data/08_reporting/
git commit -m "chore(bbox-tube-temporal): rebuild packaged models with first-crossing trigger"
```

- [ ] **Step 6: Draft the PR body**

Assemble the before/after table for the PR description (do NOT commit this — it's for the pull-request text only):

```markdown
## TTD before / after

| variant                  | split | mean_ttd (s) before | after | median_ttd (s) before | after | precision before/after | recall before/after |
|--------------------------|-------|---------------------|-------|-----------------------|-------|------------------------|---------------------|
| gru_convnext_finetune    | val   | <X>                 | <Y>   | <X>                   | <Y>   | <P>/<P>                | <R>/<R>             |
| gru_convnext_finetune    | train | …                   | …     | …                     | …     | …                      | …                   |
| vit_dinov2_finetune      | val   | …                   | …     | …                     | …     | …                      | …                   |
| vit_dinov2_finetune      | train | …                   | …     | …                     | …     | …                      | …                   |

Precision/recall/F1 are expected bit-identical by design (D2 guard).
```

---

## Self-review checklist

(Not a runtime step — ran during plan authoring.)

**Spec coverage:**
- Decision 1 (prefix scoring) → Task 1 Step 3 implementation.
- Decision 2 (earliest-crossing across qualifying tubes, tie-break tube_id) → Task 1 Step 1 tests `test_earliest_crossing_wins_across_tubes`, `test_tie_on_crossing_frame_breaks_on_smallest_tube_id`.
- Decision 3 (D2 guard) → Task 1 test `test_d2_guard_ignores_non_qualifying_tube`; Task 2 tests cover logistic D2.
- Decision 4 (serial prefix scoring, early exit) → Task 1 test `test_loop_walks_prefix_lengths_until_crossing`.
- Decision 5 (min_prefix_length = infer_min_tube_length) → Task 3 Step 4 passes `tubes_cfg["infer_min_tube_length"]`.
- Decision 6 (no config/version changes) → no edits to `config.yaml`, `manifest.yaml`, `params.yaml` anywhere in the plan.
- Algorithm → Task 1 Step 3 implementation matches pseudocode in spec.
- `details` addition (`per_tube_first_crossing`) → Task 3 Step 5.
- Edge cases (empty tubes, no qualifying, tie on frame, min-length prefix) → covered by Task 1 tests.
- Latency follow-up (Approach 2 if p95 regresses) → explicitly out of scope; no task.
- Validation (before/after TTD numbers) → Task 5.

**Placeholder scan:** No "TODO", "TBD", or "similar to above" without full code. All steps have commands + expected output. 

**Type consistency:** `find_first_crossing_trigger` signature identical across Task 1 implementation, Task 2 edit, Task 3 call site, and tests. Return tuple always 4 elements `(bool, int|None, int|None, dict)`. `min_prefix_length` keyword-only everywhere.
