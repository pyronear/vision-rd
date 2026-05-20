# Temporal Model Serving — Phase 1: Extract Inference Core to `lib/` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new, independently-tested `lib/bbox-tube-temporal/` package that is a zero-edit **mirror copy** of the bbox-tube-temporal model's inference-core modules, so the future serving service can depend on it without touching the experiment.

**Architecture:** Copy 10 source modules + 11 self-contained test files + one synthetic fixture from the experiment into a new uv project. The experiment is **not modified**. The copy keeps the same import name (`bbox_tube_temporal`) and file layout so it stays `diff`-able for drift; the distribution name is `bbox-tube-temporal-core` to avoid any clash. Verification is the copied test suite passing.

**Tech Stack:** Python 3.11, uv, hatchling, pytest, ruff, torch (CPU ok), timm, ultralytics, pydantic, pyrocore (path dep).

**Spec:** `docs/specs/2026-05-20-temporal-model-serving-design.md` (sections "Repo layout", "Lib dependency set", "Extraction approach").

> **Note on TDD framing:** This phase copies already-tested code rather than writing new features, so the usual "write a failing test first" loop is replaced by "copy code + copy its existing tests + run them green." The copied experiment tests are the executable spec for the copy.

---

## File structure (what this phase creates)

```
lib/bbox-tube-temporal/
  pyproject.toml                       # distribution bbox-tube-temporal-core; runtime deps; ruff config
  Makefile                             # install / lint / format / test
  README.md                            # one-paragraph "this is a mirror copy" note
  src/bbox_tube_temporal/
    __init__.py                        # empty (mirrors experiment)
    model.py inference.py package.py tubes.py model_input.py
    types.py logistic_calibrator.py details_schema.py temporal_classifier.py data.py
  tests/
    __init__.py
    fixtures/parity/wildfire/seq_synth01/...   # 64K synthetic parity fixture
    test_model_parity.py test_model_edge_cases.py test_inference_units.py
    test_package.py test_tubes.py test_model_input.py test_types.py
    test_logistic_calibrator.py test_details_schema.py test_temporal_classifier.py
    test_padding.py
```

Source of all copies (the **experiment**, never modified):
`experiments/temporal-models/bbox-tube-temporal/` — referred to below as `$EXP`.

The 10 modules form a closed import graph (verified): every `from .X` among them
targets another module in this set. `data.py` is included because
`model_input.py` does `from .data import find_sequence_dir` at module scope; its
only intra-package import is `from .types import ...`.

---

### Task 1: Scaffold the `lib/bbox-tube-temporal/` uv project

**Files:**
- Create: `lib/bbox-tube-temporal/pyproject.toml`
- Create: `lib/bbox-tube-temporal/Makefile`
- Create: `lib/bbox-tube-temporal/README.md`
- Create: `lib/bbox-tube-temporal/src/bbox_tube_temporal/__init__.py` (empty)
- Create: `lib/bbox-tube-temporal/tests/__init__.py` (empty)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/bbox_tube_temporal"]

[project]
name = "bbox-tube-temporal-core"
version = "0.1.0"
description = "Inference core for the bbox-tube temporal smoke classifier (mirror copy of the experiment's serving-path modules)"
requires-python = ">=3.11"
dependencies = [
    "pyrocore",
    "numpy>=1.26,<2",
    "pillow>=10.0",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "timm>=1.0",
    "torch>=2.2",
    "torchvision>=0.17",
    "ultralytics>=8.3",
]

[tool.uv.sources]
pyrocore = { path = "../pyrocore" }

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.9",
]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B", "SIM", "PLC0415"]

[tool.ruff.format]
quote-style = "double"
```

- [ ] **Step 2: Create `Makefile`**

```makefile
.PHONY: install lint format test help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

lint: ## Run ruff linter
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

test: ## Run tests with pytest
	uv run pytest tests/ -v
```

- [ ] **Step 3: Create `README.md`**

```markdown
# bbox-tube-temporal-core

Inference core for the bbox-tube temporal smoke classifier — a **mirror copy** of
the serving-path modules from
`experiments/temporal-models/bbox-tube-temporal/src/bbox_tube_temporal/`.

The production temporal-model API (`services/temporal-model-api/`) depends on this
package so it never imports the experiment. The copy keeps the same import name
and file layout as the experiment so the two can be `diff`-ed to detect drift.
Deduplication (the experiment depending on this lib) is deferred — see
`docs/specs/2026-05-20-temporal-model-serving-design.md` (Future work).
```

- [ ] **Step 4: Create the two empty `__init__.py` files**

```bash
mkdir -p lib/bbox-tube-temporal/src/bbox_tube_temporal lib/bbox-tube-temporal/tests
: > lib/bbox-tube-temporal/src/bbox_tube_temporal/__init__.py
: > lib/bbox-tube-temporal/tests/__init__.py
```

- [ ] **Step 5: Verify the project installs (no modules yet)**

Run (from `lib/bbox-tube-temporal/`): `uv sync`
Expected: resolves and installs `bbox-tube-temporal-core` plus deps (torch, timm,
ultralytics, pyrocore from the local path) with no error. A `.venv/` and
`uv.lock` are created.

- [ ] **Step 6: Commit**

```bash
git add lib/bbox-tube-temporal/pyproject.toml lib/bbox-tube-temporal/Makefile lib/bbox-tube-temporal/README.md lib/bbox-tube-temporal/src/bbox_tube_temporal/__init__.py lib/bbox-tube-temporal/tests/__init__.py lib/bbox-tube-temporal/uv.lock
git commit -m "feat(lib): scaffold bbox-tube-temporal-core package"
```

---

### Task 2: Copy the 10 inference-core modules

**Files:**
- Create (copied): `lib/bbox-tube-temporal/src/bbox_tube_temporal/{model,inference,package,tubes,model_input,types,logistic_calibrator,details_schema,temporal_classifier,data}.py`

- [ ] **Step 1: Copy the 10 modules verbatim from the experiment**

```bash
EXP=experiments/temporal-models/bbox-tube-temporal/src/bbox_tube_temporal
LIB=lib/bbox-tube-temporal/src/bbox_tube_temporal
for m in model inference package tubes model_input types logistic_calibrator details_schema temporal_classifier data; do
  cp "$EXP/$m.py" "$LIB/$m.py"
done
```

- [ ] **Step 2: Verify the copies are byte-identical to the source (drift = 0)**

```bash
EXP=experiments/temporal-models/bbox-tube-temporal/src/bbox_tube_temporal
LIB=lib/bbox-tube-temporal/src/bbox_tube_temporal
for m in model inference package tubes model_input types logistic_calibrator details_schema temporal_classifier data; do
  diff "$EXP/$m.py" "$LIB/$m.py" && echo "OK $m"
done
```
Expected: `OK <module>` for all 10, no diff output.

- [ ] **Step 3: Verify the package imports cleanly**

Run (from `lib/bbox-tube-temporal/`):
`uv run python -c "import bbox_tube_temporal.model, bbox_tube_temporal.inference, bbox_tube_temporal.package, bbox_tube_temporal.model_input, bbox_tube_temporal.data; print('import ok')"`
Expected: `import ok` (this exercises the `model_input -> data` and
`model -> package -> inference -> ...` import chains).

- [ ] **Step 4: Verify lint passes (the one sanctioned `# noqa: PLC0415` in `package.py` is copied as-is)**

Run (from `lib/bbox-tube-temporal/`): `make lint`
Expected: ruff reports no errors.

- [ ] **Step 5: Commit**

```bash
git add lib/bbox-tube-temporal/src/bbox_tube_temporal/model.py lib/bbox-tube-temporal/src/bbox_tube_temporal/inference.py lib/bbox-tube-temporal/src/bbox_tube_temporal/package.py lib/bbox-tube-temporal/src/bbox_tube_temporal/tubes.py lib/bbox-tube-temporal/src/bbox_tube_temporal/model_input.py lib/bbox-tube-temporal/src/bbox_tube_temporal/types.py lib/bbox-tube-temporal/src/bbox_tube_temporal/logistic_calibrator.py lib/bbox-tube-temporal/src/bbox_tube_temporal/details_schema.py lib/bbox-tube-temporal/src/bbox_tube_temporal/temporal_classifier.py lib/bbox-tube-temporal/src/bbox_tube_temporal/data.py
git commit -m "feat(lib): copy bbox-tube inference core (10 modules, mirror)"
```

---

### Task 3: Copy the core tests + synthetic fixture and verify green

**Files:**
- Create (copied): `lib/bbox-tube-temporal/tests/{test_model_parity,test_model_edge_cases,test_inference_units,test_package,test_tubes,test_model_input,test_types,test_logistic_calibrator,test_details_schema,test_temporal_classifier,test_padding}.py`
- Create (copied): `lib/bbox-tube-temporal/tests/fixtures/parity/` (synthetic)

- [ ] **Step 1: Copy the 11 self-contained core test files**

```bash
EXP=experiments/temporal-models/bbox-tube-temporal/tests
LIB=lib/bbox-tube-temporal/tests
for t in test_model_parity test_model_edge_cases test_inference_units test_package test_tubes test_model_input test_types test_logistic_calibrator test_details_schema test_temporal_classifier test_padding; do
  cp "$EXP/$t.py" "$LIB/$t.py"
done
```

- [ ] **Step 2: Copy the synthetic parity fixture (used by `test_model_parity`)**

```bash
cp -r experiments/temporal-models/bbox-tube-temporal/tests/fixtures/parity \
      lib/bbox-tube-temporal/tests/fixtures/parity
```

- [ ] **Step 3: Run the full copied test suite**

Run (from `lib/bbox-tube-temporal/`): `make test`
Expected: all tests in the 11 files PASS (CUDA-only tests in
`test_model_edge_cases.py` are auto-skipped via their `@pytest.mark.skipif`).
No collection errors, no missing-fixture errors.

> If any test fails ONLY because it imports a name that lives in an un-copied
> training module, stop — that means the closure analysis missed a dependency.
> Re-run the closure check from the spec and copy the missing module before
> continuing. (Per the verified analysis this should not happen.)

- [ ] **Step 4: Commit**

```bash
git add lib/bbox-tube-temporal/tests/test_model_parity.py lib/bbox-tube-temporal/tests/test_model_edge_cases.py lib/bbox-tube-temporal/tests/test_inference_units.py lib/bbox-tube-temporal/tests/test_package.py lib/bbox-tube-temporal/tests/test_tubes.py lib/bbox-tube-temporal/tests/test_model_input.py lib/bbox-tube-temporal/tests/test_types.py lib/bbox-tube-temporal/tests/test_logistic_calibrator.py lib/bbox-tube-temporal/tests/test_details_schema.py lib/bbox-tube-temporal/tests/test_temporal_classifier.py lib/bbox-tube-temporal/tests/test_padding.py
git add lib/bbox-tube-temporal/tests/fixtures/parity
git commit -m "test(lib): copy core tests + parity fixture; suite green"
```

---

### Task 4: Confirm CI discovery and final drift check

**Files:** none (verification + documentation only)

- [ ] **Step 1: Confirm the lib is discovered by the Lib CI pattern**

The Lib CI workflow auto-discovers any `lib/**` directory containing a
`pyproject.toml` and runs `ruff check`, `ruff format --check`, `pytest tests/ -v`.
Run all three locally to mirror CI (from `lib/bbox-tube-temporal/`):

Run: `uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -v`
Expected: ruff check clean, format check clean, all tests pass.

> If `ruff format --check` reports files needing formatting, the copied files
> already match the experiment's style (same ruff config), so this should pass.
> If it does not, run `make format` and re-commit — do NOT hand-edit.

- [ ] **Step 2: Record the drift baseline in the README**

Append to `lib/bbox-tube-temporal/README.md`:

```markdown

## Drift check

The 10 copied modules are byte-identical to the experiment as of this commit.
To detect drift later:

\`\`\`bash
EXP=experiments/temporal-models/bbox-tube-temporal/src/bbox_tube_temporal
LIB=lib/bbox-tube-temporal/src/bbox_tube_temporal
for m in model inference package tubes model_input types logistic_calibrator details_schema temporal_classifier data; do
  diff "$EXP/$m.py" "$LIB/$m.py" && echo "OK $m"
done
\`\`\`
```

- [ ] **Step 3: Commit**

```bash
git add lib/bbox-tube-temporal/README.md
git commit -m "docs(lib): record drift-check baseline for the mirror copy"
```

---

## Self-Review

**1. Spec coverage (Phase-1 scope):**
- "Repo layout → `lib/bbox-tube-temporal/`" → Tasks 1–3. ✓
- "Lib dependency set (lean prod runtime)" → Task 1 Step 1 `pyproject.toml` deps. ✓
- "Extraction approach — copy, experiment untouched" → all copies are one-way `cp`; no task edits the experiment. ✓
- "mirror copy, diff-able, distribution `bbox-tube-temporal-core`" → Task 1 (name), Task 2 Step 2 + Task 4 Step 2 (diff). ✓
- "ships copies of the tests that exercise these modules" → Task 3. ✓
- Out of Phase-1 scope (later plans): the LitServe service, Dockerfile, tofu, the real-sequence integration test, dedup.

**2. Placeholder scan:** No "TBD/TODO/handle edge cases" steps; every step has the exact command or file content. ✓

**3. Type consistency:** No new types are defined — modules are copied verbatim, so signatures match the experiment by construction. Module list (10) and test list (11) are identical everywhere they appear (Tasks 2, 3, 4 and the file-structure section). ✓

**4. Closure correctness:** The included-module set is closed under `from .` imports (verified), with `data.py` added for `model_input`'s `find_sequence_dir`. Task 3 Step 3 has an explicit guard if the closure is wrong.
