# Calibration requirements and repeatability

## Objective

For [#103](https://github.com/pyronear/vision-rd/issues/103), identify which
camera views need tighter calibration. Compare a difficult flat case with real
relief. Results describe sampled views, not site-wide deployment acceptance.

## Approach

For achieved tilt precision, run `uv run repeatability.py --input /path/to/bundle`.
It retains rejected frames and candidate camera changes for
[#105](https://github.com/pyronear/vision-rd/issues/105).
[Method and reproduction](repeatability.md) · [Pose results](repeatability.csv).

`study.py` provides the controlled, one-parameter-at-a-time baseline. It covers
azimuth, tilt, roll, field of view, height and both horizontal position axes,
at several ranges, heights and image positions. Its original summary CSVs
remain in [position-errors.csv](position-errors.csv) and
[precision-targets.csv](precision-targets.csv).

`terrain.py` reuses the Copernicus ray-caster from
[smoke-localization](https://github.com/pyronear/smoke-localization), pinned in
`pyproject.toml`.
It includes Earth curvature and fixed standard refraction (`k = 0.13`).
For each mast it aims at surface targets on eight compass bearings at 2, 5 and
10 km. It repeats the test on a flat surface at the same mast elevation.
Both surfaces use the same ray-caster, 10 m steps and a 15 km ray limit.
Targets missed by more than one DSM cell (30 m) are retained as
`occluded_or_unresolved`; they are not assigned an accuracy score.

For visible targets, change each calibration parameter separately in both
directions. The pinhole model is shared with `study.py`: assumed 54.2° horizontal
field of view, 16:9 sensor and pixels `(0.5,0.5)`, `(0.1,0.1)`, `(0.9,0.9)`.
Angular test amplitudes are 0.001/0.003/0.01/0.03/0.1°; height and position
amplitudes are 0.1/0.3/1/3/10 m. Horizontal position changes keep absolute camera
altitude fixed, so they do not add height error.

Precision uses a **100 m example budget**. `tested_limit` is the last amplitude
in the initial passing sequence, with both signs and all three pixels passing.
`below_grid` means even the smallest test failed; `all_tested_pass` means the
largest test also passed. These sampled limits do not guarantee intermediate
inputs or combined errors. A missing projection is a failure, not zero error.

## Data

[sites.csv](sites.csv) contains three self-supporting pylons from the
[ANFR September 2026 snapshot](https://static.data.gouv.fr/resources/donnees-sur-les-installations-radioelectriques-de-plus-de-5-watts-1/20261006-082445/20260930-export-etalab-data.zip),
table `SUP_SUPPORT.txt` (Open Licence 2.0). IDs, positions and support heights
are recorded inputs. The foothill and mountain sites are near the reviewer's
Alsace/Vosges map area; the flat comparison uses a 40 m Landes mast.

Copernicus GLO-30 is a **digital surface model (DSM)**: it includes canopy and
buildings. Camera elevation is assumed to be local DSM elevation plus the ANFR
mast height. This is not a surveyed camera mounting position or bare-ground
elevation. Camera optics and calibration errors are also assumed.

## Results

For ±0.1° tilt, the 40 m flat control shifts up to **1,502 m at 5 km**;
the Landes DSM case reaches **1,949 m**, with 6/8 nominal targets visible.
At 2 km, the Vosges case shifts up to **30 m** against **158 m** on its flat
control, but only 4/8 targets are visible. No sampled Vosges target is reached
at 5 or 10 km. Relief can reduce sensitivity while limiting coverage.

[terrain-results.csv](terrain-results.csv) records visibility, tilt sensitivity
and sampled tilt limits for both surfaces. All attempted target directions stay
in the visibility denominator; errors and precision limits cover visible targets
only. `tilt_all_tested_within_budget` counts targets whose tilt grid passes at
all three image positions. Maximum valid errors must be read with missing-hit
counts. Blank results mean no visible target, not zero error.

The full run writes per-view status, signed errors, per-parameter precision and
a summary. `config.json` records source/data hashes, the prototype revision and
installed library versions. DSM tiles and full outputs stay outside Git.

## How to reproduce

```bash
cd experiments/localisation/calibration-sensitivity
uv sync --python 3.11
uv run pytest tests/ -v
uv run ruff check .
uv run ruff format --check .
uv run terrain.py
# Compare data/08_reporting/terrain/summary.csv with terrain-results.csv.
uv run study.py
# Explicit difficult flat case, without curvature or refraction:
uv run study.py --heights 40 --distances 5000 --angular-errors 0.1
```

`terrain.py --sites other-sites.csv --budget 250` accepts the same input columns.
`--cache` and `--output` can point outside the checkout. Tests use synthetic
surfaces and require no DSM download. The experiment follows the template's
`pyproject.toml`, pytest and Ruff checks and is found by the existing CI matrix.
The prototype and geometry libraries are pinned; resolved versions are recorded.

## Limits

Real relief can shorten the ray intersection and reduce sensitivity, or hide
the target entirely. Neither a flat-plane score nor a sampled terrain result
establishes whole-site feasibility. Check camera mounting height, terrain and
known ground points before using these as calibration requirements.
No smoke image, ground-origin detection, combined uncertainty, lens distortion
or weather-dependent refraction is evaluated. Full geometry and uncertainty
remain in [#109](https://github.com/pyronear/vision-rd/issues/109) and
[#113](https://github.com/pyronear/vision-rd/issues/113).
