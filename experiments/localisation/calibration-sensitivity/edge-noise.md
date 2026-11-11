# Skyline edge bias and spatially correlated noise (#163)

## Decision

Measure **systematic edge bias and the spatial correlation of edge errors**
before choosing a larger segmentation model. Independent one-pixel column
errors averaged down to about 0.003° tilt spread in these controls; errors
shared across 128-column blocks produced about 0.015°. A shared one-pixel
offset shifted tilt by 0.04998° and did not average down.

This completes an offline control for the first step of
[#163](https://github.com/pyronear/vision-rd/issues/163). It does not measure
learned-mask error or replace the planned hand-drawn skyline evaluation.

## Reproduce

```bash
cd experiments/localisation/calibration-sensitivity
uv sync --frozen --python 3.11
uv run --frozen --no-sync python edge_noise.py --params edge_noise.yaml
```

The run generates all inputs in memory. No images, credentials, camera,
geographic terrain downloads, model inference or DVC pull is needed.
[Configuration](edge_noise.yaml) · [Aggregate results](edge-noise.csv) ·
[Run provenance](edge-noise-provenance.json).

Full signed trials and provenance are written under
`data/08_reporting/edge-noise-offline/`. The 72 aggregate rows summarize
1,464 trials. Two complete runs produced byte-identical aggregate CSVs.

## Method

- Render 1280×720 masks with a synthetic 60° horizontal field of view and
  2° tilt. Test roll 0° and 1° against a flat skyline and an analytic skyline
  containing several ridges. The ridge coefficients are in the YAML.
- Supply the analytic elevation profile in place of `dem_skyline` while
  retaining the pinned runtime's `calibrate()` fitter, column sampling and
  acceptance gates. The small constant DSM array is only an API scaffold;
  no geographic terrain is sampled. The sampler is restored after each run.
- Compare a fixed true azimuth with a joint fit searching ±2° around it.
  Flat skylines cannot determine azimuth. Known optics and a correct initial
  bearing isolate edge error from other calibration defects.
- Include a clean control and common shifts of ±1 image pixel for each
  profile, roll and fitting mode.
- Draw zero-mean Gaussian row errors with standard deviations of 1 and
  2 pixels, round to integer rows, and share each draw across blocks of
  1, 16 or 128 adjacent columns. Run seeds 0–29 for every setting. Realised
  means are not forced to zero, and the same error vectors are reused across
  profiles, rolls and fitting modes.
- Record signed error against known tilt and change relative to the matching
  clean fit. Counts against the 0.009°, 0.03° and 0.1° comparison targets use
  **absolute truth**, not the clean-subtracted change. Rejections remain in
  the trial denominator. Spread is the empirical standard deviation across
  seeds, not a confidence bound.

## Results

For one-pixel noise, flat skyline, zero roll and fixed true azimuth:

| Columns sharing each error | Tilt spread (°) | Worst absolute tilt error (°) | Within 0.009° |
|---:|---:|---:|---:|
| 1 | 0.00309 | 0.00859 | 30/30 |
| 16 | 0.00572 | 0.01177 | 27/30 |
| 128 | 0.01476 | 0.04131 | 14/30 |

Across both profiles, rolls and fitting modes, one-pixel independent noise
gave 0.00304–0.00318° spread; 128-column blocks gave 0.01472–0.01571°.
All 240 independent-noise trials met the 0.009° comparison target, versus
108/240 for 128-column blocks. These are counts in the specified controlled
cases, not estimated field success probabilities.

All 1,464 fits passed the runtime gates. Some still exceeded the comparison
targets: acceptance continues to be distinct from tilt accuracy. Clean control
errors were below 0.001° here; this simple analytic experiment does not
reproduce or supersede the DSM-based floor in [#161](skyline-floor.md).

For these known-FOV controls, one pixel corresponds to 0.04998°.
The 0.009° plains comparison therefore allows only about **0.18 image pixel
of shared bias if that term uses the entire budget**. Other error sources
need room in the same budget. Reducing independent scatter while retaining
a common edge offset will not solve that problem.

## Practical next step and limits

The full-resolution skyline references in #163 should retain per-column
signed residuals. Report each image's mean/median edge bias separately from
within-image spread, and measure whether residuals remain correlated across
neighbouring columns. Evaluate proposed refinements by their effect on tilt,
not just aggregate pixel overlap.

These controls use two idealised profiles, blockwise Gaussian errors and
30 seeds. There are no clouds, occluders, segmentation failures, real optics,
FOV errors, refraction or terrain errors. The renderer and fitter share the
runtime's angular camera model. The runtime also samples at most 384 columns,
so a wider mask does not supply one independent fit observation per column.
Field precision and the actual correlation length of mask errors remain
unmeasured; no site-level localisation guarantee follows from these results.
