# Achieved skyline tilt precision

For [#105](https://github.com/pyronear/vision-rd/issues/105), measure repeated
`calibrate()` outputs and compare them with the precision needed for localisation.
The runtime change is [smoke-localization #8](https://github.com/pyronear/smoke-localization/pull/8).

## Input and run

The privately supplied bundle has 631 frames, 18 poses and 9 camera IDs. The
[DVC pointer](data/01_raw/skyline-tilt-images.zip.dvc) records the archive checksum.
Images and camera coordinates stay outside Git; `P01`–`P18` follow `cameras.csv`
row order. These are convenience samples of false-positive detections, not a
random weather survey.

```bash
uv sync --locked
uv run repeatability.py --input /path/to/skyline-tilt-images \
  --output data/08_reporting/repeatability
```

The script caches model masks and public DSM tiles. `--cache` and `--output`
accept external directories; `--summarize` reuses recorded fits. Source/input
hashes and library versions prevent mixing incompatible runs.

## Method

- Keep declared FOV fixed. Use known mast heights for two cameras, otherwise
  assume 20 m. The platform elevation field is not mast height.
- Fit every frame with ±20° azimuth search; retry ±45° at a boundary optimum.
  Keep rejections and reasons. These are model gates, not labelled weather tests.
- On the first supported frame, test FOV, height, mask threshold, refraction and
  terrain extent using [repeatability.yaml](repeatability.yaml). Failed or
  incomplete scenarios have blank sensitivity, not zero.
- Flag candidate state changes from persistent median shifts. Require five
  accepted frames for robust spread (`1.4826 × MAD`) and empirical P95 deviation.
  Short states stay visible without a precision estimate. Changes are candidates,
  not confirmed hardware movement; unresolved changes may remain.
- Compare with a 100 m budget at 2/5 km on the centre bearing. Other inputs stay
  fixed. The range check uses 10 m steps, a 15 km cap and 30 m nominal tolerance.
  Status columns distinguish untested, unreached and below-grid targets.

## Results and limits

[repeatability.csv](repeatability.csv) contains anonymous pose summaries.
362/631 frames pass the gates; 15 poses support a spread estimate. Robust spread
is 0.053–1.320° and empirical P95 deviation is 0.314–9.407°. None of the available
P95 comparisons meets the sampled 2/5 km requirement.
Private `frames.jsonl` and `profiles.json` retain every fit and scenario;
`epochs.csv` records candidate states. Masks and complete outputs stay outside Git.

The returned radius is a conditional fit diagnostic without an assigned
confidence level. Native mask pixels span about 0.084°/0.146° at 54.2°/87° FOV;
upsampling does not add observations. This is not a proof against subpixel fitting.
Absolute bias is unmeasured and needs a verified landmark or Sun reference (#120).
No whole-site acceptance or triangulation filter follows from these samples.

A motor endstop can test return repeatability. Absolute zero also requires its
physical angle and the camera mounting alignment to be known.
