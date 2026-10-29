# Skyline tilt repeatability

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
`uv run dvc pull` requires S3 access and the archive in the configured remote;
the supplied local ZIP is sufficient for this study.

## Method

- Keep declared FOV fixed. Use known mast heights for two cameras, otherwise
  assume 20 m. The platform elevation field is not mast height.
- Fit every frame with a 25 km skyline search and ±20° azimuth search; retry
  ±45° at a boundary optimum.
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
  Report the last passing and first failing grid samples, not a continuous
  tolerance. `cap_reached` means no failing sample was tested. Missing or
  unreached targets have no bounds. These limits apply only to this view.

## Results and limits

**FOV (field of view) is the largest tested systematic sensitivity.** Across
offsets up to ±15°, the largest shift per pose is 0.426–4.431° on 13 complete
scenarios, each using the first supported frame. An incorrect
FOV can bias every frame in a pose. This measures sensitivity, not the actual
FOV error or tilt bias. Scenario amplitudes differ; they are not an error budget.

[repeatability.csv](repeatability.csv) contains anonymous pose summaries:

- **Fits:** 362/631 pass. Of 269 rejections, 177 fail the residual/roll gate,
  91 lack enough skyline columns, and one hits the azimuth search boundary.
  Most rejects therefore pass the mask-support gate but fail the terrain fit.
- **Repeatability:** 15 poses support a spread estimate. Robust spread is
  0.053–1.320°; empirical P95 deviation is 0.314–9.407°. In five 2/5 km comparisons,
  P95 exceeds the first failing tilt sample. Six reach the grid cap and are unresolved.
  For context, a flat 40 m mast at 5 km has a continuous tolerance of about
  0.0086° for a 100 m budget; the grid brackets it between 0.003° and 0.01°.
- **Azimuth:** 56 accepted frames differ from their pose tilt median by >1°.
  Of these, 38 also differ in azimuth by >0.5° (all such outliers on P02, P04,
  P09 and P14). P01/P13 need separate mask/fit inspection. This association
  does not label a fit as wrong; real camera moves can change both angles.
- **Radius coverage:** 243/362 fits (67.1%) lie within their returned tilt
  radius of the pose median. This is an in-sample diagnostic, not a calibrated
  1-sigma bound or absolute accuracy. Pose medians can also mix camera states.
- **Horizon distance:** extending 25 km to 50 km shifts tilt by 1.178° on P02,
  1.126° on P05, 1.051° on P07 and 0.873° on P11. A fixed 25 km search is too
  restrictive for these fits; inspect the wider skyline before accepting them.
- **Refraction:** changing the coefficient from 0.13 to 0 or 0.2 gives largest
  shifts of 0.0033–0.0143° across complete poses, comparable to the plains
  budget. These scenarios change the skyline fit, not the localisation target.

These results describe this implementation with a 384 px mask, a 25 km search
and platform FOV values. They do not establish the limit of skyline calibration.
Absolute bias remains unmeasured; independent validation belongs in [#120](https://github.com/pyronear/vision-rd/issues/120).

## Next steps

1. Measure FOV using the 360° sweep ([pyro-engine#425](https://github.com/pyronear/pyro-engine/issues/425)),
   then rerun these poses. Let the runtime skyline search follow the DSM extent.
2. Test an azimuth-consistency gate within verified stable camera states. Keep
   rejected candidates visible; a whole-pose gate could hide real camera moves.
3. Use [#160](https://github.com/pyronear/vision-rd/issues/160) to isolate synthetic
   defects and measure the method's floor before choosing another calibration method.

Private `frames.jsonl` and `profiles.json` retain every fit and scenario;
`epochs.csv` records candidate states. Masks and complete outputs stay outside Git.
The revised CSV re-summarizes the saved 631 fits. With the updated runtime and
NumPy 2.4.6, 21 matched checks cover every pose and rejection reason; fit outputs
and one fresh sky mask match exactly. This is not a full new image run.
No site acceptance or triangulation filter follows from these samples.
