# Calibration sensitivity

Study for [#103](https://github.com/pyronear/vision-rd/issues/103): position errors
and conditional calibration targets for a single camera. Standard library only.

## Run

```bash
cd experiments/localisation/calibration-sensitivity
uv run --python 3.11 study.py --check
uv run --python 3.11 study.py
ruff check --select E,F,I,W,UP,B,SIM --target-version py311 study.py
ruff format --check study.py
```

Full sweep files stay outside Git: `errors.csv` records signed perturbations and
no-hit cases; `precision.csv` records two-sided limits; `config.json` records
arguments, Python version and source hash. `capped=True` means a lower bound.

## Method

A perspective camera uses azimuth (direction), tilt (up/down angle), roll
(image rotation) and horizontal field of view. Azimuth and roll are clockwise;
positive tilt points down. Sensor aspect ratio is 16:9. FOV errors change the
assumed focal length for both image axes. Each known target is
placed at nine image locations (`u,v = 0.1,0.5,0.9`). Hold that pixel fixed and
change one parameter. Intersect the resulting ray with each linear terrain
segment; select the first forward hit. Horizontal position errors leave camera
altitude unchanged, separating them from height errors.

Defaults: 0.5/1/2/5/10 km; camera heights 15/35/100 m; FOVs 54.2/87 degrees;
angular errors ±0.01/0.1/1 degree; height/position errors ±0.1/1/5 m. Terrain is
flat, or rises at 2% or 10% after half the target distance. Targets and terrain are
synthetic; heights and FOVs include the prototype's
[camera registry](https://github.com/pyronear/smoke-localization/blob/be051f802809b8186cf65a058a174e5d23f4b486/data/cameras.csv).
Use `--help` for options, for example `--budget 50 --heights 35 --distances 1000 5000`.

## Results

Default run: **34,020 perturbations**, **356 no-hit cases**, **5,670 precision
rows**. All nominal targets are visible. Analytical checks cover perspective,
rotations, flat/sloped intersections, height, a nearer ridge and precision limits.

![Maximum position error across distances and camera heights for specified calibration errors on flat terrain](https://github.com/user-attachments/assets/6b5f48c3-feb2-4ed0-998b-3b9b795b0c9c)

[Exact plot values](position-errors.csv): maximum error across both signs,
both FOVs and nine image positions. CSV deltas apply in both directions.
Position changes one horizontal axis at a time. Azimuth and position curves
overlap across camera heights. Both plot axes use logarithmic scales.

[Precision targets](precision-targets.csv): minimum tolerances across the same
FOVs and image positions, for a **100 m example error budget**, **5 km target**
and **35 m camera height**. Columns cover flat, 2% rising and 10% rising terrain.
`*_capped=True` marks a lower bound.

Tilt, roll, FOV and height need particular care at long range on flat terrain.
Actual calibration priority also depends on each parameter's uncertainty.

## Interpretation

Targets apply to **one parameter at a time**. They do not guarantee the same
combined error budget or measured camera accuracy. Agree the operating range
and acceptable position error before adopting them in the epic.

The geometry excludes curvature, refraction, lens distortion, smoke-origin
error and terrain uncertainty. Profiles are extruded sideways; they are not
2D terrain maps. Long-range results are conditional sensitivity calculations.
Tolerance search follows the first sampled local budget crossing and bisects
it. Non-monotonic errors can occur on arbitrary terrain. No-hit includes range
or terrain limits and a camera below the modelled surface. Production geometry
and combined uncertainty remain in [#109](https://github.com/pyronear/vision-rd/issues/109)
and [#113](https://github.com/pyronear/vision-rd/issues/113).
