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

Generated files stay outside Git: `errors.csv` records signed perturbations and
no-hit cases; `precision.csv` records two-sided limits; `config.json` records
arguments, Python version and source hash. `capped=True` means a lower bound.

## Method

A perspective camera uses azimuth (direction), tilt (up/down angle), roll
(image rotation) and horizontal field of view. Azimuth and roll are clockwise;
positive tilt points down. Sensor aspect ratio is 16:9. Each known target is
placed at nine image locations (`u,v = 0.1,0.5,0.9`). Hold that pixel fixed and
change one parameter. Intersect the resulting ray with each linear terrain
segment; select the first forward hit. Horizontal position errors leave camera
altitude unchanged, separating them from height errors.

Defaults: 0.5/1/2/5/10 km; camera heights 15/35/100 m; FOVs 54.2/87 degrees;
angular errors ±0.01/0.1/1 degree; height/position errors ±0.1/1/5 m. Terrain is
flat, or rises at 2% or 10% after half the target distance. All values are
synthetic; height and FOV include the prototype's
[camera registry](https://github.com/pyronear/smoke-localization/blob/be051f802809b8186cf65a058a174e5d23f4b486/data/cameras.csv).
Use `--help` for options, for example `--budget 50 --heights 35 --distances 1000 5000`.

## Results

Default run: **34,020 perturbations**, **356 no-hit cases**, **5,670 precision
rows**. All nominal targets are visible. Analytical checks cover perspective,
rotations, flat/sloped intersections, height, a nearer ridge and precision limits.

**Position error in metres on flat terrain.** Each cell is the maximum across
both error signs, both fields of view (FOVs), and nine image positions. Change
one parameter at a time; the position column covers either horizontal axis.

| Distance | Camera height | Azimuth ±0.1° | Tilt ±0.01° | Roll ±0.01° | FOV ±0.01° | Height ±1 m | Position ±1 m per axis |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.5 km | 15 m | 0.87 | 2.93 | 1.77 | 1.06 | 33.33 | 1.00 |
| 0.5 km | 35 m | 0.87 | 1.26 | 0.76 | 0.45 | 14.29 | 1.00 |
| 0.5 km | 100 m | 0.87 | 0.45 | 0.27 | 0.16 | 5.00 | 1.00 |
| 1 km | 15 m | 1.75 | 11.78 | 7.09 | 4.23 | 66.67 | 1.00 |
| 1 km | 35 m | 1.75 | 5.02 | 3.03 | 1.81 | 28.57 | 1.00 |
| 1 km | 100 m | 1.75 | 1.77 | 1.06 | 0.64 | 10.00 | 1.00 |
| 2 km | 15 m | 3.49 | 47.65 | 28.55 | 16.98 | 133.33 | 1.00 |
| 2 km | 35 m | 3.49 | 20.15 | 12.14 | 7.24 | 57.14 | 1.00 |
| 2 km | 100 m | 3.49 | 7.02 | 4.24 | 2.53 | 20.00 | 1.00 |
| 5 km | 15 m | 8.73 | 308.86 | 182.30 | 107.47 | 333.33 | 1.00 |
| 5 km | 35 m | 8.73 | 127.86 | 76.54 | 45.50 | 142.86 | 1.00 |
| 5 km | 100 m | 8.73 | 44.03 | 26.53 | 15.84 | 50.00 | 1.00 |
| 10 km | 15 m | 17.45 | 1316.77 | 756.81 | 439.33 | 666.67 | 1.00 |
| 10 km | 35 m | 17.45 | 524.84 | 310.90 | 183.68 | 285.71 | 1.00 |
| 10 km | 100 m | 17.45 | 177.65 | 106.67 | 63.53 | 100.00 | 1.00 |

All selected perturbations give valid hits. `errors.csv` contains the full sweep.

At **5 km**, **35 m height**, with a **100 m example error budget**, the smallest
tolerances across both FOVs and all nine image positions are:

| Parameter | Flat | 2% rising terrain | 10% rising terrain |
|---|---:|---:|---:|
| Azimuth | 1.146 degrees | 1.146 degrees | 1.146 degrees |
| Tilt | 0.00786 degrees | 0.0191 degrees | 0.0639 degrees |
| Roll | 0.0130 degrees | 0.0316 degrees | 0.106 degrees |
| Horizontal FOV | 0.0217 degrees | 0.0528 degrees | 0.177 degrees |
| Height | 0.70 m | 1.70 m | 5.70 m |
| Position across view | 100 m | 100 m | 100 m |
| Position along view | 100 m | >=200 m (cap) | 133 m |

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
