# Calibration sensitivity

Reproducible sensitivity study for [vision-rd#103](https://github.com/pyronear/vision-rd/issues/103).
It measures how each calibration error moves a known ground point, and derives
conditional calibration tolerances for a chosen horizontal position-error budget.

## Reproduce

```bash
cd experiments/localisation/calibration-sensitivity
uv run --python 3.11 study.py --check
uv run --python 3.11 study.py
ruff check --select E,F,I,W,UP,B,SIM --target-version py311 study.py
ruff format --check study.py
```

This is a standard-library-only script with inline uv metadata. No package,
environment, notebook, model, or data download is needed. Generated CSVs are
ignored by Git. `config.json` records the arguments, Python version and input
hashes. There are no random samples.

Outputs:
- `errors.csv`: signed parameter errors, along/across displacement, total
  horizontal error, and separate `no_hit` results.
- `precision.csv`: two-sided tolerances for each case and parameter. `capped=True`
  means the tolerance is **at least** the configured cap, not an exact limit.
- `config.json`: run configuration and SHA-256 hashes.

## Geometry and data

Use a perspective pinhole camera with 16:9 aspect ratio and full azimuth, tilt
and roll rotations. Coordinates are right, forward and up in local metres.
Azimuth and roll are clockwise; positive tilt points down. Height is above the
terrain at the true camera location. Horizontal position errors leave its
vertical coordinate unchanged, so height and position errors remain separate.

The same known point is placed at nine image locations (`u,v = 0.1,0.5,0.9`).
For each case, derive the nominal pose, hold the image pixel fixed, and change
one parameter. All parameters use both error signs. The terrain intersection
is exact for each linear profile segment and selects the first forward hit.
This avoids hiding roll/FOV sensitivity by testing only the centre pixel.

Defaults: 0.5/1/2/5/10 km; 15/35/100 m camera heights; 54.2/87 degree horizontal
FOVs; angular errors of 0.01/0.1/1 degree; position/height errors of 0.1/1/5 m.
Terrain is flat, or rises at a 2% or 10% grade after half the target distance.
The heights and FOVs include those in the prototype's
[camera registry](https://github.com/pyronear/smoke-localization/blob/be051f802809b8186cf65a058a174e5d23f4b486/data/cameras.csv).

Optional terrain input is a continuous CSV profile with columns
`distance_m,elevation_m`, in metres. Distances must increase and cover zero
and every requested target. The profile is extruded sideways; it is not a
complete 2D terrain map. The camera is at distance zero. Occluded targets are
excluded from calibration recommendations and recorded in `precision.csv`.

```bash
uv run --python 3.11 study.py --budget 50 --heights 35 --distances 1000 5000
uv run --python 3.11 study.py --profile terrain.csv --max-range 15000
```

## Results

The default run produces **34,020 perturbations**, including **356 no-hit
results**, and 5,670 precision rows. All nominal synthetic targets are visible.
The checks cover analytical flat/sloped intersections, all sampled image
positions, off-axis perspective, roll, height, horizon/range failures, a nearer
ridge, and analytical calibration tolerances.

The table below reports **horizontal position error in metres on flat terrain**.
Each cell is the largest error across both signs, both fields of view (FOVs)
and all nine image positions. Each parameter changes separately. The position
column covers either horizontal axis; it does not combine both errors.

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

The full parameter sweep, including rising terrain and larger errors, is in
the generated `errors.csv`. The selected perturbations above all give valid hits.

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

These are **one-parameter-at-a-time limits**, not a combined error budget or
measured camera accuracy. Tilt, roll, FOV and height merit particular attention
at long range on flat ground. Actual priority also depends on how uncertain
each parameter is. Choose the operational range and acceptable position error
before adopting targets in the epic.

Additional checks used GLO-30 profiles through Brison (311 degrees) and
Croix-Augas (212 degrees), from the prototype's sample views. Thirty centre-ray
comparisons against its pinned ray/terrain implementation differed by at most
**0.541 m**. Five of ten nominal terrain targets were behind nearer terrain;
the study flags them instead of recommending precision for an invisible point.
This validates the profile calculation on real terrain, not localisation on
real fires. Those downloaded inputs and reports are outside Git.

## Limits

The baseline intentionally matches the prototype's straight vertical ray:
no Earth curvature, refraction, lens distortion, smoke-origin error or terrain
height uncertainty. A 10 km result is a conditional sensitivity calculation,
not a claim of absolute geographic accuracy. GLO-30 includes canopy/buildings.
Do not treat a visible plume above a ridge as a visible ground origin.

Tolerance search follows the first sampled local budget crossing, then bisects
it; arbitrary terrain can have discontinuous or non-monotonic errors. No-hit
includes a ray outside terrain coverage/range or a camera below the modelled
surface. Production geometry and combined uncertainty remain
[#109](https://github.com/pyronear/vision-rd/issues/109) and
[#113](https://github.com/pyronear/vision-rd/issues/113).
