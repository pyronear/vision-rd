# Controlled skyline calibration error (#160)

Measure `calibrate()` against known synthetic angles before drawing conclusions
about skyline calibration from the real-frame spread in [#159](repeatability.md).

## Reproduce

```bash
uv run skyline_floor.py --input /path/to/skyline-tilt-images \
  --cache data/02_intermediate/dem --output data/08_reporting/skyline-floor
# Add the per-column edge-noise experiment for #163, in a separate output:
uv run skyline_floor.py --input /path/to/skyline-tilt-images --edge-noise \
  --cache data/02_intermediate/dem --output data/08_reporting/skyline-edge-noise
```

Use the same private bundle as #159. No new images or segmentation inference
are needed; images supply only their dimensions. The committed
[skyline-floor.csv](skyline-floor.csv) contains the original 135 trials from
#161; the additional paired and column-noise cases require a new run. Full signed trials, rejection reasons,
source/runtime hashes, manifest hashes and DSM array hashes stay in the output.
Camera coordinates and image files stay outside Git.

The separate [analytic edge-noise control](edge-noise.md) can be reproduced
without this bundle or terrain downloads. Its results are kept separate
from the original DSM-based cases below.

## Method

- P02 has a distant-horizon sensitivity; P13 is the forest/plain site; P15 has
  the smallest real-frame P95. These three selected views are not a site survey.
- Render the clean 50 km DSM skyline at 1280×720. Truth: tilt 2°, roll 1°,
  declared azimuth +0.037°, coefficient `k = 0.13`. Field of view (FOV) is the platform value
  used as a synthetic truth, not an independently measured input. Mast heights
  are 15/35/20 m; P15's 20 m is assumed.
- Rotate pixel rays as in `pixel_to_ray()`. Render at 0.025° azimuth spacing;
  the fitter uses its own 0.05° skyline and 0.1° search grids, with its initial
  guess at declared azimuth +2° (1.963° from truth). Keep the default ±20°
  search and fit-quality gates. Controls with on-grid azimuth and zero roll
  help distinguish search-grid and rotation effects from raster error.
- Change one input at a time: independent Gaussian noise of 2/4 m standard
  deviation per DSM cell (seeds 0–2), FOV ±1°, canopy ±5 m, `k = 0.10/0.16`,
  and a 25 km cutoff. Camera altitude stays fixed under DSM changes.
- Downsample the perfect mask to 384×384 with area filtering, then restore the
  image dimensions with bilinear filtering. Test an edge shift of ±1 pixel at
  full resolution and separately at 384 px. No learned segmentation is simulated.
- Combined trials use the 384 px mask, ±1 native mask pixel, FOV ±1°, 4 m noise
  (three seeds), +5 m canopy, both refraction endpoints and the 25 km cutoff:
  24 combinations per pose. Report the maximum absolute **total** tilt error,
  not an error with the perfect-case bias subtracted. Read rejected counts with it.
  `combined_no_cutoff` pairs the same 24 combinations and seeds with the full
  rendered 50 km skyline. This isolates the old 25 km truncation; it does not
  test the runtime's maximum 100 km terrain search.
- With `--edge-noise`, shift the perfect full-resolution mask's edge separately
  in each column. Draw Gaussian row errors with standard deviations of 1 and
  2 image pixels, round to whole rows, and share each draw across 1, 16 or 128
  adjacent columns. Use seeds 0–2 for each setting. These are zero-mean draws
  in expectation; their realised mean is not forced to zero. Save the applied
  edge's mean, standard deviation and number of columns with interior edges
  before and after perturbation in `trials.jsonl`. Compare these cases
  with the common shift in `edge_1px` to distinguish variation that averages
  out from systematic bias. All other inputs stay at their perfect values.

## Findings

The clean full-resolution error is **0.00128–0.00146°**. The clean 384 px mask
gives **0.00126–0.00249°**: resolution alone does not rule out the 0.009° plains
target in these controlled cases. The on-grid and zero-roll controls also pass.

The clean floor depends on roll. Read both controls against the approximate
**0.009° plains** and **0.03–0.1° hills** comparison targets from #103:

| Maximum absolute tilt error (°) | P02 | P13 | P15 |
|---|---:|---:|---:|
| Clean full resolution, roll 1° | 0.00146 | 0.00128 | 0.00132 |
| Clean full resolution, roll 0° | 0.00018 | 0.00641 | 0.00158 |
| Clean 384 px mask, roll 1° | 0.00249 | 0.00126 | 0.00214 |
| Common shift, ±1 image pixel | 0.07950 | 0.04589 | 0.04593 |
| Common shift, ±1 native 384 px row | 0.15490 | 0.08939 | 0.08972 |
| FOV ±1° | 0.04056 | 0.03743 | 0.04044 |
| Combined, 25 km cutoff | 0.75439 | 0.17829 | 0.14174 |

A ±1 pixel edge shift gives 0.0459–0.0795° at full resolution and
**0.0894–0.1549° at 384 px**. A ±1° FOV error gives 0.0374–0.0406°.
The 25 km cutoff gives 0.849° on P02 and 0.0660° on P13. Refraction endpoints
alone reach 0.00353–0.00672°; 4 m DSM noise reaches 0.00850° on P15.
All **135/135 fits pass the gates**, yet each pose's worst combined error is
**0.142–0.754°**, above both comparison budgets. Fit acceptance does not certify
tilt accuracy. These are tested maxima, not a confidence interval.

At 720 image rows, one native 384 px mask row is **1.875 image pixels**.
As a local planning approximation, 1 image pixel costs about 0.05° of tilt
and 1° of FOV error costs about 0.04°. Spending the entire 0.009° plains
budget on one term would allow about **0.18 image pixel** of common edge bias
or **0.225° of FOV error**. These are separate, indicative ceilings, not
simultaneously allowable errors. P02's larger pixel sensitivity implies a
tighter edge ceiling, about 0.11 pixel. Model floor, roll, terrain, refraction
and other inputs consume part of the same budget. Independent column noise
must be measured separately from common edge bias; increasing the number of
columns cannot remove a shared offset.

**Decision:** remove the 25 km cutoff first (a runtime setting, addressed in
[smoke-localization#8](https://github.com/pyronear/smoke-localization/pull/8)),
then validate mask edges and measure FOV with the planned
[360° sweep](https://github.com/pyronear/pyro-engine/issues/425); the clean
model reaches the plains target, while field feasibility still needs measured
input errors. More pixels alone do not fix an incorrect boundary or FOV.

## Limits

The renderer and fitter share the DSM skyline sampler and the runtime's angular
camera projection. This is a controlled floor for that implementation/model,
not independent validation of terrain or physical optics. Three noise seeds
do not define a probability bound; real DSM errors can be spatially correlated.
Uniform canopy offsets are scenarios, not a vegetation map. The 0.009° plains
and 0.03–0.1° hills values from #103 are comparison targets, not guarantees for
these poses. Field bias still needs the independent reference in #120.
The column-noise scenarios have only three seeds and idealised block
correlation. They do not establish the error distribution of a learned mask,
a confidence bound, or sub-pixel edge accuracy. The current fitter samples
at most 384 skyline columns even when the input mask is wider.
