# Multi-scale detection on 4K frames

Cameras capture 3840×2160 frames, but YOLO runs at 1024, so distant smoke
shrinks 3.75× before the model sees it. This report tests whether native 4K
crops help, and whether the Pi can afford them.

**TL;DR**

- **4K is worth it.** Adding native-resolution tiles along the horizon raises
  recall on the smallest smokes from 55% to 90%.
- **Full tiling is too heavy for the Pi.** One cycle of 8 frames takes 22.5 s
  instead of 4.2 s, and 26.5 s when the temporal model runs at the same time.
  That leaves almost no margin, and long-term heat in an enclosure is a risk.
- **Recommendation, still to test:** run the global pass at a low confidence
  threshold, then re-detect only on 4K crops around the candidates.

## Method

```
3840×2160 frame
├── global pass: whole frame letterboxed to 1024            (today)
└── 5 native 1024×1024 tiles, one row on the horizon band   (≥30% overlap)
→ boxes back to frame coordinates → one NMS (IoU 0.5)
```

The model is `pyronear/yolo11s_nimble-narwhal_v6.0.0`. Matching uses the audit
app's rule: IoU ≥ 0.3 or containment ≥ 0.7.

## Offline recall

There is no annotated 4K set yet, so the 1280×720 test split stands in at the
same scale ratio:

- **global:** the whole frame at 352 px.
- **tiles:** native 352 px crops.
- **full:** the whole frame at full resolution, as an upper bound.

GT size is measured in global-pass pixels.

| conf 0.05 (2638 imgs, 1562 GT) | <8 px | 8–16 px | ≥16 px | recall | FP/img |
|---|---|---|---|---|---|
| global (today) | 54.6% | 83.6% | 92.0% | 72.3% | 0.60 |
| **global + tiles** | **89.9%** | 95.3% | 98.2% | **93.4%** | 1.03 |
| full resolution (upper bound) | 90.9% | 95.5% | 93.5% | 93.1% | 0.40 |

The same holds at confidence 0.15 and 0.25: recall goes from 66% to 90% and from
59% to 87%. Tiles reach the full-resolution recall, but with about 2× its false
positives.

## Raspberry Pi 5 cost

The model runs as NCNN on 4 threads. The load is 8 frames every 30 s, plus the
temporal model (`temporal-model` v0.6.1, ONNX).

| | today | global + 5 tiles |
|---|---|---|
| one 4K frame | 0.53 s | 2.83 s |
| 8 frames, alone | 4.2 s | **22.5 s** |
| 8 frames, with a 3-tube temporal job at the same time | — | **26.5 s** (temporal job: 3.5 → 6.3 s) |

Endurance test, 30 min (60 cycles):

- The cycle time stayed stable.
- Temperature plateaued at 69–70 °C, with 70.3 °C at most.
- There was no throttling.

The Pi ran in open air on a desk. The throttling limit is 80 °C, so a sealed
enclosure in summer sun could use up that margin. Camera capture and upload were
not measured, and the cycle leaves them 3.5 s.

## Next step: two-stage detection

1. **Global pass at a low threshold**, for example 0.05, to get candidates. This
   costs the same as today.
2. **One 1024 crop from the 4K frame around each candidate**, re-detected at
   native resolution, then a normal threshold to confirm.

Each candidate costs about 0.46 s. With few candidates per frame, the cycle stays
close to today's 4 s while keeping the 4K detail. To measure: recall against the
global + tiles variant, the number of candidates per frame, and the cycle time
on the Pi.

## Limits

- The recall figures come from a 1280×720 proxy. Annotating a few hundred 4K
  frames is needed before any decision.
- The tile band uses a fixed y position. Real cameras need one per pose.
- The temporal model sees 1280×720 frames, so it may fail to confirm a smoke that
  only 4K reveals.

## Reproduce

- Offline: [`scripts/multiscale_tiles_test.py`](../scripts/multiscale_tiles_test.py).
- Pi: [`scripts/pi_bench/`](../scripts/pi_bench/). The scripts take the NCNN
  model, the frames and the temporal sequence as arguments.
- Raw Pi logs: [`multiscale-tiles/`](multiscale-tiles/).

```bash
PYTHONPATH=src uv run python scripts/multiscale_tiles_test.py --model <best.pt> --split-dir data/01_raw/datasets/test --net 352 --ref 1024 --band-cy 0.54
```

```bash
python endurance.py 60 30 <ncnn_model> <frames4k> > yolo.log & python temporal_load.py 60 30 <seq> 5 > temporal.log
```
