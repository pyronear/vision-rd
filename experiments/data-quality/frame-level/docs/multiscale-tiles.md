# Multi-scale detection on 4K frames: global pass + horizon tiles

**Question.** Cameras capture 3840×2160 frames, but the YOLO model runs at 1024,
so a distant smoke plume shrinks 3.75× before the network sees it. Does adding
native-resolution 1024×1024 tiles along the horizon recover small smokes, and
can a Raspberry Pi 5 afford it at 8 frames every 30 s?

**Answer.** Offline, tiles raise recall on the smallest smokes from 55% to 90%
at the cost of about +70% false positives. On the Pi, the
pipeline uses 22.5 s of every 30 s cycle and holds for 30 min with no
throttling. When a temporal job runs at the same time, the cycle grows to
26.5 s, leaving 3.5 s for capture and upload. Both results are proxies: there is no annotated 4K set yet.

## Method

```
3840×2160 frame
├── global pass: whole frame letterboxed to 1024        (what runs today)
└── 5 native 1024×1024 tiles, one row centred on the horizon band
    x0 = 0, 704, 1408, 2112, 2816 · y0 = 568 · ≥30% overlap
→ boxes shifted back to frame coordinates → one NMS (IoU 0.5)
```

- Model: `pyronear/yolo11s_nimble-narwhal_v6.0.0` (trained at imgsz 1024).
- Matching: the audit app's `evaluate_frame` (IoU ≥ 0.3 or containment ≥ 0.7).
- GT size is measured in global-pass pixels (√area after the downscale).

## 1. Offline recall (proxy for 4K)

No annotated 4K frames exist yet, so the test split (1280×720) stands in at
the same ratio: the global pass runs at **352** (1280/352 ≈ 3.6, close to 3840/1024 = 3.75),
tiles are 352×352 native crops, and `full@1024` is the whole frame at full
resolution, as an upper bound. Test split: 2638 images, 1562 GT boxes, 5 tiles
per image, 93% of GT centres inside the tile band (band centre at 0.54 × height).

| conf | variant | <8 px (n=690) | 8–16 px (n=597) | 16–32 px (n=226) | ≥32 px (n=49) | recall | FP/img |
|---|---|---|---|---|---|---|---|
| 0.05 | global | 54.6% | 83.6% | 94.2% | 81.6% | 72.3% | 0.603 |
| 0.05 | tiles | 87.5% | 87.9% | 93.4% | 77.6% | 88.2% | 0.619 |
| 0.05 | **global+tiles** | **89.9%** | 95.3% | 99.6% | 91.8% | **93.4%** | 1.027 |
| 0.05 | full@1024 | 90.9% | 95.5% | 95.6% | 83.7% | 93.1% | 0.399 |
| 0.15 | global | 47.0% | 77.4% | 87.6% | 81.6% | 65.6% | 0.224 |
| 0.15 | global+tiles | 85.4% | 92.6% | 96.9% | 85.7% | 89.8% | 0.434 |
| 0.15 | full@1024 | 85.8% | 94.0% | 92.9% | 79.6% | 89.8% | 0.183 |
| 0.25 | global | 38.8% | 73.0% | 81.4% | 73.5% | 59.2% | 0.121 |
| 0.25 | global+tiles | 82.3% | 89.9% | 92.9% | 79.6% | 86.7% | 0.251 |
| 0.25 | full@1024 | 82.0% | 92.5% | 91.2% | 69.4% | 86.9% | 0.117 |

- Global+tiles matches the full-resolution upper bound on recall, but with
  about twice its false positives.
- Tiles alone miss large smokes that a tile edge cuts; the global pass covers them.
- On the 66 real high-resolution frames available (64 are in the train split,
  so not conclusive): global 95.5% recall / 0.35 FP per image, global+tiles
  98.5% / 0.76.

## 2. Raspberry Pi 5 cost

Pi 5 (4 cores, 4 GB), model exported to NCNN, 4 threads, as in pyro-engine.

| step, one 4K frame | median |
|---|---|
| JPEG decode | 63 ms |
| global pass | 466 ms |
| 5 tiles | 2299 ms (~460 ms/tile) |
| NMS fusion | 0.2 ms |
| **total** | **2828 ms** (today: 529 ms) |

For 8 frames every 30 s: **22.6 s busy** (75%), against 4.2 s today.

Temporal model (`pyronear/temporal-model` v0.6.1, ONNX, embedding cache 4096,
the way `pyro_temporal_api` runs it):

| job | 1 tube | 3 tubes |
|---|---|---|
| first 4-frame job | 1.17 s | 3.55 s |
| next jobs (5→10 frames, cache warm) | 0.21 s | 0.59 s |
| cold 10-frame job | 2.04 s | 6.01 s |

## 3. Endurance, 30 min

60 cycles of 30 s on real 4K frames (58 images from one site). Each frame is
decoded, resized to 1280×720 and JPEG-encoded as pyro-engine does, then run
through the global pass, the 5 tiles and NMS. A separate process runs a
temporal job every cycle, opening 3 tubes on 4 frames. This is pessimistic: it
assumes a new event every 30 s.

| metric | result |
|---|---|
| YOLO busy time per cycle | 22.42 / 22.51 / 22.77 s (min / median / max) |
| temperature | 67.5 °C at start, 70.3 °C max, plateau 69–70 °C after 5 min |
| throttling | none (`0x0`), ARM clock 2400 MHz throughout |
| temporal job | 3.51–3.54 s for 58 jobs, 5.91 s for the first (cold start), 0.01 s once (cache hit) |

The two processes started their 30 s schedules independently, and the logs have
no shared timestamps. Both timings match the standalone ones, so the temporal
job most likely ran in YOLO's idle window. This run measures thermals, not
contention.

**Forced overlap, 5 min.** Both processes now align on the same clock, and the
temporal job starts 5 s into each YOLO cycle. Over 10 cycles:

| metric | alone | overlapping |
|---|---|---|
| YOLO busy time per cycle | 22.5 s | **26.5–26.6 s** |
| slowest frame | 2.8–3.0 s | 5.6–5.8 s |
| temporal job, 3 tubes | 3.5 s | **6.3–6.4 s** |
| temperature, throttling | — | 69.7 °C max, none |

The two loads share the 4 cores. When they overlap, the margin left for camera
capture and upload falls from 7.5 s to 3.5 s per cycle. Capture and upload were
not measured. On a real event, the temporal job grows past 4 frames
(up to 6 s cold, or 0.6 s with a warm cache), which shrinks the margin further.

Raw logs:

- [`endurance.log`](multiscale-tiles/endurance.log)
- [`temporal.log`](multiscale-tiles/temporal.log)
- [`overlap_yolo.log`](multiscale-tiles/overlap_yolo.log)
- [`overlap_temporal.log`](multiscale-tiles/overlap_temporal.log)

## Limits and open points

- **No 4K ground truth.** Recall comes from a 1280×720 proxy. Annotating a
  few hundred 4K frames is the next step before any decision.
- **Desk test.** The Pi ran in open air. A sealed enclosure in summer sun can
  eat the ~10 °C margin before throttling (80 °C).
- **Temporal model sees 1280×720.** A smoke found only by a tile may be too
  small for the temporal model to confirm. To check on real events.
- **More false positives mean more temporal jobs.** Tiles roughly double FP
  per image at conf 0.05.
- **Fixed horizon band.** y0 = 568 assumes the horizon near mid-frame; real
  cameras need a per-pose band (the skyline work in `localisation/` could give it).
- The 2560×1440 frames of one site were excluded from the endurance run (the
  hard-coded 4K tile grid does not fit them).

## Reproduce

Offline (from `experiments/data-quality/frame-level`):

```bash
PYTHONPATH=src uv run python scripts/multiscale_tiles_test.py --model <best.pt> --split-dir data/01_raw/datasets/test --net 352 --ref 1024 --band-cy 0.54
```

On the Pi, from `scripts/pi_bench/`. Inputs:

- `<ncnn_model>`: the model exported with `yolo export model=best.pt format=ncnn imgsz=1024`.
- `<frame.jpg>`: any camera frame (upscaled to 4K in memory).
- `<frames4k>`: real 3840×2160 camera frames, other sizes rejected
  (this run used 58 frames from one site).
- `<seq>`: a camera sequence of at least 10 frames.

The YOLO scripts need `ultralytics`, the temporal ones Python 3.11 with
`temporal-model-core` (as pinned in pyro-engine's `pyro_temporal_api`). The
temporal model is downloaded from the Hugging Face Hub.

```bash
python bench_tiles.py <ncnn_model> <frame.jpg>
```

```bash
python bench_temporal.py <seq>
```

```bash
python endurance.py 60 30 <ncnn_model> <frames4k> > endurance.log & python temporal_load.py 60 30 <seq> 5 > temporal.log
```

Both scripts wait for the same 30 s clock boundary. The last argument of
`temporal_load.py` is how many seconds into each YOLO cycle its job starts.

`temporal_load.py` expects `tm_v061/model_onnx.zip`, which `bench_temporal.py` downloads.
