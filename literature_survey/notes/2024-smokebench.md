# SmokeBench: Evaluating Multimodal Large Language Models for Wildfire Smoke Detection

- **Authors:** Tianye Qi, Weihao Li, Nick Barnes
- **Year/Venue:** 2024 (arXiv:2512.11215v1, Dec 2025)
- **PDF:** `/mnt/data/ssd_1/earthtoolsmaker/projects/pyronear/papers/pdfs/2024-SmokeBench-Multimodal-LLM-Wildfire-Smoke-Detection.pdf`

## Key Idea

SmokeBench is a benchmark evaluating how well multimodal large language models (MLLMs) can detect and localize wildfire smoke. The benchmark uses the FIgLib dataset and tests 7 MLLMs (open and closed source) across four progressively harder tasks: smoke classification, tile-based localization, grid-based localization, and bounding-box smoke detection. The key finding is that all current MLLMs fail at early-stage smoke detection -- performance is strongly correlated with smoke volume (area), and no model can reliably localize small or faint smoke plumes.

## Architecture / Content

SmokeBench defines four tasks of increasing difficulty:

```
Task 1: SMOKE CLASSIFICATION (binary: smoke/no-smoke per image)
    |
Task 2: TILE-BASED LOCALIZATION (3x4 grid, classify each tile)
    |
Task 3: GRID-BASED LOCALIZATION (5x5 grid, predict which cells have smoke)
    |
Task 4: SMOKE DETECTION (output bounding box coordinates)

Models tested:
  Open-source: Idefics2 (8B), Qwen2.5-VL (7B, 32B), InternVL3 (14B)
  Grounding: Unified-IO 2, Grounding DINO
  Closed-source: GPT-4o, Gemini-2.5 Pro

Dataset: FIgLib subset -- 5,046 positive images + 1,000 negatives
Factors analyzed: smoke area (5 quantile bins) and Weber contrast (5 bins)
```

## Results

### Smoke Classification (accuracy by smoke area):
| Model | Very Small | Small | Medium | Large | Very Large | Overall |
|---|---|---|---|---|---|---|
| Idefics2 (8B) | 0.482 | **0.596** | **0.598** | 0.575 | **0.709** | **0.592** |
| Qwen2.5-VL (7B) | 0.098 | 0.232 | 0.423 | 0.463 | 0.686 | 0.380 |
| GPT-4o | **0.529** | **0.727** | **0.770** | **0.831** | **0.931** | **0.758** |

### Smoke Detection (bounding-box mIoU):
| Model | mIoU |
|---|---|
| Idefics2, Qwen2.5-VL, InternVL3 | 0.000 |
| Unified-IO 2 | 0.092 |
| Grounding DINO | 0.245 |
| **YOLOv8n (reference)** | **0.773** |

### Key findings:
- Smoke **area** is the dominant factor; contrast has marginal impact
- All MLLMs fail on "Very Small" smoke (early stage) -- exactly when detection matters most
- Even the best MLLM (GPT-4o, 0.758 classification accuracy) is far below specialized detectors
- For localization, all general-purpose MLLMs produce near-zero mIoU
- Grounding DINO (0.245 mIoU) is the best MLLM-based localizer but still far below YOLOv8n (0.773)

## Applicability to Pyronear

**Low direct applicability for edge deployment, but important strategic insight.**

- **MLLMs are NOT viable for Pyronear's use case:** The paper conclusively shows that current MLLMs cannot detect early-stage smoke (small area, low contrast). They are too computationally expensive for edge and too inaccurate for the task. This validates Pyronear's choice of specialized YOLO detectors.
- **MLLMs as server-side second opinion (future):** If MLLM costs and latency decrease, they could potentially serve as a third verification stage for ambiguous detections, providing natural-language explanations of why a detection is or is not smoke. However, their current failure on small smoke makes this premature.
- **Confirms smoke area as the critical challenge:** Pyronear's cameras view distant landscapes where smoke plumes occupy tiny portions of the image. This is precisely where MLLMs fail most.
- **Tile-based approach is interesting:** Cropping the image into tiles and classifying each independently could be a lightweight way to narrow the region of interest, but even this fails for small smoke.

## Takeaways for Implementation

1. **Do NOT consider replacing YOLO with MLLMs** for smoke detection -- specialized detectors outperform by 3x+ on mIoU.
2. **Smoke area is the bottleneck,** not contrast. Focus on improving detection of small-area smoke plumes through: higher resolution crops, multi-scale detection, temporal accumulation.
3. **The tile-classification idea** (dividing image into grid, classifying tiles) could be tested as a lightweight pre-filter on the RaspPi to reduce the search space before running YOLO on promising regions.
4. **Monitor MLLM progress** -- if models like GPT-5/6 significantly improve small-object grounding, revisiting MLLMs as a server-side verification layer could be worthwhile.
5. **FIgLib dataset** (25K images from HPWREN) is available and already annotated; PyroNear2025 paper re-annotated it with bounding boxes.
