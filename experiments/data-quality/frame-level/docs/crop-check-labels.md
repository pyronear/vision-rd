# Native 4K crop check on real detections

The engine scores the 4K frame resized to HD. This report checks, on labeled
production detections, whether scoring again on a native-resolution crop around
the box removes false positives before the temporal model.

**TL;DR**

- A crop gate at **crop < 0.1** removes **120 of 155 false positives** (77%) that
  would reach the temporal model today, and keeps **all 10 fires**.
- At crop < 0.25, it removes more false positives (136/155) but already loses
  one fire.
- The crop usually scores real fires higher than HD: 7 of 10 fires go up, and 2
  of the 4 fires that HD scored below 0.25 go above it.
- Other smokes (chimneys, burning, steam) mostly pass the gate, as expected.
  Rejecting them is the temporal model's job.

## Data

The crops come from the zoom capture test (pyro-engine #464), on 6 cameras over
3 sites, from 3 to 23 November 2026. Each record has a detection on the HD frame
and a native crop of the 4K frame centred on the box. The first version used
1024×1024 crops (60 crops) and the later one 1280×720 crops (340 crops).

The sample has 400 crops: one random record per camera, pose and hour, so a long
false positive repeated all day counts once. Each crop was labeled by hand
without seeing the scores:

| fire | other smoke | nothing | unsure |
|---|---|---|---|
| 14 | 69 | 280 | 37 |

**Crop score:** the highest confidence among the crop detections that overlap
the original box, at conf 0.05. The model is
`pyronear/yolo11s_nimble-narwhal_v6.0.0` at 1024. A crop with no overlapping
detection scores 0.

## Results

### Detections that reach the temporal model today (HD ≥ 0.25)

| label | count | removed at crop < 0.1 | removed at crop < 0.25 |
|---|---|---|---|
| nothing | 155 | **120** | 136 |
| other smoke | 48 | 10 | 20 |
| unsure | 27 | 8 | 12 |
| **fire** | 10 | **0** | 1 |

### Detections below the threshold today (HD < 0.25)

| label | count | crop ≥ 0.25 |
|---|---|---|
| nothing | 125 | 7 |
| other smoke | 21 | 6 |
| unsure | 10 | 0 |
| **fire** | 4 | **2** |

The crop raises few false positives above the threshold, but it recovers half
of the fires that HD missed.

## Recommendation

1. Detect on the HD frame, as today.
2. For each box, score a native crop of the 4K frame.
3. Send the box to the temporal model only when **crop ≥ 0.1**.

## Limits

- **14 fires is small**, and 6 of them come from one camera (nemours-01).
  Losing one fire would change the conclusion.
- **Only strong detections.** The zoom test only triggered on ongoing events,
  so this sample says little about weak boxes. The current shadow test
  (pyro-engine #482) checks every box at 0.1 or more, and uploads the 4K frame
  on disagreement so any variant can be replayed.
- **Labels are a single annotator's**, with a 9% "unsure" share.

## Data and reproduce

[`crop-check-labels/labels.jsonl`](crop-check-labels/labels.jsonl) has one line
per crop: the S3 key in `zoom-capture`, the camera, the HD and crop scores, and
the label. The tables above come from it:

```bash
python3 -c "
import json
rows = [json.loads(l) for l in open('crop-check-labels/labels.jsonl')]
for lab in ['nothing', 'other_smoke', 'unsure', 'fire']:
    s = [r for r in rows if r['label'] == lab and r['hd'] >= 0.25]
    print(lab, len(s), sum(r['crop'] < 0.1 for r in s), sum(r['crop'] < 0.25 for r in s))
"
```
