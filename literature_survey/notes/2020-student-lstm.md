# Lightweight Student LSTM for Wildfire Smoke Detection

- **Title:** Lightweight Wildfire Smoke Detection Using a Shallow Student LSTM with Teacher–Student Framework
- **Authors:** Jeong, M.; Park, M.; Nam, J.; Ko, B.C.
- **Year/Venue:** 2020, Sensors 20(19), 5508
- **PDF:** `pdfs/2020-Lightweight-Student-LSTM-Wildfire-Smoke-Detection-Jeong-et-al.pdf`

## Key Idea

Two-stage pipeline: YOLOv3 detects candidate smoke regions on keyframes, then a
lightweight LSTM verifies candidates temporally using a "smoke-tube" (30 consecutive
cropped frames around the detection). A teacher-student distillation framework
compresses a 3-layer deep LSTM (teacher) into a single-layer shallow LSTM (student)
with negligible accuracy loss, enabling real-time deployment.

## Architecture

```
┌──────────────────┐
│  Video stream     │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Keyframe detect  │ motion-based
│ (patch diff)     │ frame selection
└────────┬─────────┘
         ▼
┌──────────────────┐
│ YOLOv3 detector  │ candidate bbox
│ (416x416)        │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Smoke-tube       │ 30 frames
│ (3s @ 10fps)     │ around keyframe
└────────┬─────────┘
         ▼
┌──────────────────┐
│ ResNet50 feat.   │ 1024-d per frame
│ (216x216)        │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Student LSTM     │ 1 layer, 24 cells
│ (distilled)      │ soft-label loss
└────────┬─────────┘
         ▼
   smoke / non-smoke
```

## Results

### Detector comparison (frame-level, 24 test videos)

| Method          | F1 (%) | Recall (%) | Proc. (s) |
|-----------------|--------|------------|-----------|
| YOLOv3          | 64.17  | 95.51      | 0.014     |
| ELASTIC-YOLOv3  | 68.72  | 91.82      | 0.017     |
| Faster R-CNN    | 52.89  | 98.15      | 0.408     |

### Teacher vs Student LSTM

| Model              | F1 (%)  | Params   | Time (s) |
|--------------------|---------|----------|----------|
| Teacher (3L, 128c) | 87.85   | 861,186  | 0.174    |
| Student (1L, 24c)  | 87.39   | 102,146  | 0.155    |

- Student retains 99.5% of teacher F1 with 8.4x fewer parameters
- TPR 87.78%, TNR 82.00% for student LSTM

## Applicability to Pyronear

**High relevance -- almost identical architecture to Pyronear's two-stage pipeline.**

- Pyronear uses YOLOv8 on RasPi (stage 1) + server verification (stage 2); this
  paper uses YOLOv3 (stage 1) + LSTM verification (stage 2)
- The "smoke-tube" concept maps directly: Pyronear captures frames every 30s,
  so a tube of N consecutive detections from the same camera/region is the
  natural second-stage input
- Keyframe selection via motion detection is directly usable with fixed cameras
  and static backgrounds -- skip frames with no change
- Teacher-student distillation could compress a server-side temporal verifier
  to run on more modest GPU hardware

**Limitations for Pyronear:**
- 30s inter-frame interval is much sparser than the 10fps used here; the smoke-tube
  would span ~15 min instead of 3s, making temporal dynamics very different
- Chimney smoke / fog still confuse the system (noted by authors)
- ResNet50 feature extraction per frame is heavy; could swap for lighter backbone

## Takeaways for Implementation

1. **Smoke-tube at 30s cadence:** Build tubes of N=5-10 consecutive detections from
   the same camera region. Even at 30s intervals, smoke growth/persistence is
   observable and distinguishes real fires from transient FPs
2. **Teacher-student for LSTM:** Train a large LSTM teacher on server, distill to
   1-layer student. Use temperature-softened labels (T=2) and alpha=0.5 blended
   CE loss. Student with 24 cells is sufficient
3. **Keyframe selection is free with fixed cameras:** Simple patch-difference motion
   detection to skip empty frames -- saves YOLOv8 inference cycles on RasPi
4. **High-recall first stage, precision in second stage:** YOLOv3 chosen for recall
   (95.5%) despite low precision (48%); LSTM verification then filters FPs. Same
   philosophy applies to Pyronear's YOLOv8 + server design
5. **Feature dimension:** ResNet50 outputs 1024-d vector per frame -- could use
   lighter feature extractor (MobileNet) and still benefit from temporal LSTM
