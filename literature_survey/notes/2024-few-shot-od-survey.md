# Beyond Few-Shot Object Detection: A Detailed Survey

- **Authors:** Vishal Chudasama, Hiran Sarkar, Pankaj Wasnik, Vineeth N Balasubramanian, Jayateja Kalla
- **Year/Venue:** 2024 / ACM Computing Surveys (arXiv:2408.14249)
- **PDF:** `../pdfs/2024-Beyond-Few-Shot-Object-Detection-Survey.pdf`

## Key Idea

This survey comprehensively covers five categories of few-shot object detection (FSOD): (1) Standard FSOD -- detect novel classes with few examples while trained on base classes, (2) Generalized FSOD (G-FSOD) -- maintain base-class performance while learning novel classes, (3) Incremental FSOD (I-FSOD) -- add classes without retraining on old ones, (4) Open-set FSOD (O-FSOD) -- detect unknown objects beyond trained classes, and (5) Domain Adaptation FSOD (FSDAOD) -- adapt a detector to a new domain with few examples. The paper provides a unified taxonomy, timeline, benchmark comparisons, and identifies open challenges.

## Architecture (General FSOD Framework)

```
┌───────────────────────────┐
│ Stage 1: Base Training    │
│                           │
│ Abundant base-class data  │
│         ▼                 │
│ ┌─────────────────────┐   │
│ │ Backbone (shared)   │   │
│ └─────────┬───────────┘   │
│           ▼               │
│ ┌─────────────────────┐   │
│ │ Base class heads     │   │
│ └─────────────────────┘   │
└───────────────────────────┘
              ▼
┌───────────────────────────┐
│ Stage 2: Few-Shot Tuning  │
│                           │
│ K examples of novel class │
│         ▼                 │
│ ┌─────────────────────┐   │
│ │ Pretrained Backbone │   │
│ │ (frozen or ft)      │   │
│ └─────────┬───────────┘   │
│           ▼               │
│ ┌─────────────────────┐   │
│ │ Novel class heads    │   │
│ └─────────────────────┘   │
└───────────────────────────┘
```

## Results

- The survey does not present new experimental results but compiles benchmarks across FSOD variants
- Standard FSOD: best methods achieve ~20-30 nAP50 on PASCAL VOC novel split (K=1-10 shots)
- G-FSOD: TFA baseline still competitive; newer methods (NIFF, DiGeo) improve base+novel trade-off
- FSDAOD: most relevant for cross-domain scenarios (e.g., weather/scene changes)
- Key finding: **meta-learning approaches dominate standard FSOD**, while **fine-tuning approaches dominate G-FSOD and FSDAOD**
- FSOD timeline: LSTD (2018) was first; field has expanded rapidly since 2020

## Applicability to Pyronear

**What transfers:**
- **Domain Adaptation FSOD (FSDAOD) is the most relevant variant.** When deploying cameras at new sites with different landscapes, vegetation, and lighting, Pyronear needs to adapt its detector with few labeled examples from that site.
- **Fine-tuning based approaches** (TFA-style: freeze backbone, fine-tune detection heads on K smoke examples from new site) are simple, effective, and directly applicable.
- **Incremental FSOD** is relevant when adding new event types (e.g., different smoke types, dust, fog) without forgetting the base smoke detector.
- **Open-set FSOD** concepts could help detect unknown anomalies (unusual smoke patterns, new fire types) that were not in training data.
- **Class imbalance handling** techniques from FSOD (focal loss, balanced sampling) apply to Pyronear's imbalanced dataset (rare fire vs abundant no-fire).

**What doesn't transfer:**
- Most FSOD work assumes natural images with diverse object categories (PASCAL VOC, COCO). Pyronear deals with a single-class detection problem (smoke) in a narrow visual domain.
- FSOD's multi-class generalization concern is less relevant when the target is always "smoke."
- Many methods assume a large base-class dataset exists -- Pyronear may lack this variety.
- The survey focuses on static images, not temporal/video detection.

## Takeaways for Implementation

1. **Site-specific adaptation via FSDAOD:** When deploying at a new site, collect 5-10 annotated smoke examples from that site and fine-tune the detection head (freeze YOLOv8 backbone, update head).
2. **TFA (Two-stage Fine-tuning Approach) is a strong, simple baseline.** Pre-train on all available fire data, then fine-tune the last layers on per-site examples.
3. **For seasonal/weather adaptation:** Use domain adaptation FSOD techniques to adapt the model as visual conditions change (summer vs winter, clear vs hazy).
4. **Consider G-FSOD to avoid catastrophic forgetting** when adding new sites -- ensure the model does not degrade on existing sites when fine-tuned for a new one.
5. **Data augmentation strategies from FSOD** (cut-paste, mosaic with smoke crops) can synthetically increase the effective training set for rare fire events.
