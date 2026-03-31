# 🧪 Pyronear Vision R&D

Research and development for ML-based wildfire smoke detection at [Pyronear](https://pyronear.org/).

## 🏗️ System Overview

Pyronear deploys fixed 360° cameras on antenna towers that capture images every 30 seconds. On each Raspberry Pi, a YOLO model detects smoke candidates. Full frames and bounding box info are sent to a server where a second-stage temporal model verifies detections — reducing false positives while maintaining high recall.

## 📂 Repository Structure

```
vision-rd/
├── lib/                  # Shared packages used across experiments
│   └── pyrocore/        # Types, protocols, and base model (TemporalModel ABC)
├── literature_survey/   # Paper collection, notes, and thematic summary
├── experiments/          # R&D experiments (each a self-contained uv project)
│   ├── template/        # Starter project — copy to begin a new experiment
│   ├── README.md        # How to create and manage experiments
│   └── GUIDELINES.md    # Standards: uv, ruff, DVC, reproducibility
├── CONTRIBUTING.md      # How to contribute
├── README.md            # This file
└── LICENSE              # Apache 2.0
```

## 🔗 Quick Links

- 📚 [Literature Survey](literature_survey/README.md) — 28 papers on temporal models, video foundations, smoke detection, and related topics
- 🧪 [Experiments README](experiments/README.md) — How to start a new experiment
- 📏 [R&D Guidelines](experiments/GUIDELINES.md) — Standards for reproducibility, tooling, and benchmarking

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved — adding papers, running experiments, and code style conventions.
