# Contributing to Pyronear Vision R&D

Thank you for your interest in contributing to wildfire smoke detection research!

## Getting Started

Start by reading the [project README](README.md) for an overview of the system and repository structure.

## Key Resources

- [R&D Guidelines](experiments/GUIDELINES.md) — Standards for reproducibility, tooling, and benchmarking
- [Experiments README](experiments/README.md) — How to create and manage experiments
- [Experiment Template](experiments/template/README.md) — Starter project to copy when beginning a new experiment
- [Literature Survey](literature_survey/README.md) — Paper collection, notes, and thematic summary

## How to Contribute

### Adding a Paper

See the [Literature Survey README](literature_survey/README.md) for the full workflow: drop the PDF, update `papers.csv`, write reading notes, and update the summary.

### Running an Experiment

1. Copy the [experiment template](experiments/template/) to a new directory under `experiments/`
2. Follow the [R&D Guidelines](experiments/GUIDELINES.md) for tooling (uv, ruff, DVC) and reproducibility standards
3. Document your objective, approach, data, results, and reproduction steps in the experiment's `README.md`

### Code Style

- Use [uv](https://docs.astral.sh/uv/) for dependency management
- Use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Use [DVC](https://dvc.org/) for data and artifact tracking

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
