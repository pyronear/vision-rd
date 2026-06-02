"""Pure helpers for the playground CLI: input resolution and output formatting."""

from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def resolve_frames(inputs: list[str]) -> list[Path]:
    """Resolve CLI input(s) to a temporally ordered list of image paths.

    - A single existing directory → its images, sorted by filename.
    - One or more file paths → those paths, in the given order.

    Raises:
        FileNotFoundError: a given path does not exist.
        ValueError: a directory contains no images, or no input was given.
    """
    if not inputs:
        raise ValueError("no input frames given")

    if len(inputs) == 1 and Path(inputs[0]).is_dir():
        directory = Path(inputs[0])
        frames = sorted(
            p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not frames:
            raise ValueError(
                f"no images ({', '.join(IMAGE_EXTENSIONS)}) in {directory}"
            )
        return frames

    paths = [Path(p) for p in inputs]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    return paths


def resolve_model_package(
    *,
    model: str | None,
    model_package: Path | None,
    models_dir: Path,
) -> Path:
    """Resolve which ``.zip`` package to load. Exactly one selector required.

    - ``model``: a name under ``models_dir`` → ``models_dir/<name>/model.zip``.
    - ``model_package``: an explicit path to a ``.zip``.

    Raises:
        ValueError: neither or both selectors were given.
        FileNotFoundError: the resolved package does not exist (with available
            names listed when resolving by ``model``).
    """
    if (model is None) == (model_package is None):
        raise ValueError("pass exactly one of --model or --model-package")

    if model_package is not None:
        if not model_package.is_file():
            raise FileNotFoundError(model_package)
        return model_package

    pkg = models_dir / model / "model.zip"
    if not pkg.is_file():
        available = (
            sorted(p.name for p in models_dir.iterdir() if p.is_dir())
            if models_dir.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"no package for model {model!r} at {pkg}. Available: {available}"
        )
    return pkg


def max_probability(details: dict | None) -> float | None:
    """Largest calibrated probability across kept tubes, or None if unavailable."""
    kept = (details or {}).get("tubes", {}).get("kept", [])
    probs = [t.get("probability") for t in kept if t.get("probability") is not None]
    return max(probs) if probs else None


def format_summary(out, frame_paths: list[Path], runtime_ms: float) -> str:
    """Human-readable one-block summary of a ``pyrocore.TemporalModelOutput``.

    ``out`` is intentionally untyped to keep this module free of torch-heavy
    imports; it is a ``pyrocore.TemporalModelOutput``.
    """
    n = len(frame_paths)
    if not out.is_positive:
        return f"NO SMOKE ✗   frames={n}   runtime={runtime_ms:.0f}ms"

    idx = out.trigger_frame_index
    trigger = f"frame {idx}"
    if idx is not None and 0 <= idx < n:
        trigger += f"  ({frame_paths[idx].name})"
    prob = max_probability(out.details)
    prob_str = f"{prob:.2f}" if prob is not None else "n/a"
    return (
        f"SMOKE ✓   trigger={trigger}\n"
        f"probability={prob_str}   frames={n}   runtime={runtime_ms:.0f}ms"
    )
