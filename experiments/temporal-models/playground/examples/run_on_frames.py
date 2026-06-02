"""Minimal example: run the bbox-tube-temporal model on a set of frames.

Run from the experiment root:
    uv run python examples/run_on_frames.py
"""

from pathlib import Path

from bbox_tube_temporal.model import BboxTubeTemporalModel

# 1. Point at a packaged model (.zip) and a sequence of frames.
PACKAGE = Path("data/01_raw/models/bbox-tube-vit-dinov2/model.zip")
SEQUENCE_DIR = next(Path("data/01_raw/sample_sequences").glob("smoke-*"))

# 2. Frames must be temporally ordered. The flattened sample dirs sort correctly.
frame_paths = sorted(SEQUENCE_DIR.glob("*.jpg"))

# 3. Load the model (device=None auto-selects cuda / mps / cpu) and run it.
#    predict_sequence() turns the image paths into Frames and classifies the sequence.
model = BboxTubeTemporalModel.from_package(PACKAGE, device=None)
out = model.predict_sequence(frame_paths)

# 4. Read the decision.
print(f"sequence: {SEQUENCE_DIR.name}")
print(f"is_positive (smoke): {out.is_positive}")
print(f"trigger_frame_index: {out.trigger_frame_index}")
if out.trigger_frame_index is not None:
    print(f"trigger frame file:  {frame_paths[out.trigger_frame_index].name}")

# 5. `details` carries the per-tube breakdown (logits, probabilities, decision).
kept = out.details.get("tubes", {}).get("kept", [])
print(f"kept tubes: {len(kept)}")
