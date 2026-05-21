# tests/test_run_models.py
import temporal_model_explorer.run_models as rm
from temporal_model_explorer.run_models import load_models, run_over_store
from temporal_model_explorer.store import FrameRef, SequenceMeta, write_meta


class _Output:
    def __init__(self, is_positive, trigger, details):
        self.is_positive = is_positive
        self.trigger_frame_index = trigger
        self.details = details


class _FakeModel:
    """Keeps a sequence iff it has >= 3 frames; triggers on frame index 1."""

    def predict(self, frames):
        keep = len(frames) >= 3
        details = (
            {"tubes": {"kept": [{"probability": 0.9}]}}
            if keep
            else {"tubes": {"kept": []}}
        )
        return _Output(keep, 1 if keep else None, details)


def _seq(store, key, label, n):
    d = store / "local_zip" / key
    (d / "images").mkdir(parents=True)
    frames = []
    for i in range(n):
        (d / "images" / f"detection_{i}.jpg").write_bytes(b"x")
        frames.append(FrameRef(file=f"images/detection_{i}.jpg", detection_id=i))
    write_meta(
        d,
        SequenceMeta(
            key=key,
            sequence_id=key.split("_")[-1],
            source="local_zip",
            label=label,
            label_detail=None,
            label_source="zip_folder",
            frames=frames,
        ),
    )


def test_run_over_store_writes_results(tmp_path):
    store = tmp_path / "sequences"
    _seq(store, "zip_1", "smoke", 4)  # kept  -> kept-smoke
    _seq(store, "zip_2", "fp", 2)  # discarded -> discarded-fp
    results = tmp_path / "out" / "results.parquet"
    details = tmp_path / "out" / "details"

    df = run_over_store(store, {"fake": _FakeModel()}, results, details)

    assert results.exists()
    rows = {r["key"]: r for r in df.to_dict("records")}
    assert rows["zip_1"]["decision"] == "keep"
    assert rows["zip_1"]["outcome"] == "kept-smoke"
    assert rows["zip_1"]["trigger_frame_file"] == "images/detection_1.jpg"
    assert rows["zip_1"]["probability"] == 0.9
    assert rows["zip_2"]["decision"] == "discard"
    assert rows["zip_2"]["outcome"] == "discarded-fp"
    assert rows["zip_2"]["trigger_frame_file"] is None
    assert (details / "fake" / "zip_1.json").exists()


def test_load_models_scans_dir(tmp_path, monkeypatch):
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "model.zip").write_bytes(b"x")
    (tmp_path / "empty").mkdir()  # no model.zip -> skipped
    monkeypatch.setattr(
        rm.BboxTubeTemporalModel,
        "from_package",
        staticmethod(lambda p, device="cpu": f"loaded:{p}"),
    )
    models = rm.load_models(tmp_path)
    assert set(models) == {"m1"}


def test_load_models_missing_dir(tmp_path):
    assert load_models(tmp_path / "nope") == {}
