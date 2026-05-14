from pathlib import Path

from scripts.refresh_datasets import plan_copies


def test_plan_copies_emits_six_source_dest_pairs(tmp_path: Path):
    pyro = tmp_path / "pyro-dataset"
    dest = tmp_path / "datasets"
    pairs = plan_copies(pyro, dest)

    expected = {
        (
            pyro / "data" / "processed" / "yolo_test" / "images" / "test",
            dest / "test" / "images",
        ),
        (
            pyro / "data" / "processed" / "yolo_test" / "labels" / "test",
            dest / "test" / "labels",
        ),
        (
            pyro / "data" / "processed" / "yolo_train_val" / "images" / "train",
            dest / "train" / "images",
        ),
        (
            pyro / "data" / "processed" / "yolo_train_val" / "labels" / "train",
            dest / "train" / "labels",
        ),
        (
            pyro / "data" / "processed" / "yolo_train_val" / "images" / "val",
            dest / "val" / "images",
        ),
        (
            pyro / "data" / "processed" / "yolo_train_val" / "labels" / "val",
            dest / "val" / "labels",
        ),
    }
    assert set(pairs) == expected
