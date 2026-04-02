"""Tests for smokeynet_adapted.dataset."""

import json

import torch

from smokeynet_adapted.dataset import SmokeyNetDataset


def _write_sample(tmp_path, seq_id, num_dets=3, d_model=16, label=1.0):
    """Write a fake .pt + .json pair for a single sequence."""
    pt_data = {
        "roi_features": torch.randn(num_dets, d_model),
        "frame_indices": torch.tensor([0, 0, 1], dtype=torch.long)[:num_dets],
        "bbox_coords": torch.randn(num_dets, 4),
        "detection_labels": torch.zeros(num_dets),
        "sequence_label": torch.tensor(label),
    }
    torch.save(pt_data, tmp_path / f"{seq_id}.pt")

    metadata = {
        "tubes": [
            {
                "tube_id": 0,
                "start_frame": 0,
                "end_frame": 1,
                "entries": [
                    {
                        "frame_idx": 0,
                        "detection": {
                            "class_id": 0,
                            "cx": 0.5,
                            "cy": 0.5,
                            "w": 0.1,
                            "h": 0.1,
                            "confidence": 0.9,
                        },
                    },
                    {
                        "frame_idx": 1,
                        "detection": {
                            "class_id": 0,
                            "cx": 0.5,
                            "cy": 0.5,
                            "w": 0.1,
                            "h": 0.1,
                            "confidence": 0.8,
                        },
                    },
                ],
            }
        ]
    }
    with open(tmp_path / f"{seq_id}.json", "w") as f:
        json.dump(metadata, f)


class TestSmokeyNetDataset:
    def test_len(self, tmp_path):
        _write_sample(tmp_path, "seq_001")
        _write_sample(tmp_path, "seq_002")
        ds = SmokeyNetDataset(tmp_path)
        assert len(ds) == 2

    def test_getitem_keys(self, tmp_path):
        _write_sample(tmp_path, "seq_001")
        ds = SmokeyNetDataset(tmp_path)
        sample = ds[0]
        expected_keys = {
            "sequence_id",
            "roi_features",
            "frame_indices",
            "bbox_coords",
            "detection_labels",
            "sequence_label",
            "tubes",
        }
        assert set(sample.keys()) == expected_keys

    def test_roi_features_shape(self, tmp_path):
        _write_sample(tmp_path, "seq_001", num_dets=5, d_model=32)
        ds = SmokeyNetDataset(tmp_path)
        sample = ds[0]
        assert sample["roi_features"].shape == (5, 32)

    def test_tubes_loaded(self, tmp_path):
        _write_sample(tmp_path, "seq_001")
        ds = SmokeyNetDataset(tmp_path)
        sample = ds[0]
        tubes = sample["tubes"]
        assert len(tubes) == 1
        assert tubes[0].tube_id == 0
        assert len(tubes[0].entries) == 2
        assert tubes[0].entries[0].detection is not None

    def test_sequence_label(self, tmp_path):
        _write_sample(tmp_path, "seq_pos", label=1.0)
        _write_sample(tmp_path, "seq_neg", label=0.0)
        ds = SmokeyNetDataset(tmp_path)
        labels = {
            ds[i]["sequence_id"]: ds[i]["sequence_label"].item() for i in range(2)
        }
        assert labels["seq_pos"] == 1.0
        assert labels["seq_neg"] == 0.0

    def test_gap_entry_in_tube(self, tmp_path):
        """Tube entries with detection=None should be loaded as gaps."""
        metadata = {
            "tubes": [
                {
                    "tube_id": 0,
                    "start_frame": 0,
                    "end_frame": 2,
                    "entries": [
                        {
                            "frame_idx": 0,
                            "detection": {
                                "class_id": 0,
                                "cx": 0.5,
                                "cy": 0.5,
                                "w": 0.1,
                                "h": 0.1,
                                "confidence": 0.9,
                            },
                        },
                        {"frame_idx": 1, "detection": None},
                        {
                            "frame_idx": 2,
                            "detection": {
                                "class_id": 0,
                                "cx": 0.5,
                                "cy": 0.5,
                                "w": 0.1,
                                "h": 0.1,
                                "confidence": 0.8,
                            },
                        },
                    ],
                }
            ]
        }
        with open(tmp_path / "seq_gap.json", "w") as f:
            json.dump(metadata, f)

        pt_data = {
            "roi_features": torch.randn(2, 16),
            "frame_indices": torch.tensor([0, 2]),
            "bbox_coords": torch.randn(2, 4),
            "detection_labels": torch.zeros(2),
            "sequence_label": torch.tensor(1.0),
        }
        torch.save(pt_data, tmp_path / "seq_gap.pt")

        ds = SmokeyNetDataset(tmp_path)
        sample = ds[0]
        tube = sample["tubes"][0]
        assert tube.entries[1].detection is None
