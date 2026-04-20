from mtb_change_detection.evaluator import compute_metrics, compute_yolo_only_baseline


def _result(gt: bool, pred: bool, *, confirmed_frame_index: int | None = None) -> dict:
    return {
        "is_positive_gt": gt,
        "is_positive_pred": pred,
        "confirmed_frame_index": confirmed_frame_index if pred else None,
    }


class TestComputeMetrics:
    def test_perfect_predictions(self):
        results = [
            _result(True, True, confirmed_frame_index=2),
            _result(False, False),
        ]
        m = compute_metrics(results)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["fpr"] == 0.0

    def test_all_false_positives(self):
        results = [
            _result(False, True, confirmed_frame_index=1),
            _result(False, True, confirmed_frame_index=1),
        ]
        m = compute_metrics(results)
        assert m["precision"] == 0.0
        assert m["fpr"] == 1.0

    def test_all_false_negatives(self):
        results = [_result(True, False), _result(True, False)]
        m = compute_metrics(results)
        assert m["recall"] == 0.0
        assert m["tp"] == 0
        assert m["fn"] == 2

    def test_mixed(self):
        results = [
            _result(True, True, confirmed_frame_index=2),
            _result(True, False),
            _result(False, True, confirmed_frame_index=1),
            _result(False, False),
        ]
        m = compute_metrics(results)
        assert m["tp"] == 1
        assert m["fn"] == 1
        assert m["fp"] == 1
        assert m["tn"] == 1
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5

    def test_empty(self):
        m = compute_metrics([])
        assert m["num_sequences"] == 0
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

    def test_ttd_computation(self):
        results = [
            {
                "is_positive_gt": True,
                "is_positive_pred": True,
                "confirmed_frame_index": 2,
            },
            {
                "is_positive_gt": True,
                "is_positive_pred": True,
                "confirmed_frame_index": 6,
            },
        ]
        m = compute_metrics(results)
        assert m["mean_ttd_frames"] == 4.0
        assert m["median_ttd_frames"] == 4.0


class TestComputeYoloOnlyBaseline:
    def test_any_detection_triggers_alarm(self):
        results = [
            {
                "is_positive_gt": True,
                "is_positive_pred": False,
                "num_detections_total": 5,
                "confirmed_frame_index": None,
            },
            {
                "is_positive_gt": False,
                "is_positive_pred": False,
                "num_detections_total": 0,
                "confirmed_frame_index": None,
            },
        ]
        m = compute_yolo_only_baseline(results)
        # Sequence with detections → positive, sequence without → negative
        assert m["tp"] == 1
        assert m["tn"] == 1
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_no_detections_all_negative(self):
        results = [
            {
                "is_positive_gt": True,
                "is_positive_pred": False,
                "num_detections_total": 0,
                "confirmed_frame_index": None,
            },
        ]
        m = compute_yolo_only_baseline(results)
        assert m["fn"] == 1
        assert m["tp"] == 0
