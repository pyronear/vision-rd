"""Tests for Random Forest classifier."""

import numpy as np

from src.classifier import load_model, predict_tubes, save_model, train_classifier


class TestTrainClassifier:
    def test_basic_training(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 24)
        y = np.array([0] * 25 + [1] * 25)

        clf = train_classifier(X, y, n_estimators=10, max_depth=5)
        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_proba")

    def test_balanced_class_weight(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 24)
        y = np.array([0] * 90 + [1] * 10)  # imbalanced

        clf = train_classifier(X, y, n_estimators=10, max_depth=5)
        # Should still be able to predict both classes
        preds = clf.predict(X)
        assert set(preds) == {0, 1}


class TestPredictTubes:
    def test_output_shapes(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 24)
        y_train = np.array([0] * 25 + [1] * 25)
        clf = train_classifier(X_train, y_train, n_estimators=10, max_depth=5)

        X_test = rng.randn(10, 24)
        predictions, confidences = predict_tubes(X_test, clf)

        assert predictions.shape == (10,)
        assert confidences.shape == (10,)
        assert predictions.dtype == bool
        assert all(0.0 <= c <= 1.0 for c in confidences)


class TestSaveLoadModel:
    def test_round_trip(self, tmp_path):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 24)
        y = np.array([0] * 25 + [1] * 25)

        clf = train_classifier(X, y, n_estimators=10, max_depth=5)
        model_path = tmp_path / "classifier.pkl"
        save_model(clf, model_path)

        loaded_clf = load_model(model_path)
        X_test = rng.randn(5, 24)

        original_preds = clf.predict(X_test)
        loaded_preds = loaded_clf.predict(X_test)
        np.testing.assert_array_equal(original_preds, loaded_preds)
