"""Unit tests for src/evaluation/metrics.py"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.metrics import compute_metrics, aggregate_metrics, EvalMetrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
        m = compute_metrics(y, y, project="test")
        assert m.mae == pytest.approx(0.0)
        assert m.rmse == pytest.approx(0.0)
        assert m.accuracy_at_1 == pytest.approx(1.0)

    def test_known_mae(self):
        y_true = np.array([2.0, 4.0, 6.0])
        y_pred = np.array([1.0, 3.0, 5.0])
        m = compute_metrics(y_true, y_pred, project="test")
        assert m.mae == pytest.approx(1.0)
        assert m.accuracy_at_1 == pytest.approx(1.0)

    def test_accuracy_at_1_half(self):
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([2.0, 5.0])
        m = compute_metrics(y_true, y_pred, project="test")
        assert m.accuracy_at_1 == pytest.approx(0.5)

    def test_project_name_stored(self):
        y = np.array([3.0, 5.0])
        m = compute_metrics(y, y, project="moodle")
        assert m.project == "moodle"

    def test_test_size_correct(self):
        y = np.ones(42)
        m = compute_metrics(y, y, project="x")
        assert m.test_size == 42

    def test_to_dict_keys(self):
        y = np.array([1.0, 2.0])
        m = compute_metrics(y, y, project="springxd")
        assert set(m.to_dict().keys()) == {"project", "test_size", "mae", "rmse", "accuracy_at_1"}


class TestAggregateMetrics:
    def _make(self, maes, rmses, accs):
        return [EvalMetrics(f"p{i}", 100, m, r, a)
                for i, (m, r, a) in enumerate(zip(maes, rmses, accs))]

    def test_mean_mae(self):
        agg = aggregate_metrics(self._make([1.0, 3.0], [1.0, 3.0], [0.8, 0.6]))
        assert agg["mean_mae"] == pytest.approx(2.0)

    def test_num_projects(self):
        agg = aggregate_metrics(self._make([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.9, 0.8, 0.7]))
        assert agg["num_projects"] == 3

    def test_std_mae_zero(self):
        agg = aggregate_metrics(self._make([2.0, 2.0], [2.0, 2.0], [0.5, 0.5]))
        assert agg["std_mae"] == pytest.approx(0.0)

    def test_mean_accuracy(self):
        agg = aggregate_metrics(self._make([1.0, 1.0], [1.0, 1.0], [0.4, 0.6]))
        assert agg["mean_accuracy_at_1"] == pytest.approx(0.5)
