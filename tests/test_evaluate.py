import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.evaluate import (
    baseline_costs, bootstrap_ci, cost_curve, expected_calibration_error, expected_cost, metrics_at,
    optimal_threshold, plot_calibration, plot_confusion, plot_cost_curve, plot_pr, plot_roc,
)

Y = np.array([1, 1, 0, 0])
P = np.array([0.9, 0.2, 0.6, 0.1])


def test_expected_cost():
    assert expected_cost(Y, P, 0.5) == 1500.0


def test_expected_cost_custom_weights():
    assert expected_cost(Y, P, 0.5, c_fn=2, c_fp=1) == 750.0


def test_optimal_threshold():
    assert optimal_threshold(Y, P) == 0.11
    assert expected_cost(Y, P, 0.11) == 250.0


def test_cost_curve_shape():
    c = cost_curve(Y, P)
    assert list(c.columns) == ["threshold", "cost_per_1000"] and len(c) == 99


def test_metrics_at():
    m = metrics_at(Y, P, 0.5)
    assert m["recall"] == 0.5 and m["specificity"] == 0.5 and m["precision"] == 0.5
    assert m["cost_per_1000"] == 1500.0 and m["accuracy"] == 0.5
    assert set(m) == {"accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier", "cost_per_1000"}


def test_metrics_single_class():
    m = metrics_at(np.zeros(4), P, 0.5)
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])


def test_baseline_costs():
    assert baseline_costs(Y) == {"semua_review_manual": 500.0, "semua_fast_track": 2500.0}


def test_ece_perfect():
    assert expected_calibration_error(np.array([0, 1] * 50), np.array([0.0, 1.0] * 50)) == 0


def test_ece_miscalibrated():
    assert abs(expected_calibration_error(np.zeros(100), np.full(100, 0.3)) - 0.3) < 1e-9


def test_bootstrap_survives_rare_positive():
    y = np.r_[1, np.zeros(49)]
    p = np.linspace(0, 1, 50)
    t = bootstrap_ci(y, p, 0.5, n_boot=200)
    assert set(t["metrik"]) == {"roc_auc", "pr_auc", "recall"} and (t["n_valid"] <= 200).all()


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    t = bootstrap_ci(y, p, 0.5, n_boot=200).set_index("metrik")
    assert (t["ci_bawah"] <= t["estimasi"]).all() and (t["estimasi"] <= t["ci_atas"]).all()


def test_plots_return_figure():
    curves = {"a": (Y, P)}
    assert isinstance(plot_confusion(Y, P >= 0.5), Figure)
    assert isinstance(plot_roc(curves), Figure) and isinstance(plot_pr(curves), Figure)
    assert isinstance(plot_calibration(curves), Figure)
    assert isinstance(plot_cost_curve({"a": cost_curve(Y, P)}), Figure)
