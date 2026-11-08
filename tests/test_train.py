import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src.train import (
    MODEL_NAMES, SCORING, cv_evaluate, make_cv, make_model_specs, make_pipeline, oof_proba, tune,
)


def test_specs_cover_models():
    assert list(make_model_specs(4.0)) == MODEL_NAMES == ["dummy", "logreg", "rf", "xgb"]


def test_xgb_pos_weight():
    assert make_model_specs(4.0)["xgb"][0].get_params()["scale_pos_weight"] == 4.0


def test_specs_use_random_state():
    for name, (est, _) in make_model_specs(4.0).items():
        assert est.get_params().get("random_state") == config.RANDOM_STATE, name


def test_smote_pipeline_order():
    p = make_pipeline(LogisticRegression(), smote=True)
    assert [n for n, _ in p.steps] == ["prep", "smote", "clf"]


def test_cv_is_stratified_seeded():
    cv = make_cv()
    assert cv.n_splits == config.CV_FOLDS and cv.shuffle and cv.random_state == config.RANDOM_STATE


@pytest.mark.parametrize("name", ["dummy", "logreg", "rf", "xgb"])
def test_tune_returns_fitted_pipeline(synthetic_xy, name):
    X, y = synthetic_xy
    r = tune(name, X, y, n_iter=2)
    assert r.pipeline.predict_proba(X).shape == (len(X), 2) and r.fit_seconds > 0
    assert r.name == name and r.tuned_on_n == len(X)


def test_tune_subsample_rule(synthetic_xy, monkeypatch):
    monkeypatch.setattr(config, "TUNE_SUBSAMPLE_TRIGGER", 100)
    monkeypatch.setattr(config, "TUNE_SUBSAMPLE", 200)
    X, y = synthetic_xy
    r = tune("logreg", X, y, n_iter=2)
    assert r.tuned_on_n == 200
    assert r.pipeline.predict_proba(X).shape[0] == len(X)


def test_oof_proba_range(synthetic_xy):
    X, y = synthetic_xy
    p = oof_proba(make_pipeline(LogisticRegression(max_iter=500)), X, y)
    assert len(p) == len(y) and ((0 <= p) & (p <= 1)).all()


def test_cv_evaluate_columns(synthetic_xy):
    X, y = synthetic_xy
    t = cv_evaluate(make_pipeline(LogisticRegression(max_iter=500)), X, y)
    assert list(t.columns) == ["metrik", "mean", "std"] and set(t["metrik"]) == set(SCORING)
    assert np.isfinite(t["mean"]).all()
