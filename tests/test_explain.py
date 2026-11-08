import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src import config
from src.explain import (
    fairness_table, readable_name, reason_codes, segment_metrics, shap_explain, unwrap_pipeline,
)
from src.train import tune

L = config.FEATURE_LABELS


def test_readable_name_variants():
    assert readable_name("dti_mid") == L["dti_mid"]
    assert readable_name("missingindicator_income") == L["income"] + " (kosong)"
    assert readable_name("conforming_loan_limit_NC") == L["conforming_loan_limit"] + " = NC"
    assert readable_name("has_coapplicant_1") == L["has_coapplicant"] + " = 1"


def test_reason_codes_top_positive():
    codes = reason_codes(np.array([0.3, -0.5, 0.1, 0.4]), ["dti_mid", "income", "loan_term", "ltv_calc"], k=2)
    assert codes == [L["ltv_calc"], L["dti_mid"]]


def test_reason_codes_skip_inactive_binary_features():
    names = ["missingindicator_income", "has_coapplicant_0", "has_coapplicant_1", "dti_mid"]
    values = np.array([0.5, 0.4, 0.3, 0.1])
    data = np.array([0.0, 0.0, 1.0, 0.7])
    assert reason_codes(values, names, k=3, data=data) == [L["has_coapplicant"] + " = 1", L["dti_mid"]]


def test_reason_codes_scaled_missing_indicator():
    names = ["missingindicator_income", "missingindicator_dti_mid", "dti_mid"]
    values = np.array([0.5, 0.4, 0.1])
    data = np.array([-0.12, 8.3, 0.7])  # indikator ikut StandardScaler: tidak kosong < 0, kosong > 0
    assert reason_codes(values, names, k=3, data=data) == [L["dti_mid"] + " (kosong)", L["dti_mid"]]


def test_reason_codes_fewer_positive():
    assert len(reason_codes(np.array([-1.0, 0.2]), ["income", "dti_mid"])) == 1


def test_fairness_dir_and_eod():
    g = pd.Series(["A"] * 200 + ["B"] * 200)
    y = np.zeros(400)
    pred = np.r_[np.zeros(200), np.ones(100), np.zeros(100)]
    t = fairness_table(y, pred, g).set_index("grup")
    assert t.loc["A", "catatan"] == "referensi"
    assert t.loc["B", "dir"] == 0.5 and t.loc["B", "eod"] == -0.5 and t.loc["B", "catatan"] == "DIR < 0,8"
    assert list(t.reset_index().columns) == [
        "grup", "n", "approval_rate_pred", "approval_rate_aktual", "tpr", "fpr", "dir", "dir_aktual", "eod", "catatan"]


def test_fairness_actual_dir_uses_same_reference():
    g = pd.Series(["A"] * 200 + ["B"] * 200)
    y = np.r_[np.zeros(200), np.ones(50), np.zeros(150)]
    pred = np.r_[np.zeros(200), np.ones(100), np.zeros(100)]
    t = fairness_table(y, pred, g).set_index("grup")
    assert t.loc["B", "approval_rate_aktual"] == 0.75 and t.loc["B", "dir_aktual"] == 0.75


def test_fairness_age_8888_is_unavailable():
    g = pd.Series(["35-44"] * 200 + ["8888"] * 200)
    t = fairness_table(np.zeros(400), np.zeros(400), g).set_index("grup")
    assert t.loc["8888", "catatan"] == "tidak tersedia"


def test_fairness_group_without_positives():
    g = pd.Series(["A"] * 150 + ["B"] * 150)
    y = np.r_[np.ones(10), np.zeros(140), np.zeros(150)]
    pred = np.r_[np.ones(10), np.zeros(290)]
    t = fairness_table(y, pred, g).set_index("grup")
    assert np.isnan(t.loc["B", "tpr"]) and t.loc["A", "tpr"] == 1.0


def test_fairness_not_available_never_reference():
    g = pd.Series(["Race Not Available"] * 300 + ["White"] * 300 + ["Asian"] * 50)
    y = np.zeros(650)
    pred = np.r_[np.zeros(300), np.ones(30), np.zeros(270), np.zeros(50)]
    t = fairness_table(y, pred, g).set_index("grup")
    assert t.loc["White", "catatan"] == "referensi"
    assert t.loc["Race Not Available", "catatan"] == "tidak tersedia"
    assert t.loc["Asian", "catatan"] == "n kecil"


def test_segment_metrics_single_class_segment():
    t = segment_metrics(np.array([0, 0, 1, 0]), np.array([0.1, 0.2, 0.9, 0.3]), 0.5, pd.Series(["a", "a", "b", "b"]))
    t = t.set_index("segmen")
    assert np.isnan(t.loc["a", "roc_auc"]) and t.loc["b", "n"] == 2
    assert list(t.reset_index().columns) == ["segmen", "n", "denial_rate", "roc_auc", "pr_auc", "recall", "cost_per_1000"]


def test_shap_explain_shapes_xgb(synthetic_xy):
    X, y = synthetic_xy
    m = tune("xgb", X, y, n_iter=1).pipeline
    e = shap_explain(m, X.head(20))
    assert e.values.shape[0] == 20 and e.values.ndim == 2
    assert len(e.feature_names) == e.values.shape[1]


def test_shap_explain_logreg_and_rf(synthetic_xy):
    X, y = synthetic_xy
    for name in ("logreg", "rf"):
        e = shap_explain(tune(name, X, y, n_iter=1).pipeline, X.head(15))
        assert e.values.shape[0] == 15 and e.values.ndim == 2, name


def test_unwrap_calibrated(synthetic_xy):
    X, y = synthetic_xy
    base = tune("logreg", X, y, n_iter=1).pipeline
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3, ensemble=False).fit(X, y)
    inner = unwrap_pipeline(cal)
    assert hasattr(inner, "named_steps") and "prep" in inner.named_steps
    assert shap_explain(cal, X.head(5)).values.shape[0] == 5
