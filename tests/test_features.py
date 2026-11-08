import numpy as np
import pandas as pd

from src.features import QuantileCapper, add_engineered, build_preprocessor, get_feature_names


def test_engineered_ratio_no_inf():
    d = add_engineered(pd.DataFrame({"loan_amount": [200000.0, 100000.0], "income": [100.0, np.nan],
                                     "property_value": [250000.0, 1.0]}))
    assert d["loan_to_income"].tolist()[0] == 2.0 and np.isnan(d["loan_to_income"][1])
    assert d["property_to_income"][0] == 2.5
    assert not np.isinf(d.select_dtypes("number")).any().any()


def test_engineered_ltv_calc():
    d = add_engineered(pd.DataFrame({"loan_amount": [300000.0, 1.0], "income": [100.0, 1.0],
                                     "property_value": [400000.0, 0.0]}))
    assert d["ltv_calc"][0] == 75.0 and np.isnan(d["ltv_calc"][1])


def test_engineered_zero_income_is_nan_not_inf():
    d = add_engineered(pd.DataFrame({"loan_amount": [1.0], "income": [0.0], "property_value": [1.0]}))
    assert np.isnan(d["loan_to_income"][0]) and np.isnan(d["property_to_income"][0])


def test_capper_fit_on_train_only():
    c = QuantileCapper(0.0, 0.5).fit(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, np.nan]}))
    out = c.transform(pd.DataFrame({"a": [100.0, np.nan]}))
    assert out[0, 0] == 2.5 and np.isnan(out[1, 0])


def test_preprocessor_no_nan_output(synthetic_xy):
    X, _ = synthetic_xy
    Z = build_preprocessor().fit_transform(X)
    assert not np.isnan(Z).any()


def test_preprocessor_feature_names_match_width(synthetic_xy):
    X, _ = synthetic_xy
    p = build_preprocessor().fit(X)
    names = get_feature_names(p)
    assert p.transform(X).shape[1] == len(names)
    assert "dti_mid" in names


def test_preprocessor_has_no_missing_indicator(synthetic_xy):
    # Pola kosong HMDA membocorkan keputusan (lift tinggi), jadi tidak boleh jadi fitur.
    X, _ = synthetic_xy
    names = get_feature_names(build_preprocessor().fit(X))
    assert not any(n.startswith("missingindicator_") for n in names)


def test_preprocessor_ignores_unseen_category(synthetic_xy):
    X, _ = synthetic_xy
    p = build_preprocessor().fit(X)
    X2 = X.head(3).copy()
    X2["conforming_loan_limit"] = "ZZ"
    assert p.transform(X2).shape[1] == len(get_feature_names(p))


def test_preprocessor_all_nan_column_at_transform(synthetic_xy):
    X, _ = synthetic_xy
    p = build_preprocessor().fit(X)
    X2 = X.head(5).copy()
    X2["loan_term"] = np.nan
    assert not np.isnan(p.transform(X2)).any()
