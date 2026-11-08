import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_prep import (
    apply_scope_filters, clean, load_processed, load_raw, make_target, missingness_by_target, parse_dti,
    profile, save_processed, split_frames, stratified_split,
)
from tests.conftest import N_PASS_SCOPE


def test_load_raw_handles_hmda_strings(raw_csv):
    df = load_raw(2024, state="XX")
    assert "open-end_line_of_credit" in df.columns and "aus-1" in df.columns
    assert "extra_unused_column" not in df.columns
    assert (df.dtypes == object).all()
    assert (df["property_value"] == "Exempt").any()


def test_scope_filters_funnel(raw_hmda):
    out, funnel = apply_scope_filters(raw_hmda)
    assert funnel.iloc[0]["langkah"] == "Data mentah" and funnel.iloc[0]["n_baris"] == 12
    assert funnel["n_baris"].is_monotonic_decreasing
    assert len(funnel) == len(config.SCOPE_FILTERS) + 1
    assert funnel["n_baris"].tolist() == [12, 10, 9, 8, 7, 6, 5, 4, 3]
    assert funnel["n_dibuang"].tolist()[1:] == [2, 1, 1, 1, 1, 1, 1, 1]
    assert len(out) == N_PASS_SCOPE
    assert set(out["action_taken"].astype(int)) <= {1, 2, 3}
    assert (out["reverse_mortgage"].astype(int) == 2).all()


def test_make_target():
    y = make_target(pd.DataFrame({"action_taken": ["1", "2", "3"]}))
    assert y.tolist() == [0, 0, 1] and y.name == "y" and str(y.dtype) == "int8"


def test_make_target_rejects_other_codes():
    with pytest.raises(ValueError):
        make_target(pd.DataFrame({"action_taken": ["1", "4"]}))


def test_profile_counts_exempt_as_missing(raw_hmda):
    p = profile(raw_hmda).set_index("kolom")
    assert list(p.columns) == ["tipe", "n_missing", "pct_missing", "n_unique", "contoh"]
    assert p.loc["property_value", "n_missing"] == 2
    assert p.loc["loan_amount", "n_missing"] == 0


@pytest.mark.parametrize("raw,exp", [("<20%", 15), ("20%-<30%", 25), ("30%-<36%", 33),
                                     ("42", 42), ("50%-60%", 55), (">60%", 65)])
def test_parse_dti(raw, exp):
    assert parse_dti(pd.Series([raw]))[0] == exp


def test_parse_dti_missing():
    assert parse_dti(pd.Series(["Exempt", "NA", None])).isna().all()


def test_clean_converts_exempt_and_na(raw_hmda):
    c = clean(raw_hmda)
    assert c["property_value"].dtype.kind == "f" and c["property_value"].isna().sum() == 2
    assert c["dti_mid"].isna().sum() == 1


def test_clean_nonpositive_income_to_nan(raw_hmda):
    c = clean(raw_hmda)
    assert (c["income"].dropna() > 0).all() and c["income"].isna().sum() == 2


def test_clean_exempt_code_1111_in_categoricals(raw_hmda):
    assert clean(raw_hmda)["submission_of_application"].isna().sum() == 1


def test_clean_has_coapplicant(raw_hmda):
    c = clean(raw_hmda)
    assert set(c["has_coapplicant"]) == {0, 1} and c["has_coapplicant"].sum() == 10


def test_missingness_flags_leaky_column():
    df = pd.DataFrame({"a": [1, 1, np.nan, np.nan], "b": [1, np.nan, 1, np.nan]})
    t = missingness_by_target(df, pd.Series([0, 0, 1, 1]), ["a", "b"]).set_index("kolom")
    assert t.loc["a", "flag_leakage"] and not t.loc["b", "flag_leakage"]
    assert t.loc["a", "selisih_pp"] == 100


def test_split_frames_excludes_protected(raw_hmda):
    X, s, m = split_frames(clean(raw_hmda))
    assert set(X.columns) <= set(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
    assert {"dti_mid", "has_coapplicant", "income"} <= set(X.columns)
    assert set(X.columns).isdisjoint(config.PROTECTED_COLS)
    assert X.index.equals(s.index) and X.index.equals(m.index)
    assert list(s.columns) == config.FAIRNESS_COLS and list(m.columns) == config.META_COLS


def test_stratified_split_and_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    n = 100
    X = pd.DataFrame({"loan_amount": np.arange(n, dtype=float)})
    y = pd.Series([1] * 20 + [0] * 80, name="y", dtype="int8")
    s = pd.DataFrame({"derived_race": ["White"] * n})
    m = pd.DataFrame({"county_code": ["08031"] * n})
    parts = stratified_split(X, y, s, m)
    Xt, yt, st, mt = parts["test"]
    assert len(Xt) == 20 and yt.mean() == 0.2 and Xt.index.equals(st.index)
    save_processed("test", Xt, yt, st, mt)
    X2, y2, s2, m2 = load_processed("test")
    pd.testing.assert_frame_equal(X2, Xt)
    pd.testing.assert_series_equal(y2, yt)


def test_clean_numeric_typed_codes():
    raw = pd.DataFrame([{"debt_to_income_ratio": 44, "co-applicant_age": 9999, "submission_of_application": 1.0,
                         "income": 90, "loan_amount": 300000, "property_value": 400000}])
    c = clean(raw)
    assert c["dti_mid"][0] == 44 and c["has_coapplicant"][0] == 0 and c["submission_of_application"][0] == "1"


def test_clean_missing_coapplicant_is_unknown_not_present():
    c = clean(pd.DataFrame({"co-applicant_age": [np.nan, "35-44", "9999"]}))
    assert np.isnan(c["has_coapplicant"][0]) and c["has_coapplicant"][1] == 1 and c["has_coapplicant"][2] == 0


def test_missingness_flags_rare_high_lift_blank():
    y = pd.Series(np.r_[np.ones(200), np.zeros(1800)])
    col = np.ones(2000)
    col[:30] = np.nan          # 30 kosong di kelas ditolak
    col[200:210] = np.nan      # 10 kosong di kelas disetujui
    t = missingness_by_target(pd.DataFrame({"pv": col}), y, ["pv"]).set_index("kolom")
    assert t.loc["pv", "selisih_pp"] < 20
    assert t.loc["pv", "pct_ditolak_bila_kosong"] == 75 and t.loc["pv", "lift"] == 7.5
    assert t.loc["pv", "flag_leakage"]


def test_missingness_lift_ignores_tiny_blank_counts():
    y = pd.Series(np.r_[np.ones(100), np.zeros(900)])
    col = np.ones(1000)
    col[:5] = np.nan
    t = missingness_by_target(pd.DataFrame({"c": col}), y, ["c"]).set_index("kolom")
    assert not t.loc["c", "flag_leakage"]
