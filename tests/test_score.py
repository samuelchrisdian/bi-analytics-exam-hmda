import numpy as np
import pandas as pd
import pytest

from src.score import (TIERS, apply_business_rules, assign_tier, choose_low_threshold, contributions,
                       score_application, score_batch)

VALID_APP = {
    "loan_amount": "305000", "income": "120", "property_value": "405000",
    "debt_to_income_ratio": "30%-<36%", "loan_term": "360",
    "tract_to_msa_income_percentage": "105.2", "ffiec_msa_md_median_family_income": "110300",
    "submission_of_application": "1", "conforming_loan_limit": "C", "co-applicant_age": "45-54",
}


def test_tiers_constant():
    assert TIERS == ["Rendah", "Sedang", "Tinggi"]


def test_assign_tier():
    assert list(assign_tier(np.array([0.05, 0.3, 0.7, 0.6]), 0.1, 0.6)) == ["Rendah", "Sedang", "Tinggi", "Tinggi"]


def test_choose_low_threshold():
    y = np.r_[np.zeros(80), np.ones(20)]
    p = np.r_[np.linspace(0, 0.5, 80), np.linspace(0.4, 1, 20)]
    t = choose_low_threshold(y, p, t_high=0.6)
    assert t <= 0.6 and y[p < t].mean() <= 0.5 * y.mean()
    assert y[p < round(t + 0.01, 2)].mean() > 0.5 * y.mean() or round(t + 0.01, 2) > 0.6


def test_business_rules_override():
    X = pd.DataFrame({"income": [50, np.nan, 80], "ltv_calc": [98, 80, 70],
                      "dti_mid": [30, 30, 55], "property_value": [1e5, 1e5, 1e5]})
    tiers, hit = apply_business_rules(np.array(["Rendah", "Rendah", "Sedang"]), X)
    assert list(tiers) == ["Tinggi", "Sedang", "Tinggi"]
    assert hit.tolist() == ["melewati_batas_kebijakan", "data_tidak_lengkap", "melewati_batas_kebijakan"]


def test_business_rules_ltv_97_is_within_policy():
    X = pd.DataFrame({"income": [50], "ltv_calc": [96.5], "dti_mid": [30], "property_value": [1e5]})
    tiers, hit = apply_business_rules(np.array(["Rendah"]), X)
    assert list(tiers) == ["Rendah"] and hit.tolist() == [""]


def test_choose_low_threshold_respects_rate_parameter():
    y = np.r_[np.zeros(80), np.ones(20)]
    p = np.r_[np.linspace(0, 0.5, 80), np.linspace(0.4, 1, 20)]
    strict, loose = choose_low_threshold(y, p, 0.6, max_rel_rate=0.0), choose_low_threshold(y, p, 0.6, max_rel_rate=0.5)
    assert strict <= loose and y[p < strict].sum() == 0


def test_business_rules_keep_higher_tier():
    X = pd.DataFrame({"income": [np.nan], "ltv_calc": [80], "dti_mid": [30], "property_value": [1e5]})
    tiers, _ = apply_business_rules(np.array(["Tinggi"]), X)
    assert list(tiers) == ["Tinggi"]


def test_score_application_contract(bundle):
    r = score_application(VALID_APP, bundle)
    assert set(r) == {"proba", "tier", "reason_codes", "aturan"}
    assert 0 <= r["proba"] <= 1 and r["tier"] in TIERS and len(r["reason_codes"]) <= 3


def test_score_application_reason_codes_for_risky_app(bundle):
    risky = {**VALID_APP, "debt_to_income_ratio": ">60%"}
    r = score_application(risky, bundle)
    assert "Rasio utang terhadap pendapatan (%)" in r["reason_codes"]
    assert r["tier"] == "Tinggi" and r["aturan"] == "melewati_batas_kebijakan"


def test_score_application_missing_fields_forces_medium(bundle):
    r = score_application({"loan_amount": "250000", "income": "Exempt"}, bundle)
    assert r["tier"] in {"Sedang", "Tinggi"} and r["aturan"] != ""


def test_score_application_numeric_payload_matches_string_payload(bundle):
    numeric = {**VALID_APP, "loan_amount": 305000, "income": 120, "property_value": 405000, "loan_term": 360,
               "debt_to_income_ratio": 44, "submission_of_application": 1.0, "co-applicant_age": 9999}
    string = {**VALID_APP, "debt_to_income_ratio": "44", "co-applicant_age": "9999"}
    a, b = score_application(numeric, bundle), score_application(string, bundle)
    assert a == b


def test_score_batch_contract(bundle):
    raw = pd.DataFrame([VALID_APP, {**VALID_APP, "debt_to_income_ratio": ">60%"}])
    out = score_batch(raw, bundle)
    assert list(out.columns) == ["proba", "tier_model", "tier", "aturan"]
    assert out.index.equals(raw.index)
    assert out.loc[1, "tier"] == "Tinggi" and out.loc[1, "aturan"] == "melewati_batas_kebijakan"


def test_score_batch_matches_single(bundle):
    r = score_application(VALID_APP, bundle)
    b = score_batch(pd.DataFrame([VALID_APP]), bundle).iloc[0]
    assert b["proba"] == pytest.approx(r["proba"]) and b["tier"] == r["tier"]


def test_score_batch_incomplete_row_is_medium(bundle):
    out = score_batch(pd.DataFrame([{**VALID_APP, "income": "0", "property_value": "NA"}]), bundle)
    assert out.iloc[0]["tier"] in {"Sedang", "Tinggi"} and out.iloc[0]["aturan"] != ""


def test_contributions_nonzero_single_row(bundle):
    c = contributions(VALID_APP, bundle, top=5)
    assert list(c.columns) == ["fitur", "shap"] and len(c) <= 5
    assert (c["shap"].abs() > 0).any()
    assert c["shap"].abs().is_monotonic_decreasing


def test_contributions_skip_inactive_onehot(bundle):
    from src import config

    labels = set(contributions(VALID_APP, bundle, top=50)["fitur"])
    submission = config.FEATURE_LABELS["submission_of_application"]
    assert f"{submission} = 2" not in labels
    assert f'{config.FEATURE_LABELS["has_coapplicant"]} = 0' not in labels
