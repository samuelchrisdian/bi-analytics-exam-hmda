import numpy as np
import pandas as pd
import pytest

from src import config

_BASE = {
    "action_taken": "1", "loan_purpose": "1", "lien_status": "1", "loan_type": "1",
    "occupancy_type": "1", "reverse_mortgage": "2", "open-end_line_of_credit": "2",
    "business_or_commercial_purpose": "2",
    "loan_amount": "305000", "income": "90", "property_value": "405000",
    "loan_to_value_ratio": "75.31", "debt_to_income_ratio": "36", "loan_term": "360",
    "tract_to_msa_income_percentage": "105.2", "ffiec_msa_md_median_family_income": "110300",
    "preapproval": "2", "submission_of_application": "1", "initially_payable_to_institution": "1",
    "aus-1": "1", "conforming_loan_limit": "C", "co-applicant_age": "45-54",
    "derived_race": "White", "derived_ethnicity": "Not Hispanic or Latino",
    "derived_sex": "Joint", "applicant_age": "35-44", "derived_msa-md": "19740",
    "county_code": "08031", "tract_minority_population_percent": "30.5",
}

# Baris 0-2 lolos semua filter scope; baris 3-11 masing-masing gagal di satu filter (urut SCOPE_FILTERS).
_ROWS = [
    {"action_taken": "1"},
    {"action_taken": "2", "income": "-5", "co-applicant_age": "9999", "debt_to_income_ratio": "<20%"},
    {"action_taken": "3", "property_value": "Exempt", "debt_to_income_ratio": "Exempt", "income": "0",
     "loan_to_value_ratio": "Exempt", "submission_of_application": "1111"},
    {"action_taken": "4"},
    {"action_taken": "6", "co-applicant_age": "9999"},
    {"loan_purpose": "31"},
    {"lien_status": "2", "property_value": "NA"},
    {"loan_type": "2"},
    {"occupancy_type": "2"},
    {"reverse_mortgage": "1111"},
    {"open-end_line_of_credit": "1"},
    {"business_or_commercial_purpose": "1"},
]
N_PASS_SCOPE = 3


@pytest.fixture
def raw_hmda() -> pd.DataFrame:
    rows = []
    for override in _ROWS:
        row = {c: "NA" for c in config.RAW_USECOLS}
        row.update(_BASE)
        row.update(override)
        rows.append(row)
    return pd.DataFrame(rows, dtype=object)


@pytest.fixture
def raw_csv(raw_hmda, tmp_path, monkeypatch):
    df = raw_hmda.copy()
    df["extra_unused_column"] = "x"
    path = tmp_path / "hmda_XX_2024.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    return path


@pytest.fixture
def synthetic_xy() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = 400
    dti = rng.uniform(10, 65, n)
    ltv = rng.uniform(40, 100, n)
    income = rng.lognormal(4.5, 0.5, n)
    loan = rng.uniform(1e5, 6e5, n)
    prop = loan / (ltv / 100)
    X = pd.DataFrame({
        "loan_amount": loan, "income": income, "property_value": prop,
        "ltv_calc": ltv, "dti_mid": dti, "loan_term": rng.choice([180.0, 360.0], n),
        "tract_to_msa_income_percentage": rng.uniform(50, 150, n),
        "ffiec_msa_md_median_family_income": rng.choice([98000.0, 110300.0, 125000.0], n),
        "loan_to_income": loan / (income * 1000), "property_to_income": prop / (income * 1000),
        "submission_of_application": rng.choice(["1", "2"], n),
        "conforming_loan_limit": rng.choice(["C", "NC"], n),
        "has_coapplicant": rng.choice([0, 1], n),
    })
    for col in ["income", "ltv_calc", "dti_mid", "submission_of_application", "loan_term"]:
        X.loc[rng.random(n) < 0.1, col] = np.nan
    logit = -4 + 0.06 * dti + 0.02 * ltv
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8"), name="y")
    return X, y


@pytest.fixture(scope="module")
def bundle():
    from src.train import tune

    X, y = synthetic_xy.__wrapped__()
    return {"model": tune("logreg", X, y, n_iter=1).pipeline, "threshold": 0.6, "t_low": 0.2, "background": X}
