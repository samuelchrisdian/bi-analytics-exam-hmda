import pandas as pd
import pytest

from src import config
from src.dashboard_data import FORM_FIELDS, app_to_form, dti_to_hmda, form_to_app, load_metrics, sample_oot
from src.score import score_batch

FORM = {
    "loan_amount": 305000, "income": 120, "property_value": 405000, "debt_to_income_ratio": 33,
    "loan_term": 360, "tract_to_msa_income_percentage": 105.2, "ffiec_msa_md_median_family_income": 110300,
    "submission_of_application": "1", "conforming_loan_limit": "C", "has_coapplicant": False,
}


def test_form_fields_cover_form():
    assert set(FORM_FIELDS) == set(FORM)


def test_form_to_app_coapplicant_flag():
    assert form_to_app({**FORM, "has_coapplicant": True})["co-applicant_age"] == "8888"
    assert form_to_app({**FORM, "has_coapplicant": False})["co-applicant_age"] == "9999"


@pytest.mark.parametrize("dti,expected", [(5, "<20%"), (25, "20%-<30%"), (33, "30%-<36%"), (42.7, "42"),
                                          (50, "50%-60%"), (75, ">60%"), (None, "NA")])
def test_dti_to_hmda_bins(dti, expected):
    assert dti_to_hmda(dti) == expected


def test_form_to_app_high_dti_triggers_rule(bundle):
    out = score_batch(pd.DataFrame([form_to_app({**FORM, "debt_to_income_ratio": 75})]), bundle)
    assert out.iloc[0]["tier"] == "Tinggi"


def test_form_to_app_missing_income_is_na():
    assert form_to_app({**FORM, "income": None})["income"] == "NA"


def test_app_to_form_handles_na():
    f = app_to_form({"income": "NA", "property_value": "Exempt", "debt_to_income_ratio": "30%-<36%",
                     "co-applicant_age": "9999"})
    assert f["income"] is None and f["property_value"] is None
    assert f["debt_to_income_ratio"] == 33.0 and f["has_coapplicant"] is False


def test_app_to_form_roundtrip():
    assert app_to_form(form_to_app(FORM)) == {**FORM, "loan_amount": 305000.0, "income": 120.0,
                                              "property_value": 405000.0, "debt_to_income_ratio": 33.0,
                                              "loan_term": 360.0, "ffiec_msa_md_median_family_income": 110300.0}


@pytest.mark.skipif(not (config.DATA_INTERIM / f"hmda_filtered_{config.YEAR_OOT}.parquet").exists(),
                    reason="data interim belum dibuat")
def test_sample_oot_is_reproducible():
    a, b = sample_oot(5), sample_oot(5)
    assert a.index.equals(b.index) and len(a) == 5
    assert set(a["aktual"]) <= {"ditolak", "disetujui"}


def test_load_metrics_keys():
    m = load_metrics()
    assert set(m) == {"model_comparison", "oot_performance", "tier_distribution", "psi_score", "fairness"}


def test_load_metrics_missing_file_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TABLES", tmp_path)
    with pytest.raises(FileNotFoundError, match="t6_model_comparison"):
        load_metrics()
