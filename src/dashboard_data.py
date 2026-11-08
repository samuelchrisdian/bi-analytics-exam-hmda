import math

import pandas as pd

from src import config
from src.data_prep import DTI_MAP, MISSING_STRINGS, code_str

NUMERIC_FORM_FIELDS = [
    "loan_amount", "income", "property_value", "loan_term",
    "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income",
]
CATEGORICAL_FORM_FIELDS = ["submission_of_application", "conforming_loan_limit"]
FORM_FIELDS = NUMERIC_FORM_FIELDS + ["debt_to_income_ratio"] + CATEGORICAL_FORM_FIELDS + ["has_coapplicant"]

# Penanda HMDA: 9999 = tanpa co-applicant, 8888 = ada tetapi usia tidak tersedia.
NO_COAPPLICANT, HAS_COAPPLICANT = "9999", "8888"

METRIC_TABLES = {
    "model_comparison": ("t6_model_comparison", "make nb-06_model_selection"),
    "oot_performance": ("t7_oot_performance", "make nb-07_validation"),
    "fairness": ("t7_fairness", "make nb-07_validation"),
    "tier_distribution": ("t8_tier_distribution_oot", "make nb-08_deployment"),
    "psi_score": ("t9_psi_score", "make nb-09_monitoring"),
}


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() in MISSING_STRINGS)


def _to_float(v) -> float | None:
    if _is_missing(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def dti_to_hmda(dti: float | None) -> str:
    """Angka DTI → bin HMDA; `parse_dti` hanya mengenali bin, angka lepas di luar 36–49 akan terbaca kosong."""
    if dti is None:
        return "NA"
    if dti < 20:
        return "<20%"
    if dti < 30:
        return "20%-<30%"
    if dti < 36:
        return "30%-<36%"
    if dti < 50:
        return str(int(dti))
    if dti <= 60:
        return "50%-60%"
    return ">60%"


def form_to_app(form: dict) -> dict:
    """Nilai form dashboard → field HMDA mentah (string) untuk `score_application`/`score_batch`."""
    app = {f: "NA" if _is_missing(form.get(f)) else str(form[f]) for f in NUMERIC_FORM_FIELDS + CATEGORICAL_FORM_FIELDS}
    app["debt_to_income_ratio"] = dti_to_hmda(_to_float(form.get("debt_to_income_ratio")))
    app["co-applicant_age"] = HAS_COAPPLICANT if form.get("has_coapplicant") else NO_COAPPLICANT
    return app


def app_to_form(app: dict) -> dict:
    """Field HMDA mentah → nilai form; kode kosong/Exempt menjadi None."""
    form = {f: _to_float(app.get(f)) for f in NUMERIC_FORM_FIELDS}
    dti = app.get("debt_to_income_ratio")
    form["debt_to_income_ratio"] = None if _is_missing(dti) else DTI_MAP.get(code_str(dti))
    for f in CATEGORICAL_FORM_FIELDS:
        form[f] = None if _is_missing(app.get(f)) else code_str(app[f])
    age = app.get("co-applicant_age")
    form["has_coapplicant"] = not _is_missing(age) and code_str(age) != NO_COAPPLICANT
    return form


def sample_oot(n: int, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """n aplikasi acak dari data OOT terfilter, plus label keputusan lender aktual."""
    raw = pd.read_parquet(config.DATA_INTERIM / f"hmda_filtered_{config.YEAR_OOT}.parquet")
    out = raw.sample(n=min(n, len(raw)), random_state=seed)
    out["aktual"] = (out["action_taken"].map(code_str) == "3").map({True: "ditolak", False: "disetujui"})
    return out


def load_metrics() -> dict[str, pd.DataFrame]:
    out = {}
    for key, (name, make_cmd) in METRIC_TABLES.items():
        path = config.TABLES / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Tabel {name} tidak ditemukan di {config.TABLES}; jalankan `{make_cmd}`.")
        out[key] = pd.read_csv(path)
    return out
