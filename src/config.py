from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
DIAGRAMS = ROOT / "reports" / "diagrams"
LAPORAN = ROOT / "laporan"

# data/ tidak ikut repositori; salinan baru harus bisa langsung menjalankan pipeline.
for _dir in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS, FIGURES, TABLES):
    _dir.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
STATE = "CO"
YEAR_TRAIN = 2024
YEAR_OOT = 2025

SCOPE_FILTERS: list[tuple[str, set[int]]] = [
    ("action_taken", {1, 2, 3}),
    ("loan_purpose", {1}),
    ("lien_status", {1}),
    ("loan_type", {1}),
    ("occupancy_type", {1}),
    ("reverse_mortgage", {2}),
    ("open-end_line_of_credit", {2}),
    ("business_or_commercial_purpose", {2}),
]

# Kolom yang baru terisi setelah keputusan lender.
LEAKAGE_COLS = [
    "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
    "interest_rate", "rate_spread", "hoepa_status", "total_loan_costs",
    "total_points_and_fees", "origination_charges", "discount_points",
    "lender_credits", "purchaser_type",
]

PROTECTED_COLS = [
    "derived_race", "derived_ethnicity", "derived_sex",
    "applicant_age", "co-applicant_age",
    "applicant_age_above_62", "co-applicant_age_above_62",
    "applicant_sex", "co-applicant_sex",
    "applicant_sex_observed", "co-applicant_sex_observed",
    "applicant_race_observed", "co-applicant_race_observed",
    "applicant_ethnicity_observed", "co-applicant_ethnicity_observed",
] + [
    f"{who}_{attr}-{i}"
    for who in ("applicant", "co-applicant")
    for attr in ("race", "ethnicity")
    for i in range(1, 6)
]

FAIRNESS_COLS = ["derived_race", "derived_ethnicity", "derived_sex", "applicant_age"]
PROXY_COLS = ["tract_minority_population_percent"]
ID_COLS = ["lei", "census_tract", "county_code", "state_code", "activity_year", "derived_msa-md"]
META_COLS = ["derived_msa-md", "county_code"]

# Hasil audit leakage Tahap 3 (keputusan user, lihat reports/tables/t3_decisions.md).
LEAKAGE_AUDIT_DROPPED = ["loan_to_value_ratio", "preapproval", "aus-1", "initially_payable_to_institution"]

NUMERIC_FEATURES = [
    "loan_amount", "income", "property_value", "ltv_calc", "dti_mid",
    "loan_term", "tract_to_msa_income_percentage", "ffiec_msa_md_median_family_income",
    "loan_to_income", "property_to_income",
]
LOG_FEATURES = ["loan_amount", "income", "property_value"]
CATEGORICAL_FEATURES = ["submission_of_application", "conforming_loan_limit", "has_coapplicant"]

_DERIVED = {"dti_mid", "ltv_calc", "loan_to_income", "property_to_income", "has_coapplicant"}
RAW_USECOLS = sorted(
    {c for c, _ in SCOPE_FILTERS}
    | {"debt_to_income_ratio"}
    | (set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - _DERIVED)
    | set(LEAKAGE_COLS + LEAKAGE_AUDIT_DROPPED + PROTECTED_COLS + PROXY_COLS + ID_COLS + META_COLS)
)

FEATURE_LABELS = {
    "loan_amount": "Jumlah pinjaman (USD)",
    "income": "Pendapatan (ribu USD)",
    "property_value": "Nilai properti (USD)",
    "ltv_calc": "Rasio pinjaman terhadap nilai properti (%)",
    "dti_mid": "Rasio utang terhadap pendapatan (%)",
    "loan_term": "Tenor pinjaman (bulan)",
    "tract_to_msa_income_percentage": "Pendapatan tract terhadap MSA (%)",
    "ffiec_msa_md_median_family_income": "Median pendapatan keluarga MSA (USD)",
    "loan_to_income": "Rasio pinjaman terhadap pendapatan",
    "property_to_income": "Rasio nilai properti terhadap pendapatan",
    "submission_of_application": "Pengajuan langsung ke lender",
    "conforming_loan_limit": "Batas conforming loan",
    "has_coapplicant": "Ada co-applicant",
}

COST_FN = 5
COST_FP = 1
COST_RATIOS = [2, 5, 10]
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)

CV_FOLDS = 5
N_ITER = 25
TUNE_SUBSAMPLE = 100_000
TUNE_SUBSAMPLE_TRIGGER = 200_000
N_JOBS = 4

PSI_BINS = 10
PSI_WARN = 0.10
PSI_ACT = 0.25

TEST_SIZE = 0.20
SHAP_SAMPLE = 5_000
N_BOOT = 1_000
