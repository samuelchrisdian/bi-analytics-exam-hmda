# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tahap 2 — Sourcing of Raw Data
# Sumber: HMDA Data Browser (FFIEC/CFPB), Colorado 2024 (*One Year*) dan 2025 (*Snapshot*).
# File unduhan sudah difilter di Data Browser: `action_taken` 1–3 dan `loan_purpose` 1; filter scope lain diterapkan di sini.

# %%
import pandas as pd

from src import config
from src.data_prep import apply_scope_filters, load_raw, make_target, profile
from src.reporting import save_table

pd.set_option("display.max_columns", 50)
YEARS = [config.YEAR_TRAIN, config.YEAR_OOT]

# %% [markdown]
# ## 1. Load dan filter scope

# %%
full_cols = pd.read_csv(config.DATA_RAW / f"hmda_{config.STATE}_{config.YEAR_TRAIN}.csv", nrows=0).columns
print("Jumlah kolom pada file unduhan:", len(full_cols))

filtered, funnels, targets = {}, [], []
for year in YEARS:
    raw = load_raw(year)
    print(year, "raw shape:", raw.shape)
    out, funnel = apply_scope_filters(raw)
    funnel.insert(0, "tahun", year)
    funnels.append(funnel)
    y = make_target(out)
    targets.append({"tahun": year, "n_aplikasi": len(y), "n_ditolak": int(y.sum()),
                    "n_disetujui": int((y == 0).sum()), "pct_ditolak": y.mean() * 100})
    out.to_parquet(config.DATA_INTERIM / f"hmda_filtered_{year}.parquet")
    filtered[year] = out
    print(year, "setelah filter scope:", out.shape)

funnel_all = pd.concat(funnels, ignore_index=True)
funnel_all.loc[funnel_all["langkah"] == "Data mentah", "filter"] = (
    "unduhan Data Browser (sudah: action_taken 1–3, loan_purpose 1)"
)
save_table(funnel_all, "t2_filter_funnel", decimals=2)
funnel_all

# %%
target_dist = pd.DataFrame(targets)
save_table(target_dist, "t2_target_distribution", decimals=2)
target_dist

# %% [markdown]
# ## 2. Profiling (data 2024 setelah filter scope)

# %%
prof = profile(filtered[config.YEAR_TRAIN])
save_table(prof, "t2_profile", decimals=2)
prof.sort_values("pct_missing", ascending=False)

# %% [markdown]
# ## 3. Data dictionary kolom terpilih
# Deskripsi diterjemahkan dari *Public HMDA – LAR Data Fields* (FFIEC).

# %%
DESKRIPSI = {
    "action_taken": "Tindakan atas aplikasi (1 disetujui & dicairkan, 2 disetujui tidak diambil, 3 ditolak, 4–8 lainnya)",
    "loan_purpose": "Tujuan kredit (1 pembelian rumah, 2 renovasi, 31 refinancing, 32 cash-out refinancing, 4 lainnya)",
    "lien_status": "Status hak tanggungan (1 first lien, 2 subordinate lien)",
    "loan_type": "Jenis kredit (1 konvensional, 2 FHA, 3 VA, 4 RHS/FSA)",
    "occupancy_type": "Jenis hunian (1 tempat tinggal utama, 2 rumah kedua, 3 investasi)",
    "reverse_mortgage": "Reverse mortgage (1 ya, 2 bukan, 1111 exempt)",
    "open-end_line_of_credit": "Kredit open-end (1 ya, 2 bukan, 1111 exempt)",
    "business_or_commercial_purpose": "Terutama untuk tujuan bisnis/komersial (1 ya, 2 bukan, 1111 exempt)",
    "loan_amount": "Jumlah kredit yang diajukan (USD, dibulatkan ke titik tengah interval 10.000)",
    "income": "Pendapatan kotor tahunan yang diandalkan dalam keputusan (ribu USD)",
    "property_value": "Nilai properti yang diandalkan dalam keputusan (USD, dibulatkan ke titik tengah interval 10.000)",
    "loan_to_value_ratio": "Rasio total utang beragunan properti terhadap nilai properti (%)",
    "debt_to_income_ratio": "Rasio utang bulanan terhadap pendapatan bulanan (%, dalam rentang: <20%, 20%-<30%, 30%-<36%, 36–49, 50%-60%, >60%)",
    "loan_term": "Tenor kredit (bulan)",
    "tract_to_msa_income_percentage": "Median pendapatan keluarga tract terhadap MSA/MD (%)",
    "ffiec_msa_md_median_family_income": "Median pendapatan keluarga MSA/MD versi FFIEC (USD)",
    "preapproval": "Permintaan pra-persetujuan (1 diminta, 2 tidak diminta)",
    "submission_of_application": "Aplikasi diajukan langsung ke lender (1 ya, 2 tidak, 3 tidak berlaku, 1111 exempt)",
    "initially_payable_to_institution": "Kewajiban awalnya dibayarkan ke lender (1 ya, 2 tidak, 3 tidak berlaku, 1111 exempt)",
    "aus-1": "Automated underwriting system yang dipakai (1 DU, 2 LP, 3 TOTAL, 4 GUS, 5 lain, 6 tidak berlaku, 7 internal, 1111 exempt)",
    "conforming_loan_limit": "Status terhadap batas conforming loan GSE (C, NC, U, NA)",
    "co-applicant_age": "Usia co-applicant (rentang; 8888 tidak tersedia, 9999 tidak ada co-applicant)",
    "applicant_age": "Usia pemohon (rentang; 8888 tidak tersedia)",
    "applicant_age_above_62": "Pemohon berusia 62 tahun atau lebih",
    "co-applicant_age_above_62": "Co-applicant berusia 62 tahun atau lebih",
    "derived_race": "Kategori ras agregat pemohon dan co-applicant",
    "derived_ethnicity": "Kategori etnis agregat pemohon dan co-applicant",
    "derived_sex": "Kategori jenis kelamin agregat pemohon dan co-applicant",
    "derived_msa-md": "Kode MSA/MD 5 digit",
    "county_code": "Kode FIPS negara bagian-county",
    "census_tract": "Nomor census tract 11 digit",
    "lei": "Legal Entity Identifier lembaga keuangan",
    "state_code": "Kode negara bagian dua huruf",
    "activity_year": "Tahun kalender pelaporan",
    "tract_minority_population_percent": "Persentase penduduk minoritas di tract",
    "interest_rate": "Suku bunga kredit",
    "rate_spread": "Selisih APR terhadap APOR",
    "hoepa_status": "Status high-cost mortgage (HOEPA)",
    "total_loan_costs": "Total biaya kredit (USD)",
    "total_points_and_fees": "Total points dan fees (USD)",
    "origination_charges": "Biaya originasi yang dibayar peminjam (USD)",
    "discount_points": "Discount points untuk menurunkan suku bunga (USD)",
    "lender_credits": "Lender credits (USD)",
    "purchaser_type": "Jenis pembeli kredit di pasar sekunder",
}
for i in range(1, 5):
    DESKRIPSI[f"denial_reason-{i}"] = f"Alasan penolakan ke-{i} (1 DTI, 2 riwayat kerja, 3 riwayat kredit, 4 agunan, ...)"


def peran(col: str) -> str:
    if col == "action_taken":
        return "target"
    if col in {"debt_to_income_ratio", "co-applicant_age"}:
        return "prediktor (diturunkan)" if col == "debt_to_income_ratio" else "protected (dipakai untuk has_coapplicant)"
    if col in config.LEAKAGE_COLS:
        return "leakage"
    if col in config.LEAKAGE_AUDIT_DROPPED:
        return "leakage (audit Tahap 3)"
    if col in config.PROTECTED_COLS:
        return "protected"
    if col in config.PROXY_COLS:
        return "proxy"
    if col in config.META_COLS:
        return "meta (segmentasi)"
    if col in config.ID_COLS:
        return "ID"
    if col in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
        return "prediktor"
    if col in {c for c, _ in config.SCOPE_FILTERS}:
        return "filter scope"
    return "dibuang"


tipe = prof.set_index("kolom")["tipe"]
dd = pd.DataFrame({
    "nama": config.RAW_USECOLS,
    "deskripsi": [DESKRIPSI.get(c, "Atribut ras/etnis/jenis kelamin pemohon atau co-applicant") for c in config.RAW_USECOLS],
    "tipe": [tipe.get(c, "-") for c in config.RAW_USECOLS],
    "peran": [peran(c) for c in config.RAW_USECOLS],
})
dd = pd.concat([dd, pd.DataFrame([
    {"nama": "dti_mid", "deskripsi": "Nilai tengah rentang debt_to_income_ratio (%)", "tipe": "numerik", "peran": "prediktor (turunan)"},
    {"nama": "ltv_calc", "deskripsi": "loan_amount / property_value × 100 (pengganti LTV laporan)", "tipe": "numerik", "peran": "prediktor (turunan)"},
    {"nama": "loan_to_income", "deskripsi": "loan_amount / (income × 1.000)", "tipe": "numerik", "peran": "prediktor (turunan)"},
    {"nama": "property_to_income", "deskripsi": "property_value / (income × 1.000)", "tipe": "numerik", "peran": "prediktor (turunan)"},
    {"nama": "has_coapplicant", "deskripsi": "1 bila ada co-applicant (co-applicant_age ≠ 9999)", "tipe": "kategorikal", "peran": "prediktor (turunan)"},
])], ignore_index=True)
save_table(dd, "t2_data_dictionary")
dd["peran"].value_counts()

# %% [markdown]
# ## 4. Rantai dokumentasi data

# %%
n_train, n_oot = len(filtered[config.YEAR_TRAIN]), len(filtered[config.YEAR_OOT])
raw_n = funnel_all[funnel_all["langkah"] == "Data mentah"].set_index("tahun")["n_baris"]
predictors = ", ".join(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
doc = pd.DataFrame([
    ("Source", "HMDA Data Browser — https://ffiec.cfpb.gov/data-browser/ (dokumentasi kolom: https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields)"),
    ("Owner/Provider", "Federal Financial Institutions Examination Council (FFIEC) dan Consumer Financial Protection Bureau (CFPB), Amerika Serikat"),
    ("Dataset Description", "Loan/Application Register (LAR) publik: satu baris = satu aplikasi/kredit KPR yang dilaporkan lender sesuai Home Mortgage Disclosure Act"),
    ("Period", f"{config.YEAR_TRAIN} (dataset One Year) sebagai data latih/uji; {config.YEAR_OOT} (dataset Snapshot) sebagai data out-of-time"),
    ("Number of Records", f"Unduhan: {raw_n[config.YEAR_TRAIN]:,} ({config.YEAR_TRAIN}) dan {raw_n[config.YEAR_OOT]:,} ({config.YEAR_OOT}); setelah filter scope: {n_train:,} dan {n_oot:,}".replace(",", ".")),
    ("Number of Variables", f"{len(full_cols)} kolom pada file unduhan; {len(config.RAW_USECOLS)} kolom dibaca; {len(config.NUMERIC_FEATURES) + len(config.CATEGORICAL_FEATURES)} fitur kandidat model"),
    ("Target Variable", "y = 1 bila action_taken = 3 (ditolak); y = 0 bila action_taken = 1 atau 2 (disetujui)"),
    ("Key Predictors", predictors),
    ("Unit of Analysis", "Satu baris = satu aplikasi KPR"),
    ("Limitations", "Tanpa credit score/riwayat pembayaran; konteks AS (Colorado); nilai disamarkan untuk privasi (pembulatan, rentang DTI/usia); target = keputusan lender bukan default; cross-sectional per tahun; data 2025 berupa Snapshot yang dapat direvisi"),
], columns=["elemen", "keterangan"])
save_table(doc, "t2_data_documentation")
doc

# %% [markdown]
# ## 5. Contoh raw data (evidence)

# %%
sample_cols = ["activity_year", "derived_msa-md", "action_taken", "loan_type", "loan_purpose", "loan_amount",
               "income", "property_value", "loan_to_value_ratio", "debt_to_income_ratio", "loan_term", "aus-1"]
sample = filtered[config.YEAR_TRAIN].sample(10, random_state=config.RANDOM_STATE)[sample_cols]
save_table(sample, "t2_raw_sample")
sample
