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
# # Tahap 3 — Data Preprocessing & Preparation
# Semua analisis EDA memakai data **train 2024** saja. Data 2025 (OOT) hanya di-clean dan disimpan.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.data_prep import (
    clean, make_target, missingness_by_target, save_processed, split_frames, stratified_split,
)
from src.features import add_engineered
from src.reporting import save_figure, save_table, set_plot_style

set_plot_style()
LBL = config.FEATURE_LABELS
raw = {y: pd.read_parquet(config.DATA_INTERIM / f"hmda_filtered_{y}.parquet") for y in (config.YEAR_TRAIN, config.YEAR_OOT)}

# %% [markdown]
# ## 1. Target

# %%
y_year = {yr: make_target(df) for yr, df in raw.items()}
for yr, y in y_year.items():
    print(yr, "n =", len(y), "| ditolak =", int(y.sum()), f"({y.mean():.2%})")

# %% [markdown]
# ## 2. Audit leakage (data 2024)
# Kolom pasca-keputusan dibuang lebih dulu; sisa kolom diuji pola missing per kelas target.

# %%
c24 = clean(raw[config.YEAR_TRAIN])
y24 = y_year[config.YEAR_TRAIN]
audit_cols = [c for c in c24.columns if c not in config.LEAKAGE_COLS + ["action_taken"]]
leak = missingness_by_target(c24, y24, audit_cols)
leak = leak.reindex(leak["selisih_pp"].abs().sort_values(ascending=False).index)
save_table(leak, "t3_leakage_test", decimals=2)
leak.head(12)

# %% [markdown]
# Uji tambahan: kode kategori yang hampir menentukan target (artefak pelaporan/proses).

# %%
code_rows = []
for col in ["preapproval", "aus-1", "initially_payable_to_institution", "submission_of_application"]:
    g = pd.DataFrame({"kode": c24[col].fillna("kosong"), "y": y24}).groupby("kode")["y"].agg(["size", "mean"])
    for kode, r in g.iterrows():
        code_rows.append({"kolom": col, "kode": kode, "n": int(r["size"]), "pct_ditolak": r["mean"] * 100})
code_test = pd.DataFrame(code_rows)
save_table(code_test, "t3_leakage_code_test", decimals=2)
code_test

# %%
ltv_calc = c24["loan_amount"] / c24["property_value"] * 100
ltv_check = pd.DataFrame({
    "fitur": ["loan_to_value_ratio (laporan)", "ltv_calc (hitungan)"],
    "pct_na_y0": [c24.loc[y24 == 0, "loan_to_value_ratio"].isna().mean() * 100, ltv_calc[y24 == 0].isna().mean() * 100],
    "pct_na_y1": [c24.loc[y24 == 1, "loan_to_value_ratio"].isna().mean() * 100, ltv_calc[y24 == 1].isna().mean() * 100],
})
both = ltv_calc.notna() & c24["loan_to_value_ratio"].notna()
ltv_check["korelasi_dengan_ltv_laporan"] = [1.0, np.corrcoef(ltv_calc[both], pd.to_numeric(c24.loc[both, "loan_to_value_ratio"]))[0, 1]]
save_table(ltv_check, "t3_ltv_replacement", decimals=3)
ltv_check

# %% [markdown]
# ## 3. Cleaning + feature engineering (kedua tahun) dan pemeriksaan kualitas data

# %%
quality, frames = [], {}
for yr, df in raw.items():
    dup = int(df.duplicated().sum())
    inc_raw = pd.to_numeric(df["income"], errors="coerce")
    cl = add_engineered(clean(df))
    frames[yr] = split_frames(cl)
    quality.append({
        "tahun": yr, "n": len(df), "duplikat_persis": dup,
        "income_nonpositif": int((inc_raw <= 0).sum()),
        "ltv_calc_di_atas_150": int((cl["ltv_calc"] > 150).sum()),
        "loan_lebih_besar_2x_properti": int((cl["loan_amount"] > 2 * cl["property_value"]).sum()),
        "dti_exempt_atau_na": int(cl["dti_mid"].isna().sum()),
    })
quality = pd.DataFrame(quality)
save_table(quality, "t3_data_quality")
quality

# %% [markdown]
# ## 4. Split: 2024 → train/test stratified 80/20; 2025 → OOT utuh

# %%
X24, s24, m24 = frames[config.YEAR_TRAIN]
parts = stratified_split(X24, y24, s24, m24)
for name, (X, y, s, m) in parts.items():
    save_processed(name, X, y, s, m)
X25, s25, m25 = frames[config.YEAR_OOT]
save_processed("oot", X25, y_year[config.YEAR_OOT], s25, m25)
X_train, y_train = parts["train"][0], parts["train"][1]

balance = pd.DataFrame([
    {"split": n, "n": len(y), "n_ditolak": int(y.sum()), "n_disetujui": int((y == 0).sum()), "pct_ditolak": y.mean() * 100}
    for n, y in (("train 2024", parts["train"][1]), ("test 2024", parts["test"][1]), ("OOT 2025", y_year[config.YEAR_OOT]))
])
balance["rasio_negatif_per_positif"] = balance["n_disetujui"] / balance["n_ditolak"]
save_table(balance, "t3_class_balance", decimals=2)
balance

# %% [markdown]
# ## 5. Pemeriksaan set fitur final (evidence)

# %%
feats = list(X_train.columns)
assert set(feats).isdisjoint(config.LEAKAGE_COLS + config.PROTECTED_COLS + config.PROXY_COLS + config.LEAKAGE_AUDIT_DROPPED)
save_table(pd.DataFrame({
    "fitur": feats,
    "label": [LBL[f] for f in feats],
    "jenis": ["numerik" if f in config.NUMERIC_FEATURES else "kategorikal" for f in feats],
    "log_transform": [f in config.LOG_FEATURES for f in feats],
}), "t3_final_features")
print(len(feats), "fitur:", feats)

# %% [markdown]
# ## 6. Ringkasan missing value (train)

# %%
miss = (X_train.isna().mean() * 100).rename("pct_missing_train").to_frame()
miss["pct_missing_test"] = parts["test"][0].isna().mean() * 100
miss = miss.reset_index(names="fitur").sort_values("pct_missing_train", ascending=False)
save_table(miss, "t3_missing_summary", decimals=2)

fig, ax = plt.subplots(figsize=(7, 4))
m = miss[miss["pct_missing_train"] > 0]
ax.barh([LBL[f] for f in m["fitur"]], m["pct_missing_train"], color="#4C72B0")
ax.invert_yaxis()
ax.set_xlabel("Data kosong pada data latih (%)")
ax.set_ylabel("")
save_figure(fig, "f3_missing_summary")
miss

# %% [markdown]
# ## 7. EDA (data latih)

# %%
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar([str(y) for y in y_year], [y.mean() * 100 for y in y_year.values()], color=["#4C72B0", "#DD8452"])
for i, y in enumerate(y_year.values()):
    ax.text(i, y.mean() * 100 + 0.2, f"{y.mean() * 100:.2f}".replace(".", ","), ha="center")
ax.set_xlabel("Tahun data")
ax.set_ylabel("Aplikasi ditolak (%)")
save_figure(fig, "f3_target_per_tahun")


# %%
from src.reporting import fmt_id


def bin_label(b) -> str:
    if isinstance(b, pd.Interval):
        if np.isinf(b.right):
            return f"> {fmt_id(b.left, 0)}"
        return f"{fmt_id(b.left, 0)}–{fmt_id(b.right, 0)}"
    return str(b)


def denial_bar(series: pd.Series, y: pd.Series, xlabel: str, name: str) -> pd.DataFrame:
    g = pd.DataFrame({"bin": series, "y": y}).dropna().groupby("bin", observed=True, sort=True)["y"].agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar([bin_label(b) for b in g.index], g["mean"] * 100, color="#4C72B0")
    ax.axhline(y.mean() * 100, color="#C44E52", linestyle="--", linewidth=1, label="Rata-rata data latih")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tingkat penolakan (%)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_figure(fig, name)
    return g


dti_order = [str(int(v)) for v in sorted(X_train["dti_mid"].dropna().unique())] + ["Kosong"]
dti_bin = pd.Categorical(X_train["dti_mid"].map(lambda v: "Kosong" if pd.isna(v) else str(int(v))), categories=dti_order, ordered=True)
denial_bar(pd.Series(dti_bin, index=X_train.index), y_train, LBL["dti_mid"] + " — nilai tengah rentang", "f3_denial_rate_dti")
denial_bar(pd.cut(X_train["ltv_calc"], [0, 60, 70, 80, 85, 90, 95, 100, np.inf]), y_train, LBL["ltv_calc"], "f3_denial_rate_ltv")
denial_bar(pd.qcut(X_train["income"], 5, precision=0), y_train, LBL["income"] + " — kuintil", "f3_denial_rate_income")
denial_bar(pd.qcut(X_train["loan_amount"], 5, precision=0), y_train, LBL["loan_amount"] + " — kuintil", "f3_denial_rate_loan_amount")

# %%
lo, hi = X_train[config.LOG_FEATURES].quantile(0.01), X_train[config.LOG_FEATURES].quantile(0.99)
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for j, col in enumerate(config.LOG_FEATURES):
    v = X_train[col].dropna()
    axes[0, j].hist(v, bins=50, color="#4C72B0")
    axes[0, j].xaxis.set_major_locator(plt.MaxNLocator(3))
    axes[0, j].set_xlabel(LBL[col] + " (asli)")
    axes[1, j].hist(np.log1p(v.clip(lo[col], hi[col])), bins=50, color="#55A868")
    axes[1, j].set_xlabel(LBL[col] + " (capping 1–99 + log)")
    axes[0, j].set_ylabel("Frekuensi")
    axes[1, j].set_ylabel("Frekuensi")
fig.tight_layout()
save_figure(fig, "f3_dist_before_after_log")

# %%
num = [c for c in config.NUMERIC_FEATURES]
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for ax, col in zip(axes.ravel(), num):
    ax.boxplot(X_train[col].dropna(), vert=True, widths=0.5, flierprops={"markersize": 2})
    ax.set_xticks([])
    ax.set_ylabel(LBL[col], fontsize=8)
fig.tight_layout()
save_figure(fig, "f3_boxplot_numerik")

outlier = pd.DataFrame({
    "fitur": num,
    "p1": [X_train[c].quantile(0.01) for c in num],
    "median": [X_train[c].median() for c in num],
    "p99": [X_train[c].quantile(0.99) for c in num],
    "maks": [X_train[c].max() for c in num],
    "pct_di_atas_p99": [(X_train[c] > X_train[c].quantile(0.99)).mean() * 100 for c in num],
})
save_table(outlier, "t3_outlier_percentiles", decimals=2)
outlier

# %%
corr = X_train[num].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 6.5))
labels = [LBL[c] for c in num]
sns.heatmap(corr, annot=corr.map(lambda v: fmt_id(v, 2)), fmt="", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 7}, cbar_kws={"label": "Korelasi Spearman"})
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
plt.setp(ax.get_yticklabels(), fontsize=7)
save_figure(fig, "f3_corr_heatmap")
corr.round(2)

# %% [markdown]
# ## 8. Catatan keputusan preprocessing

# %%
D = [
    ("Target", "action_taken 1/2/3; kode lain tidak relevan", "y = 1 bila 3 (ditolak), y = 0 bila 1 atau 2", "Hanya kode 1–3 adalah keputusan kredit atas aplikasi penuh"),
    ("Kolom pasca-keputusan (denial_reason-1..4, interest_rate, rate_spread, hoepa_status, total_loan_costs, total_points_and_fees, origination_charges, discount_points, lender_credits, purchaser_type)", "Baru terisi setelah keputusan; banyak kosong untuk aplikasi ditolak", "Dibuang", "Leakage langsung: tidak tersedia saat aplikasi masuk"),
    ("loan_to_value_ratio (LTV laporan)", "Kosong 52,04% pada aplikasi ditolak vs 2,52% disetujui; tingkat tolak 73% bila kosong (t3_leakage_test, t3_ltv_replacement)", "Dibuang dan diganti ltv_calc = loan_amount / property_value × 100", "Pola kosong terjadi karena lender tidak menghitung LTV untuk aplikasi yang ditolak lebih awal, sehingga 'kosong' membocorkan keputusan. Informasi LTV sendiri penting bagi underwriting, dan dua komponennya (jumlah pinjaman yang diajukan dan nilai properti) sudah diketahui saat aplikasi masuk. ltv_calc berkorelasi tinggi dengan LTV laporan dan selisih pola kosongnya di bawah ambang 20 poin persen"),
    ("preapproval", "Kode 1 (pra-persetujuan diminta): 2.298 aplikasi, 0% ditolak (t3_leakage_code_test)", "Dibuang", "Artefak aturan pelaporan HMDA: pra-persetujuan yang ditolak dilaporkan sebagai action_taken 7 (di luar scope), sehingga kode 1 secara mekanis tidak pernah berlabel ditolak. Model akan mempelajari aturan pelaporan, bukan risiko pemohon"),
    ("aus-1", "Kode 6 'tidak berlaku/tidak memakai AUS': 9.056 aplikasi, tingkat tolak ±46% vs ±5% untuk kode AUS lain", "Dibuang", "Kode 6 umumnya muncul karena aplikasi dihentikan/ditolak sebelum dievaluasi AUS, sehingga mencerminkan jalannya proses (pasca-keputusan) dan bukan karakteristik pemohon. Hasil AUS sendiri tidak tersedia di data publik"),
    ("initially_payable_to_institution", "Kode 3 'tidak berlaku': 681 aplikasi, tingkat tolak ±81%", "Dibuang", "Kode 'tidak berlaku' terisi ketika tidak ada kewajiban kredit yang terbentuk (aplikasi tidak dicairkan), sehingga membocorkan hasil keputusan"),
    ("Atribut protected (derived_race/ethnicity/sex, applicant_age, kolom ras/etnis/jenis kelamin rinci)", "Atribut dilindungi fair lending", "Dikeluarkan dari fitur; derived_race, derived_ethnicity, derived_sex, applicant_age disimpan di sensitive_*.parquet untuk uji fairness", "Mencegah diskriminasi langsung"),
    ("tract_minority_population_percent", "Persentase penduduk minoritas di tract", "Dikeluarkan dari fitur", "Potensi proxy ras (diskriminasi tidak langsung)"),
    ("lei, census_tract, county_code, state_code, activity_year", "Identitas lender/lokasi dengan kardinalitas tinggi atau konstan", "Dibuang dari fitur", "Model dipakai satu bank (lei tidak relevan); lokasi rinci rawan proxy dan overfitting"),
    ("derived_msa-md", "Kode wilayah MSA", "Disimpan di meta_*.parquet hanya untuk analisis segmen", "Bukan prediktor; berguna untuk uji robustness per wilayah"),
    ("String 'NA', 'Exempt', '' pada semua kolom", "Kode kosong/pengecualian HMDA", "Diubah menjadi NaN", "Diperlakukan sebagai missing lalu diimputasi di pipeline"),
    ("Kode 1111 pada kolom kategorikal", "Exempt", "Diubah menjadi NaN", "Tidak memuat informasi kategori"),
    ("income ≤ 0", "Nilai tidak wajar (lihat t3_data_quality)", "Diubah menjadi NaN", "Pendapatan non-positif tidak masuk akal untuk KPR; mencegah rasio tak hingga"),
    ("debt_to_income_ratio", "Rentang string (<20%, 20%-<30%, 30%-<36%, 36–49, 50%-60%, >60%)", "Dipetakan ke nilai tengah: 15, 25, 33, 36–49, 55, 65 (dti_mid)", "Mempertahankan urutan ordinal; rentang terbuka diberi nilai tengah wajar"),
    ("co-applicant_age", "9999 = tidak ada co-applicant", "Fitur has_coapplicant (0/1); kolom usia sendiri tidak dipakai (protected)", "Keberadaan co-applicant relevan bagi kapasitas bayar tanpa memakai usia"),
    ("Fitur turunan", "-", "loan_to_income = loan_amount/(income×1.000); property_to_income = property_value/(income×1.000); ltv_calc", "Rasio beban kredit terhadap pendapatan dan agunan adalah logika inti underwriting"),
    ("Pola kosong property_value, income, DTI (uji lift)", "Jarang kosong (≤ 3,1%) tetapi tingkat tolak saat kosong jauh di atas rata-rata (lift ≥ 2; lihat kolom lift di t3_leakage_test)", "Nilai tetap dipakai, tetapi status 'kosong' TIDAK dijadikan fitur (tanpa missing indicator); aplikasi berdata tidak lengkap ditangani aturan bisnis data_tidak_lengkap (Tahap 8)", "Kosong terjadi karena lender berhenti memproses aplikasi yang ditolak lebih awal (mekanisme sama dengan LTV laporan) — informasi pasca-keputusan; pada aplikasi baru nilai ini selalu diisi pemohon"),
    ("Missing value numerik", "Kosong ≤ 3,1% per fitur (t3_missing_summary)", "Imputasi median di dalam Pipeline (fit di train), tanpa missing indicator", "Median robust terhadap skew; indikator kosong dihapus karena membocorkan keputusan (baris di atas)"),
    ("Missing value kategorikal", "Sangat sedikit", "Imputasi modus + OneHotEncoder(handle_unknown='ignore') di Pipeline", "Aman terhadap kategori baru pada test/OOT"),
    ("Outlier & skew (loan_amount, income, property_value)", "Distribusi sangat menceng kanan (f3_dist_before_after_log, t3_outlier_percentiles)", "Capping persentil 1–99 (QuantileCapper, fit di train) lalu log1p", "Mengurangi pengaruh nilai ekstrem tanpa membuang baris"),
    ("Outlier fitur numerik lain", "Ekor panjang pada rasio (f3_boxplot_numerik)", "Capping persentil 1–99 tanpa log", "Konsisten antar-fitur dan tetap di dalam pipeline"),
    ("Skala", "Rentang fitur sangat berbeda", "StandardScaler di pipeline", "Wajib untuk Logistic Regression; tidak berpengaruh pada tree"),
    ("Duplikat persis", "Lihat t3_data_quality", "Dipertahankan", "Tanpa ID aplikasi, baris identik bisa merupakan aplikasi berbeda (nilai dibulatkan/dibinning)"),
    ("Split", "Data 2024", "Stratified 80/20 train/test, random_state 42; 2025 utuh sebagai OOT", "Test untuk evaluasi akhir, OOT untuk generalisasi lintas waktu"),
    ("Imbalance", "±11,75% ditolak (t3_class_balance)", "Diputuskan di Tahap 5 (class weighting utama, SMOTE pembanding)", "Penanganan harus di dalam CV agar tidak bocor"),
    ("Teknik tidak dipakai: penghapusan baris missing", "-", "Tidak dipakai", "Akan membuang aplikasi ditolak secara tidak proporsional (bias)"),
    ("Teknik tidak dipakai: winsorizing/penghapusan outlier di luar pipeline", "-", "Tidak dipakai", "Batas harus dipelajari dari train saja; dilakukan oleh QuantileCapper"),
    ("Teknik tidak dipakai: PCA/reduksi dimensi", "Hanya 13 fitur", "Tidak dipakai", "Jumlah fitur kecil dan interpretabilitas lebih penting"),
    ("Teknik tidak dipakai: target encoding", "Kardinalitas kategori rendah", "Tidak dipakai", "One-hot cukup dan tanpa risiko leakage target"),
]
decisions = pd.DataFrame(D, columns=["kolom_aspek", "temuan", "keputusan", "alasan"])
decisions.insert(0, "no", range(1, len(decisions) + 1))
save_table(decisions, "t3_decisions")
decisions
