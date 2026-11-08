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
# # Tahap 7 — Validation
# Model final sudah dikunci di Tahap 6. Notebook ini pertama kali membaca data OOT 2025.

# %%
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src import config
from src.data_prep import load_processed
from src.evaluate import bootstrap_ci, metrics_at
from src.explain import fairness_table, readable_name, reason_codes, segment_metrics, shap_explain
from src.reporting import fmt_id, localize_figure, save_figure, save_table, set_plot_style

set_plot_style()
LBL = config.FEATURE_LABELS
meta = json.loads((config.MODELS / "final_threshold.json").read_text())
model = joblib.load(config.MODELS / "final_model.joblib")
THR = meta["threshold"]
print(meta)

X_train, y_train, s_train, m_train = load_processed("train")
X_test, y_test, s_test, m_test = load_processed("test")
X_oot, y_oot, s_oot, m_oot = load_processed("oot")
p_train = model.predict_proba(X_train)[:, 1]
p_test = model.predict_proba(X_test)[:, 1]
p_oot = model.predict_proba(X_oot)[:, 1]

# %% [markdown]
# ## 1. Overfitting / underfitting: train vs CV vs test

# %%
cv = pd.read_csv(config.TABLES / "t5_cv_results.csv")
cv = cv[cv["model"] == meta["model"]].set_index("metrik")["mean"]
m_tr, m_te = metrics_at(y_train, p_train, THR), metrics_at(y_test, p_test, THR)
gap = pd.DataFrame([
    {"metrik": "roc_auc", "train": m_tr["roc_auc"], "cv_5_fold": cv["roc_auc"], "test": m_te["roc_auc"]},
    {"metrik": "pr_auc", "train": m_tr["pr_auc"], "cv_5_fold": cv["average_precision"], "test": m_te["pr_auc"]},
    {"metrik": "recall", "train": m_tr["recall"], "cv_5_fold": np.nan, "test": m_te["recall"]},
    {"metrik": "precision", "train": m_tr["precision"], "cv_5_fold": np.nan, "test": m_te["precision"]},
    {"metrik": "cost_per_1000", "train": m_tr["cost_per_1000"], "cv_5_fold": np.nan, "test": m_te["cost_per_1000"]},
])
gap["selisih_train_test"] = gap["train"] - gap["test"]
gap["flag_gap_gt_0_05"] = (gap["metrik"].isin(["roc_auc", "pr_auc"])) & (gap["selisih_train_test"].abs() > 0.05)
save_table(gap, "t7_train_test_gap", decimals=3)
gap

# %% [markdown]
# Catatan: CV recall/precision Tahap 5 dihitung pada threshold 0,5, sehingga tidak sebanding dengan threshold final; hanya AUC yang dibandingkan.

# %% [markdown]
# ## 2. Stabilitas: bootstrap 1.000× pada test

# %%
boot = bootstrap_ci(y_test, p_test, THR)
save_table(boot, "t7_bootstrap_ci", decimals=3)
boot

# %% [markdown]
# ## 3. Robustness lintas waktu: test 2024 vs OOT 2025 (threshold tetap)

# %%
oot_rows = []
for name, y, p in (("Test 2024", y_test, p_test), ("OOT 2025", y_oot, p_oot)):
    oot_rows.append({"data": name, "n": len(y), "proporsi_ditolak": y.mean(), **metrics_at(y, p, THR)})
oot_perf = pd.DataFrame(oot_rows)
save_table(oot_perf, "t7_oot_performance", decimals=3)
oot_perf

# %% [markdown]
# ### Sensitivitas: hanya aplikasi berdata lengkap
# Bila model masih memanfaatkan pola kosong, kinerjanya pada data lengkap akan jauh lebih rendah daripada pada semua data.

# %%
CORE = ["property_value", "income", "dti_mid"]
sens_rows = []
for name, X, y, p in (("Test 2024", X_test, y_test, p_test), ("OOT 2025", X_oot, y_oot, p_oot)):
    complete = X[CORE].notna().all(axis=1).to_numpy()
    for subset, m in (("semua aplikasi", np.ones(len(y), bool)), ("data lengkap", complete), ("ada yang kosong", ~complete)):
        sens_rows.append({"data": name, "subset": subset, "n": int(m.sum()), "proporsi_ditolak": y[m].mean(),
                          **{k: v for k, v in metrics_at(y[m], p[m], THR).items() if k in ("roc_auc", "pr_auc", "recall", "precision", "cost_per_1000")}})
sens = pd.DataFrame(sens_rows)
save_table(sens, "t7_complete_data_sensitivity", decimals=3)
sens

# %% [markdown]
# ## 4. Robustness segmen (test 2024)

# %%
def quartile_segments(col: str) -> pd.Series:
    edges = np.quantile(X_train[col].dropna(), [0, 0.25, 0.5, 0.75, 1])
    edges[0], edges[-1] = -np.inf, np.inf
    labels = [f"Q{i}" for i in range(1, 5)]
    return pd.cut(X_test[col], edges, labels=labels).astype(object)


top_msa = m_train["derived_msa-md"].value_counts().head(5).index
msa = m_test["derived_msa-md"].where(m_test["derived_msa-md"].isin(top_msa))
seg_tables = []
for dim, seg in (("Kuartil jumlah pinjaman", quartile_segments("loan_amount")),
                 ("Kuartil pendapatan", quartile_segments("income")),
                 ("MSA terbesar (kode)", msa)):
    t = segment_metrics(y_test, p_test, THR, seg)
    t.insert(0, "dimensi", dim)
    seg_tables.append(t.sort_values("segmen"))
segments = pd.concat(seg_tables, ignore_index=True)
save_table(segments, "t7_segment_performance", decimals=3)
segments

# %% [markdown]
# ## 5. Pemeriksaan leakage

# %%
feats = list(X_train.columns)
assert set(feats).isdisjoint(config.LEAKAGE_COLS + config.PROTECTED_COLS + config.PROXY_COLS + config.LEAKAGE_AUDIT_DROPPED)
print("Fitur final:", feats)
if m_te["roc_auc"] > 0.95:
    print(f"RED FLAG: ROC-AUC test {m_te['roc_auc']:.3f} > 0,95 — periksa leakage")
else:
    print(f"ROC-AUC test {m_te['roc_auc']:.3f} ≤ 0,95 — tidak ada red flag AUC")

# %% [markdown]
# ## 6. Explainability — SHAP (sampel 5.000 baris test)

# %%
sample_idx = X_test.sample(min(config.SHAP_SAMPLE, len(X_test)), random_state=config.RANDOM_STATE).index
Xs, ys = X_test.loc[sample_idx], y_test.loc[sample_idx]
background = X_train.sample(1000, random_state=config.RANDOM_STATE)
expl = shap_explain(model, Xs, background=background)
names = list(expl.feature_names)

display_data = np.array(expl.data, dtype=float)
for j, n in enumerate(names):
    if n in Xs.columns and n in config.NUMERIC_FEATURES:
        display_data[:, j] = Xs[n].to_numpy()
ex = shap.Explanation(values=expl.values, base_values=expl.base_values, data=display_data,
                      feature_names=[readable_name(n) for n in names])

share = np.abs(expl.values).mean(axis=0)
share = pd.Series(share / share.sum(), index=names).sort_values(ascending=False)
print("Kontribusi |SHAP| terbesar:", share.head(5).round(3).to_dict())
if share.iloc[0] > 0.5:
    print("PERINGATAN: satu fitur > 50% total |SHAP| — periksa kemungkinan leakage")


def shap_axes_id(fig):
    for ax in fig.axes:
        if "SHAP" in ax.get_xlabel():
            ax.set_xlabel("Nilai SHAP (dampak terhadap skor risiko penolakan)")
        if ax.get_ylabel() == "Feature value":
            ax.set_ylabel("Nilai fitur")
            ax.set_yticklabels(["Rendah", "Tinggi"])
    return fig


shap.plots.beeswarm(ex, max_display=15, show=False)
save_figure(localize_figure(shap_axes_id(plt.gcf())), "f7_shap_summary")
shap.plots.bar(ex, max_display=15, show=False)
fig = plt.gcf()
fig.axes[0].set_xlabel("Rata-rata |nilai SHAP|")
save_figure(localize_figure(fig), "f7_shap_bar")

# %%
raw_top = [n for n in share.index if n in config.NUMERIC_FEATURES][:3]
for i, n in enumerate(raw_top, start=1):
    shap.plots.scatter(ex[:, names.index(n)], show=False)
    fig = plt.gcf()
    fig.axes[0].set_xlabel(LBL[n])
    fig.axes[0].set_ylabel("Nilai SHAP")
    save_figure(fig, f"f7_shap_dependence_{i}")
raw_top

# %%
ps = model.predict_proba(Xs)[:, 1]
examples = {"disetujui": int(np.argmin(ps)), "borderline": int(np.argmin(np.abs(ps - THR))), "ditolak": int(np.argmax(ps))}
rc_rows = []
for label, i in examples.items():
    shap.plots.waterfall(ex[i], max_display=10, show=False)
    save_figure(localize_figure(plt.gcf()), f"f7_shap_waterfall_{label}")
    rc_rows.append({"contoh": label, "probabilitas": ps[i], "aktual": "ditolak" if ys.iloc[i] == 1 else "disetujui",
                    "reason_codes": "; ".join(reason_codes(expl.values[i], names, data=np.asarray(expl.data[i]))) or "-"})
rc = pd.DataFrame(rc_rows)
save_table(rc, "t7_reason_codes_examples", decimals=3)
rc

# %% [markdown]
# ## 7. Permutation importance (cross-check)

# %%
perm = permutation_importance(model, Xs, ys, scoring="average_precision", n_repeats=5,
                              random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS)
perm_tbl = pd.DataFrame({"fitur": Xs.columns, "penurunan_pr_auc_mean": perm.importances_mean,
                         "penurunan_pr_auc_std": perm.importances_std}).sort_values("penurunan_pr_auc_mean", ascending=False)

raw_share = {}
for n, v in share.items():
    base = n.removeprefix("missingindicator_")
    base = next((c for c in config.CATEGORICAL_FEATURES if base.startswith(c + "_")), base)
    raw_share[base] = raw_share.get(base, 0) + v
raw_share = pd.Series(raw_share)
perm_tbl["porsi_shap"] = perm_tbl["fitur"].map(raw_share)
perm_tbl["peringkat_permutation"] = range(1, len(perm_tbl) + 1)
perm_tbl["peringkat_shap"] = perm_tbl["porsi_shap"].rank(ascending=False, method="min").astype(int)
perm_tbl["label"] = perm_tbl["fitur"].map(LBL)
save_table(perm_tbl, "t7_importance_comparison", decimals=4)

fig, ax = plt.subplots(figsize=(7, 4.5))
pt = perm_tbl.iloc[::-1]
ax.barh(pt["label"], pt["penurunan_pr_auc_mean"], xerr=pt["penurunan_pr_auc_std"], color="#4C72B0")
ax.set_xlabel("Penurunan PR-AUC saat fitur diacak")
save_figure(fig, "f7_perm_importance")
perm_tbl

# %% [markdown]
# ## 8. Business sense check

# %%
EXPECT = {
    "dti_mid": ("+", "DTI tinggi = kapasitas bayar rendah → risiko naik"),
    "ltv_calc": ("+", "LTV tinggi = uang muka kecil, agunan tipis → risiko naik"),
    "income": ("-", "Pendapatan tinggi → kapasitas bayar naik → risiko turun"),
    "loan_to_income": ("+", "Pinjaman besar relatif pendapatan → risiko naik"),
    "property_to_income": ("+", "Properti mahal relatif pendapatan → beban naik"),
    "loan_amount": ("?", "Efek bergantung pendapatan/agunan; tidak ada ekspektasi tunggal"),
    "property_value": ("?", "Efek bergantung pinjaman/pendapatan"),
    "loan_term": ("?", "Tenor standar 30 tahun dominan; tenor tidak standar bisa berisiko"),
    "tract_to_msa_income_percentage": ("-", "Wilayah berpendapatan lebih tinggi → risiko agunan/ekonomi lebih rendah"),
    "ffiec_msa_md_median_family_income": ("?", "Konteks pasar wilayah; tidak ada ekspektasi tunggal"),
}
bs_rows = []
for n in share.index:
    if n not in EXPECT or n not in Xs.columns:
        continue
    x = Xs[n].to_numpy()
    v = expl.values[:, names.index(n)]
    ok = ~np.isnan(x)
    rho = pd.Series(x[ok]).corr(pd.Series(v[ok]), method="spearman")
    arah = "+" if rho > 0.1 else ("-" if rho < -0.1 else "datar/non-monoton")
    exp_sign, reason = EXPECT[n]
    bs_rows.append({"fitur": LBL[n], "porsi_shap": share[n], "arah_pengaruh": arah, "korelasi_spearman_nilai_vs_shap": rho,
                    "ekspektasi": exp_sign, "logika_underwriting": reason,
                    "sesuai": "-" if exp_sign == "?" else ("ya" if arah == exp_sign else "tidak")})
for col in ("has_coapplicant", "conforming_loan_limit", "submission_of_application"):
    cols = [n for n in names if n.startswith(col + "_")]
    effects = {}
    for c in cols:
        j = names.index(c)
        active = np.asarray(expl.data[:, j]) > 0
        if active.any():
            effects[readable_name(c)] = float(expl.values[active, j].mean())
    bs_rows.append({"fitur": LBL[col], "porsi_shap": raw_share.get(col, np.nan), "arah_pengaruh": "; ".join(f"{k}: {fmt_id(v, 3)}" for k, v in effects.items()),
                    "korelasi_spearman_nilai_vs_shap": np.nan,
                    "ekspektasi": "-" if col == "has_coapplicant" else "?",
                    "logika_underwriting": {"has_coapplicant": "Ada co-applicant menambah pendapatan rumah tangga → risiko turun",
                                            "conforming_loan_limit": "Kredit non-conforming (jumbo) punya syarat lebih ketat",
                                            "submission_of_application": "Jalur pengajuan (langsung/broker) dapat memengaruhi kelengkapan berkas"}[col],
                    "sesuai": "lihat arah per kategori"})
business = pd.DataFrame(bs_rows)
save_table(business, "t7_business_sense", decimals=3)
business

# %% [markdown]
# ## 9. Fairness (atribut protected hanya untuk evaluasi)

# %%
pred_test = (p_test >= THR).astype(int)
fair_tables, na_rows = [], []
for col in config.FAIRNESS_COLS:
    t = fairness_table(y_test, pred_test, s_test[col])
    t.insert(0, "atribut", col)
    fair_tables.append(t)
    na = s_test[col].astype(str).str.contains("Not Available|8888|Free Form", regex=True) | s_test[col].isna()
    na_rows.append({"atribut": col, "pct_tidak_tersedia": na.mean() * 100})
fair = pd.concat(fair_tables, ignore_index=True)
save_table(fair, "t7_fairness", decimals=3)
save_table(pd.DataFrame(na_rows), "t7_fairness_missing", decimals=2)
fair
