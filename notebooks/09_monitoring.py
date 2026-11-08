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
# # Tahap 9 — Monitoring, Feedback & Continuous Improvement
# Simulasi drift nyata: data latih 2024 sebagai acuan, OOT 2025 sebagai "data produksi".

# %%
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data_prep import load_processed
from src.monitor import psi, psi_status, psi_table
from src.reporting import fmt_id, save_figure, save_table, set_plot_style
from src.score import TIERS, assign_tier

set_plot_style()
LBL = config.FEATURE_LABELS
meta = json.loads((config.MODELS / "final_threshold.json").read_text())
model = joblib.load(config.MODELS / "final_model.joblib")
X_train, y_train, _, _ = load_processed("train")
X_test, y_test, _, _ = load_processed("test")
X_oot, y_oot, _, _ = load_processed("oot")

# %% [markdown]
# ## 1. PSI fitur: train 2024 vs test 2024 (kontrol) dan vs OOT 2025

# %%
psi_oot = psi_table(X_train, X_oot)
psi_ctrl = psi_table(X_train, X_test).set_index("fitur")["psi"]
psi_oot["psi_kontrol_train_vs_test"] = psi_oot["fitur"].map(psi_ctrl)
psi_oot["label"] = psi_oot["fitur"].map(LBL)
save_table(psi_oot, "t9_psi_features", decimals=4)

fig, ax = plt.subplots(figsize=(7, 4.5))
pt = psi_oot.iloc[::-1]
XMAX = 0.3
ax.barh(pt["label"], pt["psi"].clip(upper=XMAX), color="#4C72B0")
for i, v in enumerate(pt["psi"]):
    if v > XMAX:
        ax.text(XMAX * 0.98, i, f"PSI {fmt_id(v, 2)} →", va="center", ha="right", color="white", fontsize=8)
ax.set_xlim(0, XMAX)
ax.axvline(config.PSI_WARN, color="#DD8452", linestyle="--", linewidth=1, label="Ambang waspada (0,10)")
ax.axvline(config.PSI_ACT, color="#C44E52", linestyle="--", linewidth=1, label="Ambang tindakan (0,25)")
ax.set_xlabel("PSI data latih 2024 vs OOT 2025")
ax.legend(fontsize=8)
save_figure(fig, "f9_psi")
psi_oot

# %% [markdown]
# ## 2. PSI skor model dan distribusi tier

# %%
p_tr, p_te, p_oot = (model.predict_proba(X)[:, 1] for X in (X_train, X_test, X_oot))
tiers = {n: pd.Series(assign_tier(p, meta["t_low"], meta["threshold"])) for n, p in (("train", p_tr), ("test", p_te), ("oot", p_oot))}
score_rows = [
    {"objek": "Skor probabilitas model", "pembanding": "train vs test 2024 (kontrol)", "psi": psi(pd.Series(p_tr), pd.Series(p_te))},
    {"objek": "Skor probabilitas model", "pembanding": "train 2024 vs OOT 2025", "psi": psi(pd.Series(p_tr), pd.Series(p_oot))},
    {"objek": "Distribusi risk tier", "pembanding": "train vs test 2024 (kontrol)", "psi": psi(tiers["train"], tiers["test"], categorical=True)},
    {"objek": "Distribusi risk tier", "pembanding": "train 2024 vs OOT 2025", "psi": psi(tiers["train"], tiers["oot"], categorical=True)},
]
score_psi = pd.DataFrame(score_rows)
score_psi["status"] = score_psi["psi"].map(psi_status)
for n in ("train", "oot"):
    vc = tiers[n].value_counts(normalize=True).reindex(TIERS, fill_value=0) * 100
    for t in TIERS:
        score_psi.loc[len(score_psi)] = {"objek": f"Porsi tier {t} (%)", "pembanding": n, "psi": vc[t], "status": "-"}
save_table(score_psi, "t9_psi_score", decimals=4)
score_psi

# %% [markdown]
# ## 3. Indikasi model drift: test 2024 vs OOT 2025 (dari Tahap 7)

# %%
oot_perf = pd.read_csv(config.TABLES / "t7_oot_performance.csv")
oot_perf[["data", "proporsi_ditolak", "roc_auc", "pr_auc", "recall", "precision", "cost_per_1000"]]

# %% [markdown]
# ## 4. Analisis error pada test (threshold final)

# %%
imp = pd.read_csv(config.TABLES / "t7_importance_comparison.csv").sort_values("porsi_shap", ascending=False)
top = [f for f in imp["fitur"] if f in config.NUMERIC_FEATURES][:5]
pred = (p_te >= meta["threshold"]).astype(int)
group = np.select([(y_test == 1) & (pred == 1), (y_test == 1) & (pred == 0), (y_test == 0) & (pred == 1), (y_test == 0) & (pred == 0)],
                  ["TP (ditolak, terdeteksi)", "FN (ditolak, lolos)", "FP (disetujui, ditandai)", "TN (disetujui, aman)"], default="-")
err = X_test[top].groupby(group).median()
err.insert(0, "n", pd.Series(group).value_counts())
err = err.rename(columns=LBL).reset_index(names="kelompok")
save_table(err, "t9_error_analysis", decimals=2)
err
