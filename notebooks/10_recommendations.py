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
# # Tahap 10 — Kuantifikasi dampak bisnis
# Biaya harapan per 1.000 aplikasi (C_FN = 5, C_FP = 1): model final vs skenario tanpa model dan baseline Dummy.

# %%
import json

import joblib
import pandas as pd

from src import config
from src.data_prep import load_processed
from src.evaluate import baseline_costs, expected_cost
from src.reporting import fmt_id, save_table

meta = json.loads((config.MODELS / "final_threshold.json").read_text())
model = joblib.load(config.MODELS / "final_model.joblib")
dummy = joblib.load(config.MODELS / "dummy.joblib")
dummy_thr = pd.read_csv(config.TABLES / "t6_threshold.csv").set_index("kode").loc["dummy", "threshold_optimal"]

rows = []
for split, label in (("test", "Test 2024"), ("oot", "OOT 2025")):
    X, y, _, _ = load_processed(split)
    base = baseline_costs(y)
    manual = base["semua_review_manual"]
    scen = {
        "Semua aplikasi review manual (tanpa model)": manual,
        "Semua aplikasi fast-track (tanpa model)": base["semua_fast_track"],
        "Dummy classifier (baseline acak)": expected_cost(y, dummy.predict_proba(X)[:, 1], dummy_thr),
        "Model final": expected_cost(y, model.predict_proba(X)[:, 1], meta["threshold"]),
    }
    for name, cost in scen.items():
        rows.append({"skenario": name, "data": label, "cost_per_1000": cost,
                     "selisih_vs_review_manual_pct": (cost / manual - 1) * 100})
impact = pd.DataFrame(rows)
save_table(impact, "t10_impact", decimals=1)
impact

# %%
tiers = pd.read_csv(config.TABLES / "t8_tier_distribution_oot.csv").set_index("tier")
print(f"Porsi fast-track (tier Rendah, setelah aturan bisnis) OOT 2025: {fmt_id(tiers.loc['Rendah', 'pct_setelah_aturan'], 2)}% "
      f"dengan tingkat tolak aktual {fmt_id(tiers.loc['Rendah', 'denial_rate_setelah_aturan'], 2)}%")
