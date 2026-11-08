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
# # Tahap 8 — Model Deployment (desain)
# Risk tier dibentuk dari probabilitas model final. Batas tier Rendah dipilih dari OOF data latih; evaluasi beban kerja memakai OOT 2025.

# %%
import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_prep import load_processed
from src.reporting import fmt_id, save_table
from src.score import DTI_LIMIT, LOW_TIER_MAX_REL_RATE, LTV_LIMIT, TIERS, apply_business_rules, assign_tier, choose_low_threshold, score_application

meta_path = config.MODELS / "final_threshold.json"
meta = json.loads(meta_path.read_text())
model = joblib.load(config.MODELS / "final_model.joblib")
X_train, y_train, _, _ = load_processed("train")
X_oot, y_oot, _, _ = load_processed("oot")
oof = pd.read_parquet(config.DATA_PROCESSED / "oof_proba_train.parquet")

# %% [markdown]
# ## 1. Batas tier

# %%
t_high = meta["threshold"]
t_low = choose_low_threshold(y_train, oof["final"], t_high, max_rel_rate=LOW_TIER_MAX_REL_RATE)
meta["t_low"] = t_low
meta_path.write_text(json.dumps(meta, indent=2))
X_train.sample(1000, random_state=config.RANDOM_STATE).to_parquet(config.MODELS / "shap_background.parquet")
print(f"t_low = {t_low}, t_high = {t_high}; denial rate OOF tier Rendah = {y_train[oof['final'] < t_low].mean():.2%} "
      f"(batas {LOW_TIER_MAX_REL_RATE * y_train.mean():.2%})")

# Justifikasi parameter memakai data latih/OOF saja; OOT hanya konfirmasi.
choice_rows = []
for rate in (0.5, LOW_TIER_MAX_REL_RATE):
    tl = choose_low_threshold(y_train, oof["final"], t_high, max_rel_rate=rate)
    tiers_oof = assign_tier(oof["final"].to_numpy(), tl, t_high)
    row = {"max_rel_rate": rate, "t_low": tl}
    for t in TIERS:
        m = tiers_oof == t
        row[f"pct_{t.lower()}"] = m.mean() * 100
        row[f"tolak_{t.lower()}"] = y_train[m].mean() * 100 if m.any() else np.nan
    choice_rows.append(row)
save_table(pd.DataFrame(choice_rows), "t8_low_tier_choice_oof", decimals=2)
ltv_rows = []
for lim in (95, 97, 100):
    m = (X_train["ltv_calc"] > lim).to_numpy()
    ltv_rows.append({"aturan": f"ltv_calc > {lim}", "pct_terpicu_train": m.mean() * 100, "tolak_train": y_train[m].mean() * 100})
m = (X_train["dti_mid"] > DTI_LIMIT).to_numpy()
ltv_rows.append({"aturan": f"dti_mid > {DTI_LIMIT}", "pct_terpicu_train": m.mean() * 100, "tolak_train": y_train[m].mean() * 100})
save_table(pd.DataFrame(ltv_rows), "t8_policy_limits_train", decimals=2)
pd.DataFrame(choice_rows)

# %%
tier_def = pd.DataFrame([
    {"tier": "Rendah", "rentang_probabilitas": f"< {fmt_id(t_low, 2)}", "aksi": "Fast-track: verifikasi dokumen ringan oleh underwriter"},
    {"tier": "Sedang", "rentang_probabilitas": f"{fmt_id(t_low, 2)} – < {fmt_id(t_high, 2)}", "aksi": "Review standar oleh underwriter"},
    {"tier": "Tinggi", "rentang_probabilitas": f"≥ {fmt_id(t_high, 2)}", "aksi": "Review mendalam oleh senior underwriter; tidak ada auto-reject"},
])
save_table(tier_def, "t8_risk_tier")
tier_def

# %% [markdown]
# ## 2. Distribusi tier dan tingkat tolak aktual pada OOT 2025

# %%
p_oot = model.predict_proba(X_oot)[:, 1]
tier_model = assign_tier(p_oot, t_low, t_high)
tier_final, hit = apply_business_rules(tier_model, X_oot)
rows = []
for t in TIERS:
    mm, mf = tier_model == t, tier_final == t
    rows.append({"tier": t, "n_model": int(mm.sum()), "pct_model": mm.mean() * 100,
                 "denial_rate_model": y_oot[mm].mean() * 100 if mm.any() else np.nan,
                 "n_setelah_aturan": int(mf.sum()), "pct_setelah_aturan": mf.mean() * 100,
                 "denial_rate_setelah_aturan": y_oot[mf].mean() * 100 if mf.any() else np.nan})
dist = pd.DataFrame(rows)
save_table(dist, "t8_tier_distribution_oot", decimals=2)
print("Denial rate naik monoton (model):", dist["denial_rate_model"].is_monotonic_increasing)
dist

# %%
rules = hit.value_counts().rename_axis("aturan").reset_index(name="n")
rules["pct"] = rules["n"] / len(hit) * 100
rules["denial_rate"] = [y_oot[hit == a].mean() * 100 for a in rules["aturan"]]
rules["aturan"] = rules["aturan"].replace({"": "tidak ada aturan terpicu"})
changed = int((tier_model != tier_final).sum())
print(f"Tier berubah karena aturan bisnis: {changed} aplikasi ({changed / len(hit):.2%})")
rules_def = pd.DataFrame([
    {"aturan": "data_tidak_lengkap", "kondisi": "income, ltv_calc, dti_mid, atau property_value kosong", "efek": "Tier minimal Sedang"},
    {"aturan": "melewati_batas_kebijakan", "kondisi": f"dti_mid > {DTI_LIMIT} atau ltv_calc > {LTV_LIMIT}", "efek": "Tier menjadi Tinggi"},
])
save_table(rules_def, "t8_business_rules")
save_table(rules, "t8_business_rules_hits_oot", decimals=2)
rules

# %% [markdown]
# ## 3. Demo fungsi scoring untuk tiga aplikasi OOT (input = field HMDA mentah)

# %%
raw_oot = pd.read_parquet(config.DATA_INTERIM / f"hmda_filtered_{config.YEAR_OOT}.parquet")
picks = {"risiko rendah": int(np.argmin(p_oot)), "dekat threshold": int(np.argmin(np.abs(p_oot - t_high))), "risiko tinggi": int(np.argmax(p_oot))}
bundle = {**meta, "model": model, "background": pd.read_parquet(config.MODELS / "shap_background.parquet")}
demo = []
for label, i in picks.items():
    idx = X_oot.index[i]
    app = raw_oot.loc[idx].to_dict()
    r = score_application(app, bundle)
    demo.append({"contoh": label, "loan_amount": app["loan_amount"], "income": app["income"],
                 "debt_to_income_ratio": app["debt_to_income_ratio"], "probabilitas": r["proba"], "tier": r["tier"],
                 "aturan": r["aturan"] or "-", "reason_codes": "; ".join(r["reason_codes"]) or "-",
                 "aktual": "ditolak" if y_oot.loc[idx] == 1 else "disetujui"})
demo = pd.DataFrame(demo)
save_table(demo, "t8_scoring_demo", decimals=3)
demo
