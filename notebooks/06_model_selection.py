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
# # Tahap 6 — Choosing the Appropriate Model
# *Threshold* dan pemilihan model memakai **OOF predictions data latih** (Tahap 5). Test 2024 hanya untuk konfirmasi/pelaporan.
# OOT 2025 tidak dibaca.

# %%
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict

from src import config
from src.data_prep import load_processed
from src.evaluate import (
    baseline_costs, cost_curve, expected_calibration_error, expected_cost, metrics_at, optimal_threshold,
    plot_calibration, plot_confusion, plot_cost_curve, plot_pr, plot_roc,
)
from src.reporting import fmt_id, save_figure, save_table, set_plot_style
from src.train import MODEL_NAMES, make_cv

set_plot_style()
LABEL = {"dummy": "Dummy (baseline)", "logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
INTERPRETABILITY = ["logreg", "rf", "xgb"]

X_train, y_train, _, _ = load_processed("train")
X_test, y_test, _, _ = load_processed("test")
oof = pd.read_parquet(config.DATA_PROCESSED / "oof_proba_train.parquet")
assert oof.index.equals(X_train.index)
models = {m: joblib.load(config.MODELS / f"{m}.joblib") for m in MODEL_NAMES}
test_proba = {m: models[m].predict_proba(X_test)[:, 1] for m in MODEL_NAMES}

# %% [markdown]
# ## 1. Threshold cost-optimal dari OOF (C_FN = 5, C_FP = 1)

# %%
thr_rows = []
for m in MODEL_NAMES:
    t = optimal_threshold(y_train, oof[m])
    thr_rows.append({"model": LABEL[m], "kode": m, "threshold_optimal": t,
                     "biaya_oof_threshold_optimal": expected_cost(y_train, oof[m], t),
                     "biaya_oof_threshold_0_5": expected_cost(y_train, oof[m], 0.5)})
thr = pd.DataFrame(thr_rows)
save_table(thr, "t6_threshold", decimals=2)
thr

# %% [markdown]
# ## 2. Aturan pemilihan
# 1. Biaya OOF terendah pada threshold optimal masing-masing; 2. PR-AUC/recall; 3. stabilitas CV;
# 4. bila selisih biaya OOF model terbaik dan Logistic Regression < 2% relatif, pilih Logistic Regression (interpretability).
#
# Aturan seri (ditetapkan saat model freeze, tanpa melihat test): model dengan biaya OOF dalam 1% dari yang terendah
# dianggap setara, karena selisih sekecil itu di bawah variasi antar-fold; di antara yang setara dipilih PR-AUC CV tertinggi (kriteria 2).

# %%
TIE = 0.01
cand = thr[thr["kode"] != "dummy"].set_index("kode")
cv = pd.read_csv(config.TABLES / "t5_cv_results.csv").pivot(index="model", columns="metrik", values="mean")
cv_std = pd.read_csv(config.TABLES / "t5_cv_results.csv").pivot(index="model", columns="metrik", values="std")
cost = cand["biaya_oof_threshold_optimal"]
best = cost.idxmin()
gap_lr = cost["logreg"] / cost[best] - 1
tied = cost[cost <= cost[best] * (1 + TIE)].index.tolist()
if gap_lr < 0.02:
    chosen = "logreg"
else:
    chosen = cv.loc[tied, "average_precision"].idxmax()
print(f"Biaya OOF terendah: {best}; setara (≤ {TIE:.0%}): {tied}; selisih LR {gap_lr:.2%} → dipilih: {chosen}")

sens_oof = {}
for ratio in config.COST_RATIOS:
    sens_oof[ratio] = {m: expected_cost(y_train, oof[m], optimal_threshold(y_train, oof[m], ratio, 1), ratio, 1) for m in ("logreg", "rf", "xgb")}
rationale = pd.DataFrame([
    {"kriteria": "1. Biaya OOF terendah pada threshold optimal",
     **{LABEL[m]: cost[m] for m in ("logreg", "rf", "xgb")},
     "keterangan": f"Selisih Random Forest vs XGBoost {fmt_id(abs(cost['rf'] / cost['xgb'] - 1) * 100, 2)}% (< 1%) → dianggap setara"},
    {"kriteria": "2. PR-AUC CV (mean)", **{LABEL[m]: cv.loc[m, "average_precision"] for m in ("logreg", "rf", "xgb")},
     "keterangan": "Pemecah seri: kemampuan memeringkat kelas minoritas (ditolak)"},
    {"kriteria": "2b. Recall CV (threshold 0,5)", **{LABEL[m]: cv.loc[m, "recall"] for m in ("logreg", "rf", "xgb")},
     "keterangan": "FN lebih mahal, recall tinggi diutamakan"},
    {"kriteria": "3. Stabilitas: std PR-AUC CV", **{LABEL[m]: cv_std.loc[m, "average_precision"] for m in ("logreg", "rf", "xgb")},
     "keterangan": "Semua stabil (std < 0,01)"},
    *[{"kriteria": f"Sensitivitas: biaya OOF rasio C_FN = {r}", **{LABEL[m]: v[m] for m in ("logreg", "rf", "xgb")},
       "keterangan": f"Terendah: {LABEL[min(v, key=v.get)]}"} for r, v in sens_oof.items()],
    {"kriteria": "4. Interpretability", "Logistic Regression": np.nan, "Random Forest": np.nan, "XGBoost": np.nan,
     "keterangan": f"LR paling transparan, tetapi biayanya {fmt_id(gap_lr * 100, 1)}% lebih tinggi (> 2%) → tidak dipilih; RF/XGBoost dijelaskan dengan SHAP"},
])
save_table(rationale, "t6_selection_rationale", decimals=3)
rationale

# %% [markdown]
# ## 3. Kalibrasi model terpilih (probabilitas dipakai untuk risk tier)

# %%
ece_before = expected_calibration_error(y_train, oof[chosen])
calibrated = ece_before > 0.02
method = None
final_model, final_oof = models[chosen], oof[chosen].to_numpy()
if calibrated:
    method = "isotonic" if y_train.sum() >= 1000 else "sigmoid"
    cal = CalibratedClassifierCV(clone(models[chosen]), method=method, cv=make_cv(), ensemble=False)
    final_oof = cross_val_predict(cal, X_train, y_train, cv=make_cv(), method="predict_proba", n_jobs=config.N_JOBS)[:, 1]
    final_model = cal.fit(X_train, y_train)
ece_after = expected_calibration_error(y_train, final_oof)
final_thr = optimal_threshold(y_train, final_oof)
oof["final"] = final_oof
oof.to_parquet(config.DATA_PROCESSED / "oof_proba_train.parquet")
calib = pd.DataFrame([{"model": LABEL[chosen], "ece_oof_sebelum": ece_before, "dikalibrasi": calibrated,
                       "metode": method or "-", "ece_oof_sesudah": ece_after, "threshold_final": final_thr,
                       "biaya_oof_final": expected_cost(y_train, final_oof, final_thr)}])
save_table(calib, "t6_calibration", decimals=4)
calib

# %% [markdown]
# ## 4. Evaluasi test 2024: threshold 0,5 vs threshold OOF-optimal

# %%
rows = []
thr_of = cand["threshold_optimal"].to_dict() | {"dummy": thr.set_index("kode").loc["dummy", "threshold_optimal"]}
for m in MODEL_NAMES:
    for label, t in (("0,5", 0.5), ("OOF-optimal", thr_of[m])):
        rows.append({"model": LABEL[m], "threshold": t, "jenis_threshold": label, **metrics_at(y_test, test_proba[m], t)})
final_test_proba = final_model.predict_proba(X_test)[:, 1]
if calibrated:
    rows.append({"model": f"{LABEL[chosen]} terkalibrasi (final)", "threshold": final_thr, "jenis_threshold": "OOF-optimal",
                 **metrics_at(y_test, final_test_proba, final_thr)})
for name, cost in baseline_costs(y_test).items():
    rows.append({"model": "Tanpa model: " + name.replace("_", " "), "threshold": np.nan, "jenis_threshold": "-", "cost_per_1000": cost})
comparison = pd.DataFrame(rows)
save_table(comparison, "t6_model_comparison", decimals=3)
comparison

# %%
opt = comparison[comparison["jenis_threshold"] == "OOF-optimal"].set_index("model")
print("Accuracy tertinggi (OOF-optimal):", opt["accuracy"].idxmax(), "| biaya test terendah:", opt["cost_per_1000"].idxmin())
print("Peringkat biaya OOF:", list(cand["biaya_oof_threshold_optimal"].sort_values().index))
print("Peringkat biaya test:", list(opt.drop(index=[LABEL["dummy"]], errors="ignore")["cost_per_1000"].sort_values().index))

# %% [markdown]
# ## 5. Sensitivitas rasio biaya (C_FP = 1, C_FN = 2, 5, 10)

# %%
sens = []
for ratio in config.COST_RATIOS:
    for m in INTERPRETABILITY:
        t = optimal_threshold(y_train, oof[m], c_fn=ratio, c_fp=1)
        sens.append({"rasio_c_fn": ratio, "model": LABEL[m], "threshold": t,
                     "biaya_oof": expected_cost(y_train, oof[m], t, ratio, 1),
                     "biaya_test": expected_cost(y_test, test_proba[m], t, ratio, 1)})
sens = pd.DataFrame(sens)
sens["terbaik_oof"] = sens.groupby("rasio_c_fn")["biaya_oof"].transform("min") == sens["biaya_oof"]
save_table(sens, "t6_cost_sensitivity", decimals=3)
sens

# %% [markdown]
# ## 6. Grafik

# %%
curves = {LABEL[m]: (y_test.to_numpy(), test_proba[m]) for m in MODEL_NAMES}
save_figure(plot_roc(curves), "f6_roc")
save_figure(plot_pr({k: v for k, v in curves.items() if k != LABEL["dummy"]}), "f6_pr")
for m in INTERPRETABILITY:
    save_figure(plot_confusion(y_test, test_proba[m] >= thr_of[m]), f"f6_confusion_{m}")
if calibrated:
    save_figure(plot_confusion(y_test, final_test_proba >= final_thr), "f6_confusion_final")
save_figure(plot_cost_curve({LABEL[m]: cost_curve(y_train, oof[m]) for m in INTERPRETABILITY}), "f6_cost_curve")
cal_curves = {LABEL[m]: (y_test.to_numpy(), test_proba[m]) for m in INTERPRETABILITY}
if calibrated:
    cal_curves[f"{LABEL[chosen]} terkalibrasi"] = (y_test.to_numpy(), final_test_proba)
save_figure(plot_calibration(cal_curves), "f6_calibration")

# %% [markdown]
# ## 7. Simpan model final (freeze)

# %%
joblib.dump(final_model, config.MODELS / "final_model.joblib")
meta = {"model": chosen, "threshold": final_thr, "c_fn": config.COST_FN, "c_fp": config.COST_FP,
        "calibrated": bool(calibrated), "calibration_method": method}
(config.MODELS / "final_threshold.json").write_text(json.dumps(meta, indent=2))
meta
