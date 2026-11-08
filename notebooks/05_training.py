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
# # Tahap 5 — Training & Testing the Model
# Hanya data **train 2024** yang dipakai. Test dan OOT tidak dibaca di notebook ini.

# %%
import os
import platform

import imblearn
import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone

from src import config
from src.data_prep import load_processed
from src.reporting import fmt_auto, fmt_id, save_table
from src.train import MODEL_NAMES, SCORING, cv_evaluate, make_model_specs, make_pipeline, oof_proba, tune

X_train, y_train, _, _ = load_processed("train")
pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
print(X_train.shape, f"positif {y_train.mean():.2%}", f"pos_weight {pos_weight:.3f}")

# %% [markdown]
# ## 1. Tuning, CV, dan OOF predictions per model

# %%
results, cv_tables, oof = {}, [], pd.DataFrame({"y": y_train}, index=X_train.index)
for name in MODEL_NAMES:
    r = tune(name, X_train, y_train)
    cv = cv_evaluate(r.pipeline, X_train, y_train)
    cv.insert(0, "model", name)
    cv_tables.append(cv)
    oof[name] = oof_proba(r.pipeline, X_train, y_train)
    joblib.dump(r.pipeline, config.MODELS / f"{name}.joblib")
    results[name] = r
    print(f"{name}: {r.fit_seconds:.1f} s | best {r.best_params}")
    print(cv.round(4).to_string(index=False))

oof.to_parquet(config.DATA_PROCESSED / "oof_proba_train.parquet")

# %%
cv_all = pd.concat(cv_tables, ignore_index=True)
cv_all["mean_std"] = [f"{fmt_id(m, 3)} ± {fmt_id(s, 3)}" for m, s in zip(cv_all["mean"], cv_all["std"])]
save_table(cv_all, "t5_cv_results", decimals=4)
cv_all.pivot(index="model", columns="metrik", values="mean_std").loc[MODEL_NAMES]

# %%
best_rows = []
for name, r in results.items():
    params = {k.replace("clf__", ""): v for k, v in r.best_params.items()}
    best_rows.append({
        "model": name,
        "best_params": ", ".join(f"{k}={fmt_id(v, 4) if isinstance(v, float) else v}" for k, v in params.items()) or "-",
        "waktu_tuning_detik": r.fit_seconds,
        "n_data_tuning": r.tuned_on_n,
    })
best = pd.DataFrame(best_rows)
save_table(best, "t5_best_params", decimals=1)
best

# %% [markdown]
# ## 2. Eksperimen imbalance pada model terbaik (CV PR-AUC)

# %%
ap = cv_all[cv_all["metrik"] == "average_precision"].set_index("model")["mean"]
best_name = ap.drop("dummy").idxmax()
print("Model terbaik menurut CV PR-AUC:", best_name)

weighted = results[best_name].pipeline
base_clf = clone(weighted.named_steps["clf"])
if best_name == "xgb":
    base_clf.set_params(scale_pos_weight=1)
else:
    base_clf.set_params(class_weight=None)
smote_pipe = make_pipeline(base_clf, smote=True)

imb_rows = []
for label, pipe in (("class weighting", weighted), ("SMOTE (tanpa bobot kelas)", smote_pipe)):
    t = cv_evaluate(pipe, X_train, y_train)
    t.insert(0, "strategi", label)
    imb_rows.append(t)
imb = pd.concat(imb_rows, ignore_index=True)
imb.insert(0, "model", best_name)
imb["mean_std"] = [f"{fmt_id(m, 3)} ± {fmt_id(s, 3)}" for m, s in zip(imb["mean"], imb["std"])]
save_table(imb, "t5_imbalance_experiment", decimals=4)
imb.pivot(index="strategi", columns="metrik", values="mean_std")

# %% [markdown]
# ## 3. Setup eksperimen dan ruang hyperparameter

# %%
mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
cpu = next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
setup = pd.DataFrame([
    ("Data latih", f"{fmt_id(len(X_train))} baris (2024, 80% stratified), {X_train.shape[1]} fitur input"),
    ("Proporsi kelas positif (ditolak)", f"{fmt_id(y_train.mean() * 100, 2)}% (rasio negatif:positif {fmt_id(pos_weight, 2)})"),
    ("Validasi", f"StratifiedKFold {config.CV_FOLDS} fold, shuffle, random_state {config.RANDOM_STATE}"),
    ("Pencarian hyperparameter", f"RandomizedSearchCV, n_iter {config.N_ITER}, refit pada PR-AUC (average_precision)"),
    ("Metrik CV", ", ".join(SCORING)),
    ("Subsample tuning", f"Tidak (data latih {fmt_id(len(X_train))} ≤ ambang {fmt_id(config.TUNE_SUBSAMPLE_TRIGGER)} baris)"),
    ("Penanganan imbalance", "Utama: class weighting (balanced / scale_pos_weight); pembanding: SMOTE di dalam imblearn Pipeline"),
    ("Preprocessing", "Di dalam Pipeline: capping persentil 1–99, log1p (3 fitur nominal), imputasi median (tanpa missing indicator), StandardScaler, one-hot"),
    ("Seed", f"RANDOM_STATE = {config.RANDOM_STATE} di semua komponen"),
    ("Library", f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}, imbalanced-learn {imblearn.__version__}, pandas {pd.__version__}"),
    ("Perangkat keras", f"{cpu}, {os.cpu_count()} thread, RAM {fmt_id(mem_gb, 1)} GB; n_jobs = {config.N_JOBS}"),
    ("Test & OOT", "Tidak dipakai di tahap ini (disimpan untuk Tahap 6–7)"),
], columns=["aspek", "keterangan"])
save_table(setup, "t5_experiment_setup")
setup

# %%
REASON = {
    "C": "Kekuatan regularisasi L2 (C kecil = regularisasi kuat); meredam multikolinearitas fitur ukuran",
    "n_estimators": "Jumlah pohon; lebih banyak = lebih stabil tetapi lebih lambat",
    "max_depth": "Kedalaman pohon; mengendalikan kompleksitas dan overfitting",
    "min_samples_leaf": "Minimal sampel per daun; regularisasi pohon pada kelas minoritas",
    "max_features": "Proporsi fitur per split; menambah keragaman pohon",
    "learning_rate": "Langkah boosting; kecil = lebih general tetapi butuh lebih banyak pohon",
    "subsample": "Proporsi baris per pohon; mengurangi overfitting",
    "colsample_bytree": "Proporsi fitur per pohon; mengurangi overfitting",
    "min_child_weight": "Minimal bobot per daun; regularisasi",
}
space_rows = []
for name, (est, space) in make_model_specs(pos_weight).items():
    if not space:
        space_rows.append({"model": name, "hyperparameter": "strategy", "ruang_pencarian": "stratified (tetap)", "alasan": "Baseline acak sesuai proporsi kelas"})
    for k, dist in space.items():
        p = k.replace("clf__", "")
        if isinstance(dist, list):
            rng = ", ".join(fmt_id(v, 1) if isinstance(v, float) else str(v) for v in dist)
        elif dist.dist.name == "randint":
            lo, hi = dist.support()
            rng = f"integer {lo}–{hi}"
        elif dist.dist.name == "loguniform":
            rng = f"log-uniform {fmt_auto(dist.args[0])}–{fmt_auto(dist.args[1])}"
        else:
            rng = f"uniform {fmt_id(dist.args[0], 1)}–{fmt_id(dist.args[0] + dist.args[1], 1)}"
        space_rows.append({"model": name, "hyperparameter": p, "ruang_pencarian": rng, "alasan": REASON[p]})
fixed = {
    "logreg": "class_weight=balanced, max_iter=2000, penalty L2",
    "rf": "class_weight=balanced_subsample",
    "xgb": f"scale_pos_weight={fmt_id(pos_weight, 3)} (rasio negatif/positif), tree_method=hist, eval_metric=aucpr",
}
for name, txt in fixed.items():
    space_rows.append({"model": name, "hyperparameter": "(tetap)", "ruang_pencarian": txt, "alasan": "Penanganan imbalance / konfigurasi dasar"})
space_tbl = pd.DataFrame(space_rows)
save_table(space_tbl, "t5_hyperparameter_space")
space_tbl
