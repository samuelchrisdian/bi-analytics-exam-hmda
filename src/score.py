import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_prep import clean, split_frames
from src.explain import _is_binary_column, readable_name, reason_codes, shap_explain
from src.features import add_engineered

TIERS = ["Rendah", "Sedang", "Tinggi"]
_RANK = {t: i for i, t in enumerate(TIERS)}
REQUIRED_FIELDS = ["income", "ltv_calc", "dti_mid", "property_value"]
DTI_LIMIT = 50
LTV_LIMIT = 97
# Tier Rendah (fast-track) hanya bila tingkat tolaknya ≤ 1/5 rata-rata; 1/2 membuat tier Sedang kosong setelah kalibrasi.
LOW_TIER_MAX_REL_RATE = 0.2


def choose_low_threshold(y_oof, proba_oof, t_high: float, max_rel_rate: float = 0.5, grid=None) -> float:
    """Ambang terbesar ≤ t_high yang menjaga tingkat tolak tier Rendah ≤ max_rel_rate × rata-rata."""
    grid = config.THRESHOLD_GRID if grid is None else grid
    y, p = np.asarray(y_oof), np.asarray(proba_oof)
    limit = max_rel_rate * y.mean()
    ok = [t for t in grid if t <= t_high and (not (p < t).any() or y[p < t].mean() <= limit)]
    return float(max(ok)) if ok else float(grid[0])


def assign_tier(proba: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    proba = np.asarray(proba)
    return np.where(proba >= t_high, "Tinggi", np.where(proba < t_low, "Rendah", "Sedang")).astype(object)


def apply_business_rules(tiers: np.ndarray, X: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """Contoh kebijakan: data inti kosong → minimal Sedang; DTI > 50 atau LTV > 97 → Tinggi."""
    ranks = np.array([_RANK[t] for t in tiers])
    incomplete = X.reindex(columns=REQUIRED_FIELDS).isna().any(axis=1).to_numpy()
    over_limit = ((X["dti_mid"] > DTI_LIMIT) | (X["ltv_calc"] > LTV_LIMIT)).to_numpy()
    ranks = np.where(incomplete, np.maximum(ranks, _RANK["Sedang"]), ranks)
    ranks = np.where(over_limit, _RANK["Tinggi"], ranks)
    hit = np.where(over_limit, "melewati_batas_kebijakan", np.where(incomplete, "data_tidak_lengkap", ""))
    return np.array([TIERS[r] for r in ranks], dtype=object), pd.Series(hit, index=X.index)


def load_bundle() -> dict:
    meta = json.loads((config.MODELS / "final_threshold.json").read_text())
    bundle = {**meta, "model": joblib.load(config.MODELS / "final_model.joblib")}
    bg = config.MODELS / "shap_background.parquet"
    bundle["background"] = pd.read_parquet(bg) if bg.exists() else None
    return bundle


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Field HMDA mentah → 13 fitur model, lewat cleaning yang sama dengan training."""
    raw = raw.reindex(columns=config.RAW_USECOLS).astype(object)
    return split_frames(add_engineered(clean(raw)))[0]


def score_batch(raw: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Skor banyak aplikasi sekaligus tanpa SHAP; index mengikuti `raw`."""
    X = prepare_features(raw)
    proba = bundle["model"].predict_proba(X)[:, 1]
    tier_model = assign_tier(proba, bundle["t_low"], bundle["threshold"])
    tiers, hit = apply_business_rules(tier_model, X)
    return pd.DataFrame({"proba": proba, "tier_model": tier_model, "tier": tiers, "aturan": hit.to_numpy()}, index=raw.index)


def contributions(app: dict, bundle: dict, top: int = 8) -> pd.DataFrame:
    """Kontribusi SHAP (log-odds) terbesar untuk satu aplikasi, urut |nilai| menurun."""
    X = prepare_features(pd.DataFrame([app]))
    expl = shap_explain(bundle["model"], X, background=bundle.get("background"))
    names, data = list(expl.feature_names), np.asarray(expl.data[0])
    # Sama dengan reason_codes: kolom one-hot tidak aktif bukan atribut pemohon.
    keep = [i for i, f in enumerate(names) if not (_is_binary_column(f) and data[i] <= 0)]
    out = pd.DataFrame({"fitur": [readable_name(names[i]) for i in keep], "shap": expl.values[0][keep]})
    return out.reindex(out["shap"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)


def score_application(app: dict, bundle: dict | None = None) -> dict:
    """Skor satu aplikasi dari field HMDA mentah; field yang hilang diperlakukan kosong."""
    bundle = bundle or load_bundle()
    X = prepare_features(pd.DataFrame([app]))
    model = bundle["model"]
    proba = float(model.predict_proba(X)[0, 1])
    tiers, hit = apply_business_rules(assign_tier(np.array([proba]), bundle["t_low"], bundle["threshold"]), X)
    expl = shap_explain(model, X, background=bundle.get("background"))
    return {
        "proba": proba,
        "tier": str(tiers[0]),
        "reason_codes": reason_codes(expl.values[0], list(expl.feature_names), data=np.asarray(expl.data[0])),
        "aturan": str(hit.iloc[0]),
    }
