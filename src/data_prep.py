import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src import config

MISSING_STRINGS = {"NA", "Exempt", ""}


def load_raw(year: int, state: str | None = None) -> pd.DataFrame:
    """Baca CSV HMDA sebagai string; konversi tipe dilakukan di clean()."""
    state = state or config.STATE
    path = config.DATA_RAW / f"hmda_{state}_{year}.csv"
    usecols = set(config.RAW_USECOLS)
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False,
                       usecols=lambda c: c in usecols).astype(object)


def apply_scope_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{"langkah": "Data mentah", "filter": "-", "n_baris": len(df), "n_dibuang": 0}]
    out = df
    for col, allowed in config.SCOPE_FILTERS:
        before = len(out)
        out = out[pd.to_numeric(out[col], errors="coerce").isin(allowed)]
        rows.append({
            "langkah": col,
            "filter": f"{col} ∈ {{{', '.join(map(str, sorted(allowed)))}}}",
            "n_baris": len(out),
            "n_dibuang": before - len(out),
        })
    funnel = pd.DataFrame(rows)
    funnel["persen_tersisa"] = funnel["n_baris"] / len(df) * 100
    return out.copy(), funnel


def make_target(df: pd.DataFrame) -> pd.Series:
    codes = pd.to_numeric(df["action_taken"], errors="coerce")
    if not codes.isin({1, 2, 3}).all():
        raise ValueError("action_taken di luar {1, 2, 3}; jalankan apply_scope_filters dulu")
    return (codes == 3).astype("int8").rename("y")


def _is_missing(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().isin(MISSING_STRINGS)


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        miss = _is_missing(s)
        present = s[~miss]
        numeric = pd.to_numeric(present, errors="coerce").notna().all() and len(present) > 0
        rows.append({
            "kolom": col,
            "tipe": "numerik" if numeric else "kategorikal/teks",
            "n_missing": int(miss.sum()),
            "pct_missing": float(miss.mean() * 100),
            "n_unique": int(present.nunique()),
            "contoh": str(present.iloc[0]) if len(present) else "-",
        })
    return pd.DataFrame(rows)


DTI_MAP = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}
DTI_MAP.update({str(v): float(v) for v in range(36, 50)})

_DERIVED = {"dti_mid", "ltv_calc", "loan_to_income", "property_to_income", "has_coapplicant"}
RAW_NUMERIC = [c for c in config.NUMERIC_FEATURES if c not in _DERIVED]
RAW_CATEGORICAL = [c for c in config.CATEGORICAL_FEATURES if c not in _DERIVED]


def parse_dti(s: pd.Series) -> pd.Series:
    """Rentang DTI HMDA → nilai tengah; rentang terbuka <20% dan >60% dipetakan ke 15 dan 65."""
    return s.map(DTI_MAP).astype(float).rename("dti_mid")


def code_str(v):
    """Kode HMDA sebagai string kanonik: 44, 44.0, "44" → "44"; kosong tetap NaN."""
    if pd.isna(v):
        return np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v).strip()
    return str(int(f)) if f.is_integer() else str(f)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include="object").columns
    out[obj_cols] = out[obj_cols].mask(out[obj_cols].isin(MISSING_STRINGS))
    # Input API/CSV bisa bertipe angka; samakan dengan format string HMDA sebelum mapping.
    for col in [c for c in RAW_CATEGORICAL + ["debt_to_income_ratio", "co-applicant_age"] if c in out]:
        out[col] = out[col].map(code_str).astype(object)
    for col in [c for c in RAW_CATEGORICAL if c in out]:
        out[col] = out[col].mask(out[col] == "1111")
    for col in [c for c in RAW_NUMERIC if c in out]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "income" in out:
        out.loc[out["income"] <= 0, "income"] = np.nan
    if "debt_to_income_ratio" in out:
        out["dti_mid"] = parse_dti(out["debt_to_income_ratio"])
    if "co-applicant_age" in out:
        # 9999 = tidak ada co-applicant; 8888 = ada tetapi usia tidak tersedia; kosong = tidak diketahui.
        age = out["co-applicant_age"]
        out["has_coapplicant"] = np.where(age.isna(), np.nan, (age != "9999").astype(float))
    return out


LIFT_FLAG = 2.0
MIN_BLANKS_FOR_LIFT = 30


def missingness_by_target(df: pd.DataFrame, y: pd.Series, cols: list[str]) -> pd.DataFrame:
    """Uji leakage lewat pola kosong: selisih absolut antarkelas ATAU lift tingkat tolak saat kosong.

    Lift menangkap kolom yang jarang kosong tetapi hampir selalu ditolak ketika kosong.
    """
    yv = np.asarray(y)
    base = yv.mean()
    rows = []
    for col in cols:
        na = df[col].isna().to_numpy()
        p0, p1 = na[yv == 0].mean() * 100, na[yv == 1].mean() * 100
        denial_if_na = yv[na].mean() if na.any() else np.nan
        rows.append({"kolom": col, "pct_na_y0": p0, "pct_na_y1": p1, "selisih_pp": p1 - p0,
                     "n_kosong": int(na.sum()), "pct_ditolak_bila_kosong": denial_if_na * 100,
                     "lift": round(denial_if_na / base, 6) if na.any() else np.nan})
    out = pd.DataFrame(rows)
    high_lift = (out["lift"] >= LIFT_FLAG) & (out["n_kosong"] >= MIN_BLANKS_FOR_LIFT)
    out["flag_leakage"] = (out["selisih_pp"].abs() >= 20) | high_lift
    return out


def split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feats = [c for c in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES if c in df]
    return df[feats].copy(), df.reindex(columns=config.FAIRNESS_COLS), df.reindex(columns=config.META_COLS)


def stratified_split(X, y, sensitive, meta) -> dict[str, tuple]:
    idx_train, idx_test = train_test_split(
        X.index, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    return {name: (X.loc[idx], y.loc[idx], sensitive.loc[idx], meta.loc[idx])
            for name, idx in (("train", idx_train), ("test", idx_test))}


def save_processed(split: str, X, y, sensitive, meta) -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    X.to_parquet(config.DATA_PROCESSED / f"X_{split}.parquet")
    y.to_frame("y").to_parquet(config.DATA_PROCESSED / f"y_{split}.parquet")
    sensitive.to_parquet(config.DATA_PROCESSED / f"sensitive_{split}.parquet")
    meta.to_parquet(config.DATA_PROCESSED / f"meta_{split}.parquet")


def load_processed(split: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    d = config.DATA_PROCESSED
    return (pd.read_parquet(d / f"X_{split}.parquet"), pd.read_parquet(d / f"y_{split}.parquet")["y"],
            pd.read_parquet(d / f"sensitive_{split}.parquet"), pd.read_parquet(d / f"meta_{split}.parquet"))
