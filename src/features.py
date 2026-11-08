import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    income_usd = out["income"] * 1000
    out["loan_to_income"] = out["loan_amount"] / income_usd
    out["property_to_income"] = out["property_value"] / income_usd
    # LTV laporan HMDA bocor (sering kosong saat ditolak); hitung ulang dari data pengajuan.
    out["ltv_calc"] = out["loan_amount"] / out["property_value"] * 100
    ratio_cols = ["loan_to_income", "property_to_income", "ltv_calc"]
    out[ratio_cols] = out[ratio_cols].replace([np.inf, -np.inf], np.nan)
    return out


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Clip tiap kolom ke persentil yang dihitung saat fit (data train)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.lower_ = np.nanpercentile(arr, self.lower * 100, axis=0)
        self.upper_ = np.nanpercentile(arr, self.upper * 100, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return getattr(self, "feature_names_in_", np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object))


def _log1p_nonneg(X):
    return np.log1p(np.clip(X, 0, None))


def _as_str_keep_nan(X):
    from src.data_prep import code_str

    return pd.DataFrame(X).apply(lambda c: c.map(code_str)).astype(object)


def _numeric_pipeline(log: bool) -> Pipeline:
    steps = [("cap", QuantileCapper())]
    if log:
        steps.append(("log", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")))
    # Tanpa missing indicator: pola kosong HMDA membocorkan keputusan (lihat t3_leakage_test, kolom lift).
    steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    return Pipeline(steps)


def build_preprocessor() -> ColumnTransformer:
    linear_cols = [c for c in config.NUMERIC_FEATURES if c not in config.LOG_FEATURES]
    categorical = Pipeline([
        ("to_str", FunctionTransformer(_as_str_keep_nan, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("log", _numeric_pipeline(log=True), config.LOG_FEATURES),
         ("num", _numeric_pipeline(log=False), linear_cols),
         ("cat", categorical, config.CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    return list(fitted_preprocessor.get_feature_names_out())
