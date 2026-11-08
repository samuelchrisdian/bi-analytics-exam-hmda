import numpy as np
import pandas as pd

from src import config

_FLOOR = 1e-4
_NAN_BIN = "__kosong__"


def _bin_labels(s: pd.Series, edges: np.ndarray) -> pd.Series:
    labels = pd.cut(s, edges, include_lowest=True).astype(object)
    return labels.where(s.notna(), _NAN_BIN).astype(str)


def psi(expected: pd.Series, actual: pd.Series, bins: int | None = None, categorical: bool = False) -> float:
    """Population Stability Index; bin numerik = kuantil data acuan, nilai kosong = bin tersendiri."""
    bins = bins or config.PSI_BINS
    expected, actual = pd.Series(expected), pd.Series(actual)
    if categorical:
        e_lab = expected.astype(object).where(expected.notna(), _NAN_BIN).astype(str)
        a_lab = actual.astype(object).where(actual.notna(), _NAN_BIN).astype(str)
    else:
        expected, actual = expected.astype(float), actual.astype(float)
        inner = np.unique(np.nanquantile(expected, np.linspace(0, 1, bins + 1)[1:-1])) if expected.notna().any() else []
        edges = np.r_[-np.inf, inner, np.inf]
        e_lab, a_lab = _bin_labels(expected, edges), _bin_labels(actual, edges)
    cats = sorted(set(e_lab) | set(a_lab))
    e = e_lab.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=_FLOOR).to_numpy()
    a = a_lab.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=_FLOOR).to_numpy()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float) -> str:
    if value < config.PSI_WARN:
        return "Stabil"
    if value <= config.PSI_ACT:
        return "Waspada"
    return "Tindakan"


def psi_table(ref: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
        v = psi(ref[col], new[col], categorical=col in config.CATEGORICAL_FEATURES)
        rows.append({"fitur": col, "psi": v, "status": psi_status(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
