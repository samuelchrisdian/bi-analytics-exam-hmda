import numpy as np
import pandas as pd
import pytest

from src import config
from src.monitor import psi, psi_status, psi_table

rng = np.random.default_rng(42)


def test_psi_identical():
    s = pd.Series(rng.normal(size=5000))
    assert psi(s, s) < 1e-6


def test_psi_shift():
    assert psi(pd.Series(rng.normal(size=5000)), pd.Series(rng.normal(1, 1, 5000))) > 0.25


def test_psi_all_nan_actual():
    assert np.isfinite(psi(pd.Series(rng.normal(size=500)), pd.Series([np.nan] * 500)))


def test_psi_nan_bin_counts():
    e = pd.Series(np.r_[rng.normal(size=900), [np.nan] * 100])
    a = pd.Series(np.r_[rng.normal(size=500), [np.nan] * 500])
    assert psi(e, a) > 0.25


def test_psi_categorical():
    assert psi(pd.Series(list("aab") * 100), pd.Series(list("abb") * 100), categorical=True) > 0


def test_psi_categorical_new_category_finite():
    assert np.isfinite(psi(pd.Series(list("ab") * 50), pd.Series(list("abc") * 50), categorical=True))


@pytest.mark.parametrize("v,s", [(0.0999, "Stabil"), (0.10, "Waspada"), (0.25, "Waspada"), (0.2501, "Tindakan")])
def test_psi_status(v, s):
    assert psi_status(v) == s


def test_psi_table(synthetic_xy):
    X, _ = synthetic_xy
    shifted = X.copy()
    shifted["dti_mid"] = shifted["dti_mid"] + 20
    t = psi_table(X, shifted)
    assert list(t.columns) == ["fitur", "psi", "status"]
    assert set(t["fitur"]) == set(config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES)
    assert t.iloc[0]["fitur"] == "dti_mid" and t.iloc[0]["status"] == "Tindakan"
