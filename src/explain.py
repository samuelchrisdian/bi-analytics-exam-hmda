import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from src import config
from src.evaluate import metrics_at
from src.features import get_feature_names

UNAVAILABLE_MARKERS = ("Not Available", "Free Form", "8888")


def unwrap_pipeline(model):
    """Pipeline dasar di balik CalibratedClassifierCV(ensemble=False); selain itu model apa adanya."""
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def readable_name(feature: str) -> str:
    labels = config.FEATURE_LABELS
    if feature.startswith("missingindicator_"):
        base = feature.removeprefix("missingindicator_")
        return f"{labels.get(base, base)} (kosong)"
    if feature in labels:
        return labels[feature]
    for col in config.CATEGORICAL_FEATURES:
        if feature.startswith(col + "_"):
            return f"{labels[col]} = {feature.removeprefix(col + '_')}"
    return feature


def shap_explain(model, X: pd.DataFrame, background: pd.DataFrame | None = None) -> shap.Explanation:
    """Nilai SHAP kelas positif (ditolak) pada ruang fitur hasil preprocessing.

    background: data acuan untuk LinearExplainer; wajib diisi saat menjelaskan sedikit baris
    (mis. satu aplikasi), karena acuan = X membuat semua SHAP nol.
    """
    pipe = unwrap_pipeline(model)
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Z = prep.transform(X)
    names = get_feature_names(prep)
    if isinstance(clf, LogisticRegression):
        bg = Z if background is None else prep.transform(background)
        ex = shap.LinearExplainer(clf, bg)(Z)
    else:
        ex = shap.TreeExplainer(clf)(Z)
    values, base = ex.values, np.asarray(ex.base_values)
    if values.ndim == 3:
        values = values[:, :, 1]
        base = base[:, 1] if base.ndim == 2 else base
    return shap.Explanation(values=values, base_values=base, data=Z, feature_names=names)


def _is_binary_column(name: str) -> bool:
    return name.startswith("missingindicator_") or any(name.startswith(c + "_") for c in config.CATEGORICAL_FEATURES)


def reason_codes(values: np.ndarray, feature_names: list[str], k: int = 3, data: np.ndarray | None = None) -> list[str]:
    """Maksimal k fitur dengan kontribusi positif terbesar terhadap risiko penolakan.

    data: nilai fitur hasil preprocessing; kolom one-hot/indikator yang tidak aktif (≤ 0, indikator ikut di-scale) dilewati karena
    tidak menggambarkan atribut yang dimiliki pemohon.
    """
    codes = []
    for i in np.argsort(values)[::-1]:
        if len(codes) == k or values[i] <= 0:
            break
        if data is not None and _is_binary_column(feature_names[i]) and data[i] <= 0:
            continue
        codes.append(readable_name(feature_names[i]))
    return codes


def _is_unavailable(group) -> bool:
    return any(m in str(group) for m in UNAVAILABLE_MARKERS)


def fairness_table(y_true, y_pred, group: pd.Series, min_n: int = 100) -> pd.DataFrame:
    """Hasil menguntungkan = diprediksi disetujui (pred = 0)."""
    y = np.asarray(y_true)
    pred = np.asarray(y_pred).astype(int)
    g = np.asarray(group.fillna("Tidak tercatat"), dtype=object)
    rows = []
    for name in pd.Series(g).value_counts().index:
        m = g == name
        yy, pp = y[m], pred[m]
        rows.append({
            "grup": name, "n": int(m.sum()),
            "approval_rate_pred": float((pp == 0).mean()),
            "approval_rate_aktual": float((yy == 0).mean()),
            "tpr": float(pp[yy == 1].mean()) if (yy == 1).any() else np.nan,
            "fpr": float(pp[yy == 0].mean()) if (yy == 0).any() else np.nan,
        })
    t = pd.DataFrame(rows)
    eligible = t[(t["n"] >= min_n) & ~t["grup"].map(_is_unavailable)]
    ref = eligible.loc[eligible["approval_rate_pred"].idxmax()] if len(eligible) else None
    if ref is None:
        t["dir"], t["dir_aktual"], t["eod"] = np.nan, np.nan, np.nan
    else:
        t["dir"] = t["approval_rate_pred"] / ref["approval_rate_pred"]
        t["dir_aktual"] = t["approval_rate_aktual"] / ref["approval_rate_aktual"]
        t["eod"] = (1 - t["fpr"]) - (1 - ref["fpr"])

    def note(r) -> str:
        if _is_unavailable(r["grup"]):
            return "tidak tersedia"
        if ref is not None and r["grup"] == ref["grup"]:
            return "referensi"
        if r["n"] < min_n:
            return "n kecil"
        if r["dir"] < 0.8:
            return "DIR < 0,8"
        return ""

    t["catatan"] = t.apply(note, axis=1)
    return t[["grup", "n", "approval_rate_pred", "approval_rate_aktual", "tpr", "fpr", "dir", "dir_aktual", "eod", "catatan"]]


def segment_metrics(y, proba, threshold, segment: pd.Series) -> pd.DataFrame:
    y = np.asarray(y)
    proba = np.asarray(proba)
    seg = pd.Series(np.asarray(segment, dtype=object))
    rows = []
    for name in seg.dropna().unique():
        m = (seg == name).to_numpy()
        met = metrics_at(y[m], proba[m], threshold)
        rows.append({"segmen": name, "n": int(m.sum()), "denial_rate": float(y[m].mean()),
                     "roc_auc": met["roc_auc"], "pr_auc": met["pr_auc"], "recall": met["recall"],
                     "cost_per_1000": met["cost_per_1000"]})
    return pd.DataFrame(rows)
