"""Evaluasi berbasis biaya. Prediksi positif (ditolak) bila proba >= threshold."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from src import config
from src.reporting import fmt_id

CLASS_LABELS = ["Disetujui", "Ditolak"]


def _counts(y, proba, threshold):
    y = np.asarray(y)
    pred = np.asarray(proba) >= threshold
    fn = int(((y == 1) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    return fn, fp, len(y)


def expected_cost(y, proba, threshold, c_fn=None, c_fp=None) -> float:
    c_fn = config.COST_FN if c_fn is None else c_fn
    c_fp = config.COST_FP if c_fp is None else c_fp
    fn, fp, n = _counts(y, proba, threshold)
    return 1000 * (c_fn * fn + c_fp * fp) / n


def cost_curve(y, proba, c_fn=None, c_fp=None, grid=None) -> pd.DataFrame:
    grid = config.THRESHOLD_GRID if grid is None else grid
    return pd.DataFrame({
        "threshold": grid,
        "cost_per_1000": [expected_cost(y, proba, t, c_fn, c_fp) for t in grid],
    })


def optimal_threshold(y, proba, c_fn=None, c_fp=None, grid=None) -> float:
    curve = cost_curve(y, proba, c_fn, c_fp, grid)
    return float(curve["threshold"].iloc[int(np.argmin(curve["cost_per_1000"].to_numpy()))])


def metrics_at(y, proba, threshold, c_fn=None, c_fp=None) -> dict:
    y = np.asarray(y)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    two_classes = len(np.unique(y)) == 2
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    return {
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if two_classes else np.nan,
        "pr_auc": float(average_precision_score(y, proba)) if two_classes else np.nan,
        "brier": float(brier_score_loss(y, proba, labels=[0, 1])),
        "cost_per_1000": expected_cost(y, proba, threshold, c_fn, c_fp),
    }


def baseline_costs(y, c_fn=None, c_fp=None) -> dict:
    """Biaya bila semua aplikasi direview manual (semua positif) atau semua di-fast-track (semua negatif)."""
    n = len(y)
    return {
        "semua_review_manual": expected_cost(y, np.ones(n), 0.5, c_fn, c_fp),
        "semua_fast_track": expected_cost(y, np.zeros(n), 0.5, c_fn, c_fp),
    }


def expected_calibration_error(y, proba, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    proba = np.asarray(proba, dtype=float)
    bins = np.digitize(proba, np.linspace(0, 1, n_bins + 1)[1:-1])
    ece = 0.0
    for b in np.unique(bins):
        mask = bins == b
        ece += abs(proba[mask].mean() - y[mask].mean()) * mask.mean()
    return float(ece)


def bootstrap_ci(y, proba, threshold, n_boot=None, seed=None) -> pd.DataFrame:
    n_boot = config.N_BOOT if n_boot is None else n_boot
    rng = np.random.default_rng(config.RANDOM_STATE if seed is None else seed)
    y = np.asarray(y)
    proba = np.asarray(proba)
    keys = ["roc_auc", "pr_auc", "recall"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = metrics_at(y[idx], proba[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    full = metrics_at(y, proba, threshold)
    rows = []
    for k in keys:
        s = np.asarray(samples[k])
        lo, hi = (np.percentile(s, [2.5, 97.5]) if len(s) else (np.nan, np.nan))
        rows.append({"metrik": k, "estimasi": full[k], "ci_bawah": lo, "ci_atas": hi, "n_valid": len(s)})
    return pd.DataFrame(rows)


def plot_roc(curves: dict) -> Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {fmt_id(roc_auc_score(y, p), 3)})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("Tingkat positif palsu (FPR)")
    ax.set_ylabel("Tingkat positif benar (TPR)")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def plot_pr(curves: dict) -> Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP {fmt_id(average_precision_score(y, p), 3)})")
    prevalence = np.mean(next(iter(curves.values()))[0])
    ax.axhline(prevalence, linestyle="--", color="grey", linewidth=1, label="Proporsi kelas positif")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def plot_confusion(y, pred) -> Figure:
    cm = confusion_matrix(np.asarray(y), np.asarray(pred).astype(int), labels=[0, 1])
    annot = np.vectorize(lambda v: fmt_id(int(v)))(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    return fig


def plot_cost_curve(curves_by_model: dict) -> Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, df in curves_by_model.items():
        line, = ax.plot(df["threshold"], df["cost_per_1000"], label=name)
        best = df.loc[df["cost_per_1000"].idxmin()]
        ax.scatter([best["threshold"]], [best["cost_per_1000"]], color=line.get_color(), zorder=3)
    ax.set_xlabel("Ambang (threshold)")
    ax.set_ylabel("Biaya harapan per 1.000 aplikasi")
    ax.legend(fontsize=8)
    return fig


def plot_calibration(curves: dict, n_bins: int = 10) -> Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac, marker="o", markersize=3, label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Kalibrasi sempurna")
    ax.set_xlabel("Probabilitas prediksi")
    ax.set_ylabel("Proporsi aktual ditolak")
    ax.legend(fontsize=8)
    return fig
