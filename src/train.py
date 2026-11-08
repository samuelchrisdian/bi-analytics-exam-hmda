import time
from dataclasses import dataclass, field

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.features import build_preprocessor

MODEL_NAMES = ["dummy", "logreg", "rf", "xgb"]
SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    # zero_division=0: dummy/model ekstrem bisa tidak memprediksi positif sama sekali di satu fold.
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
}


def make_model_specs(pos_weight: float) -> dict[str, tuple]:
    rs = config.RANDOM_STATE
    return {
        "dummy": (DummyClassifier(strategy="stratified", random_state=rs), {}),
        "logreg": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=rs),
            {"clf__C": loguniform(1e-3, 1e2)},
        ),
        "rf": (
            RandomForestClassifier(class_weight="balanced_subsample", n_jobs=config.N_JOBS, random_state=rs),
            {
                "clf__n_estimators": randint(200, 501),
                "clf__max_depth": [6, 10, 14, 20, None],
                "clf__min_samples_leaf": randint(1, 51),
                "clf__max_features": ["sqrt", 0.3, 0.5],
            },
        ),
        "xgb": (
            XGBClassifier(scale_pos_weight=pos_weight, tree_method="hist", eval_metric="aucpr",
                          n_jobs=config.N_JOBS, random_state=rs),
            {
                "clf__n_estimators": randint(200, 801),
                "clf__max_depth": randint(3, 9),
                "clf__learning_rate": loguniform(0.01, 0.2),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.6, 0.4),
                "clf__min_child_weight": randint(1, 11),
            },
        ),
    }


def make_pipeline(estimator, smote: bool = False):
    """SMOTE berada di dalam imblearn Pipeline agar hanya diterapkan pada fold latih."""
    if smote:
        return ImbPipeline([("prep", build_preprocessor()),
                            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
                            ("clf", estimator)])
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


@dataclass
class TuneResult:
    name: str
    pipeline: object
    best_params: dict
    cv_results: pd.DataFrame = field(repr=False)
    fit_seconds: float
    tuned_on_n: int


def tune(name: str, X: pd.DataFrame, y: pd.Series, n_iter: int | None = None) -> TuneResult:
    n_iter = n_iter or config.N_ITER
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    estimator, space = make_model_specs(pos_weight)[name]
    pipe = make_pipeline(estimator)
    start = time.perf_counter()

    if not space:
        pipe.fit(X, y)
        return TuneResult(name, pipe, {}, pd.DataFrame(), time.perf_counter() - start, len(X))

    X_tune, y_tune = X, y
    if len(X) > config.TUNE_SUBSAMPLE_TRIGGER:
        X_tune, _, y_tune, _ = train_test_split(
            X, y, train_size=config.TUNE_SUBSAMPLE, stratify=y, random_state=config.RANDOM_STATE)

    search = RandomizedSearchCV(
        pipe, space, n_iter=n_iter, scoring=SCORING, refit="average_precision", cv=make_cv(),
        n_jobs=config.N_JOBS, random_state=config.RANDOM_STATE,
    )
    search.fit(X_tune, y_tune)
    best = search.best_estimator_
    if len(X_tune) < len(X):
        best = clone(best).fit(X, y)
    return TuneResult(name, best, search.best_params_, pd.DataFrame(search.cv_results_),
                      time.perf_counter() - start, len(X_tune))


def cv_evaluate(pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    res = cross_validate(pipeline, X, y, cv=make_cv(), scoring=SCORING, n_jobs=config.N_JOBS)
    return pd.DataFrame([
        {"metrik": m, "mean": res[f"test_{m}"].mean(), "std": res[f"test_{m}"].std()} for m in SCORING
    ])


def oof_proba(pipeline, X: pd.DataFrame, y: pd.Series):
    return cross_val_predict(pipeline, X, y, cv=make_cv(), method="predict_proba", n_jobs=config.N_JOBS)[:, 1]
