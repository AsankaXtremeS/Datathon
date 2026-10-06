"""Task 1 validation: forward-in-time folds that mimic the test set, plus metrics.

Each fold is rebuilt like the real test: features for the validation window are computed with
build_feature_tables(train=rows before the cutoff, test=validation rows), so the validation rows only
see history from before the cutoff (frozen), exactly like the test rows only see train history.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (log_loss, brier_score_loss, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)

from task1_features import build_feature_tables, LABELS

# 6-week validation windows (the test window 2026-02-16..2026-03-28 is 6 weeks)
FOLDS = [
    ("A_FebMar25", "2025-02-17", "2025-03-29"),   # same season as test, one year earlier
    ("B_SepOct25", "2025-09-01", "2025-10-12"),   # different season, monsoon-heavy
    ("C_recent",   "2026-01-03", "2026-02-15"),   # most recent data before the test
]


def make_fold(train_lab: pd.DataFrame, actuals: pd.DataFrame, t: dict, start: str, end: str):
    """Return (X_train, X_valid, meta) with X_valid carrying its labels for scoring."""
    d = pd.to_datetime(train_lab.date)
    tr = train_lab[d < start]
    va = train_lab[(d >= start) & (d < end)]
    X_tr, X_va, meta = build_feature_tables(tr, actuals, va.drop(columns=LABELS), t)
    X_va = X_va.merge(va[["delivery_id"] + LABELS], on="delivery_id", how="left", validate="one_to_one")
    assert X_va[LABELS].notna().all().all()
    return X_tr, X_va, meta


# ------------------------------------------------------------------ metrics
def service_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred),
            "MAE_log": mean_absolute_error(np.log(y_true), np.log(np.clip(y_pred, 0.5, None)))}


def expected_calibration_error(y_true, p, bins: int = 10) -> float:
    y_true, p = np.asarray(y_true, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return ece


def late_metrics(y_true, p) -> dict:
    y_true = np.asarray(y_true, int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"logloss": log_loss(y_true, p, labels=[0, 1]),
            "brier": brier_score_loss(y_true, p),
            "AUC": roc_auc_score(y_true, p) if 0 < y_true.mean() < 1 else np.nan,
            "ECE": expected_calibration_error(y_true, p),
            "mean_pred": p.mean(), "rate": y_true.mean()}
