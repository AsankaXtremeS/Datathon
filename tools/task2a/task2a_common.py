"""Shared helpers for Task 2A (weekly depot x brand demand forecast).

Import from a notebook in notebooks/task2a/:
    import sys; sys.path.append("../../tools/task2a")
    from task2a_common import *
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]            # tools/task2a/ -> repo root
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim" / "task2a"
METRICS = ROOT / "reports" / "metrics" / "task2a"
FIGURES = ROOT / "reports" / "figures" / "task2a"
SUBMISSION_DIR = ROOT / "HackBots_Datathon_Submision"
MODELS = SUBMISSION_DIR / "models"
PREDICTIONS = SUBMISSION_DIR / "predictions"

DEPOTS = ("Kandy", "Peliyagoda")
BRANDS = ("Fresh", "Style", "Tech")
SERIES = [(d, b) for d in DEPOTS for b in BRANDS]
FEST_TYPES = ["new_year", "vesak", "poson", "esala", "deepavali", "christmas", "thai_pongal"]
SEED = 42

# Rolling-origin design: Mondays every 2 weeks; each origin forecasts the next 10 ISO weeks (the real horizon).
# The last origin must leave 10 full weeks of history before 2026-03-28.
ORIGINS = pd.date_range("2024-10-07", "2026-01-19", freq="14D")
HORIZON_WEEKS = 10


def load_orders():
    """Train + Task 1 test orders stacked (every order once, incl. deferred/not_run) and the calendar."""
    tr = pd.read_csv(RAW / "train" / "deliveries_train.csv")
    te = pd.read_csv(RAW / "test" / "task1_test_inputs.csv")
    cal = pd.read_csv(RAW / "reference" / "calendar.csv", parse_dates=["date"])
    o = pd.concat([tr.assign(src="train"), te.assign(src="test")], ignore_index=True)
    o["date"] = pd.to_datetime(o.order_date)
    o["ch_m3"] = np.where(o.temp_requirement == "chilled", o.order_volume_m3, 0.0)
    return o, cal


def metric_row(actual, pred):
    """MAE, RMSE, WAPE and signed bias (fraction of actual) on weekly depot x brand totals."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    e = p - a
    return dict(n=len(a), MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), WAPE=np.abs(e).sum() / a.sum(), bias=e.sum() / a.sum())
