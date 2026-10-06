"""Shared helpers for Task 1 (service time + lateness).

Import from a notebook in notebooks/task1/:
    import sys; sys.path.append("../../tools/task1")
    from task1_common import *
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---- paths (resolved from the repo root, so notebooks can run from anywhere) ----
ROOT = Path(__file__).resolve().parents[2]          # tools/task1/ -> repo root
RAW = ROOT / "data" / "raw"                         # shared by all tasks, never modified
INTERIM = ROOT / "data" / "interim" / "task1"
PROCESSED = ROOT / "data" / "processed" / "task1"
METRICS = ROOT / "reports" / "metrics" / "task1"
FIGURES = ROOT / "reports" / "figures" / "task1"
SUBMISSION_DIR = ROOT / "HackBots_Datathon_Submision"
MODELS = SUBMISSION_DIR / "models"
PREDICTIONS = SUBMISSION_DIR / "predictions"

SEED = 42
TASK1_COLS = ["delivery_id", "pred_service_min", "pred_late_prob"]


# ---- time helpers ----
def hhmm_to_min(s: pd.Series) -> pd.Series:
    """'HH:MM' -> minutes since midnight (NaN stays NaN). Clock times only, no date."""
    parts = s.astype("string").str.split(":", expand=True)
    return pd.to_numeric(parts[0]) * 60 + pd.to_numeric(parts[1])


def add_day_offset(minutes: pd.Series, group_keys: list, df: pd.DataFrame) -> pd.Series:
    """Unwrap clock times that cross midnight within a group (e.g. a route).

    Expects `minutes` to be ordered along each group's route; any decrease versus the
    previous value is treated as a +1440 day rollover.
    """
    prev = minutes.groupby([df[k] for k in group_keys]).shift()
    rollover = (minutes < prev).astype(int)
    return minutes + 1440 * rollover.groupby([df[k] for k in group_keys]).cumsum()


# ---- labels (training only: they need actual times) ----
def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Task 1 labels from a joined train table that has the *_m minute columns.

    service_min = leave_outlet - max(arrival, window_open)   (early vehicles wait; the wait is not handling)
    late        = arrival > window_close                      (strictly after the window closes)
    wait_min    = max(window_open - arrival, 0)               (kept for diagnostics only)
    """
    arrival, opens = df["arrival_time_m"], df["window_open_time_m"]
    return pd.DataFrame({
        "service_min": df["leave_outlet_time_m"] - np.maximum(arrival, opens),
        "late": (arrival > df["window_close_time_m"]).astype(int),
        "wait_min": (opens - arrival).clip(lower=0),
    }, index=df.index)


# ---- loaders ----
def load_raw(name: str, subdir: str) -> pd.DataFrame:
    return pd.read_csv(RAW / subdir / name)


def load_task1_tables() -> dict:
    """All tables needed for Task 1, keyed by short name."""
    return {
        "deliveries_train": load_raw("deliveries_train.csv", "train"),
        "legs_train": load_raw("route_legs_train.csv", "train"),
        "orders_test": load_raw("task1_test_inputs.csv", "test"),
        "legs_test": load_raw("route_legs_test.csv", "test"),
        "outlets": load_raw("outlets.csv", "reference"),
        "calendar": load_raw("calendar.csv", "reference"),
        "service_allowance": load_raw("service_allowance.csv", "reference"),
        "traffic": load_raw("traffic_speed.csv", "reference"),
        "roads": load_raw("road_conditions.csv", "reference"),
        "vehicles": load_raw("vehicles.csv", "reference"),
        "district_travel": load_raw("district_travel.csv", "reference"),
        "template": load_raw("submission_task1.csv", "templates"),
    }


def join_orders_to_legs(orders: pd.DataFrame, legs: pd.DataFrame) -> pd.DataFrame:
    """Join dispatched orders to their route leg on (route_id, seq_in_route == seq).

    Order columns win on name clashes; leg-only columns get suffix '_leg'.
    """
    dispatched = orders.dropna(subset=["route_id"])
    return dispatched.merge(
        legs, left_on=["route_id", "seq_in_route"], right_on=["route_id", "seq"],
        how="left", suffixes=("", "_leg"), validate="one_to_one",
    )


# ---- submission validation ----
def validate_submission(sub: pd.DataFrame, template: pd.DataFrame) -> None:
    """Raise AssertionError if the Task 1 submission breaks the template contract."""
    assert list(sub.columns) == TASK1_COLS, f"columns must be {TASK1_COLS}"
    assert len(sub) == len(template), f"row count {len(sub)} != {len(template)}"
    assert (sub["delivery_id"].values == template["delivery_id"].values).all(), \
        "delivery_id values/order differ from the template"
    assert sub[TASK1_COLS[1:]].notna().all().all(), "NaN in prediction columns"
    assert np.isfinite(sub[TASK1_COLS[1:]].to_numpy(dtype=float)).all(), "non-finite values"
    assert (sub["pred_service_min"] > 0).all(), "pred_service_min must be positive"
    assert sub["pred_late_prob"].between(0, 1).all(), "pred_late_prob must be in [0, 1]"
