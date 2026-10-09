"""Calendar features and the daily demand panel for Task 2A."""
import numpy as np
import pandas as pd

from task2a_common import BRANDS, DEPOTS, FEST_TYPES, SERIES

DOW = [f"d{i}" for i in range(6)]                       # Mon..Sat (Sunday never operates)
BASE_COLS = DOW + ["is_payday", "festival_ramp", "is_holiday"]
RAMP_COLS = [f"r_{t}" for t in FEST_TYPES]
GB_COLS = ["dow", "iso_week", "dtn", "dsl", "festival_ramp", "is_payday"]


def enrich_calendar(cal):
    """Add everything that is known in advance for any date:

    * dow dummies, years since 2024-01-01 (`t`)
    * `nf`: name of the next festival (including today); `dtn`/`dsl`: days to next / since last festival (capped at 30)
    * `r_<festival>`: festival_ramp, but only on the run-up to that specific festival. Festivals differ in size
      (the Sinhala New Year run-up is far larger than Vesak or Poson), so each gets its own coefficient.
    """
    c = cal.sort_values("date").reset_index(drop=True).copy()
    fc = c[c.festival.notna() & (c.festival != "")][["date", "festival"]].rename(columns={"date": "fd"})
    nxt = pd.merge_asof(c[["date"]], fc, left_on="date", right_on="fd", direction="forward")
    prv = pd.merge_asof(c[["date"]], fc.rename(columns={"fd": "pd_"}), left_on="date", right_on="pd_", direction="backward")
    c["nf"] = nxt.festival.values
    c["dtn"] = np.minimum((nxt.fd - c.date).dt.days.fillna(99).values, 30)
    c["dsl"] = np.minimum((c.date - prv.pd_).dt.days.fillna(99).values, 30)
    for t in FEST_TYPES:
        c[f"r_{t}"] = c.festival_ramp.values * (c.nf.values == t)
    # closures: how many Mon-Sat days of the ISO week are closed, and was the previous Mon-Sat day closed?
    mon_sat = c.dow < 6
    closed = (c.is_operating == 0) & mon_sat
    c["closed_wk"] = closed.groupby([c.iso_year, c.iso_week]).transform("sum").astype(float)
    prev_closed = []
    d2c = dict(zip(c.date, closed))
    for d in c.date:
        q = d - pd.Timedelta(days=1)
        if q.dayofweek == 6:
            q -= pd.Timedelta(days=1)
        prev_closed.append(float(d2c.get(q, False)))
    c["post_gap"] = prev_closed
    c["t"] = (c.date - pd.Timestamp("2024-01-01")).dt.days / 365.0
    for i in range(6):
        c[f"d{i}"] = (c.dow == i).astype(float)
    return c


def build_panel(o, cal2):
    """Daily panel: one row per OPERATING day x depot x brand with total/chilled volume (0 on days without orders).

    Orders from `task1_test_inputs` were all dispatched, so `not_run` orders are missing from 2026-02-16 on; those days
    are scaled by 1 / (1 - train not_run share of volume) per series.
    """
    tr = o[o.src == "train"]
    tot = tr.groupby(["depot", "brand"]).order_volume_m3.sum()
    nr = tr[tr.dispatch_status == "not_run"].groupby(["depot", "brand"]).order_volume_m3.sum()
    gap_start = o[o.src == "test"].date.min()
    last = o.date.max()
    g = o.groupby(["depot", "brand", "date"]).agg(vol=("order_volume_m3", "sum"), ch=("ch_m3", "sum"), n_orders=("delivery_id", "size")).reset_index()
    ops = cal2[(cal2.is_operating == 1) & (cal2.date <= last)]
    rows = []
    for dep, b in SERIES:
        x = ops.merge(g[(g.depot == dep) & (g.brand == b)], on="date", how="left").fillna({"vol": 0.0, "ch": 0.0, "n_orders": 0})
        x[["vol", "ch", "n_orders"]] = x[["vol", "ch", "n_orders"]].astype(float)
        up = 1.0 / (1.0 - nr.get((dep, b), 0.0) / tot[(dep, b)])
        m = x.date >= gap_start
        x.loc[m, ["vol", "ch", "n_orders"]] *= up
        x["depot"], x["brand"] = dep, b
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


def not_run_share(o):
    tr = o[o.src == "train"]
    tot = tr.groupby(["depot", "brand"]).order_volume_m3.sum()
    nr = tr[tr.dispatch_status == "not_run"].groupby(["depot", "brand"]).order_volume_m3.sum()
    return (nr / tot).reindex(tot.index).fillna(0.0)


def future_ops(cal2, panel):
    """Operating days after the last observed day (what the forecast is summed over)."""
    return cal2[(cal2.is_operating == 1) & (cal2.date > panel.date.max())].copy()
