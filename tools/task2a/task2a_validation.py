"""Rolling-origin validation for Task 2A (forward in time, the real 10-week horizon)."""
import numpy as np
import pandas as pd

from task2a_common import HORIZON_WEEKS, ORIGINS, metric_row


def backtest(panel, brand, make, origins=ORIGINS, name=None, target="vol"):
    """For each origin: fit on days before the cut, forecast the next 10 weeks, aggregate to ISO weeks per depot.

    `make` is a zero-argument callable returning a fresh model (so nothing leaks between origins).
    Returns one row per origin x depot x ISO week, with the forecast horizon (1..10 weeks ahead).
    """
    x = panel[panel.brand == brand]
    rows = []
    for cut in origins:
        end = cut + pd.Timedelta(weeks=HORIZON_WEEKS)
        trn, fut = x[x.date < cut], x[(x.date >= cut) & (x.date < end)]
        m = make().fit(trn)
        p = m.predict(fut, cut)
        w = fut.assign(pred=p, k=fut.iso_year * 100 + fut.iso_week).groupby(["depot", "k"]).agg(
            actual=(target, "sum"), pred=("pred", "sum"), n_days=("date", "size"), ramp=("festival_ramp", "mean"), first=("date", "min"))
        w = w.reset_index()
        w["horizon"] = ((w["first"] - cut).dt.days // 7 + 1).clip(1, HORIZON_WEEKS)
        w["cut"], w["brand"], w["model"] = cut, brand, name or type(m).__name__
        rows.append(w)
    return pd.concat(rows, ignore_index=True)


def summarize(bt, by=("model", "brand")):
    return bt.groupby(list(by)).apply(lambda g: pd.Series(metric_row(g.actual, g.pred))).round(4).reset_index()


def festival_split(bt, thr=0.4):
    """Same metrics split into festival run-up weeks (mean ramp > thr) and other weeks."""
    b = bt.assign(bucket=np.where(bt.ramp > thr, "festival run-up weeks", "other weeks"))
    return summarize(b, by=("brand", "bucket", "model"))
