"""Task 2A models. Every model forecasts DAILY mean volume on future operating days; weekly totals are sums of those.

Interface: model.fit(trn) ; model.predict(fut, cut) -> array aligned with `fut`.
`trn` = history rows (before the cut-off), `fut` = future operating-day rows, `cut` = first forecast day.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from task2a_common import DEPOTS, SEED
from task2a_features import BASE_COLS, DOW, GB_COLS, RAMP_COLS

GB_KW = dict(max_depth=3, max_iter=60, learning_rate=0.05, min_samples_leaf=20, random_state=SEED)


def trend_future(dates, cut, t_last, phi):
    """Trend position (years) of future days. phi=1: full trend, phi<1: damped (weekly decay), phi=0: hold last level."""
    wk = np.ceil(((dates - cut).dt.days.values + 1) / 7.0)
    if phi >= 1:
        return t_last + wk / 52.0
    return t_last + np.array([sum(phi ** k for k in range(1, int(w) + 1)) for w in wk]) / 52.0


class DailyRidgeGB:
    """Ridge on log daily volume (+ optional boosting on its log residuals), forecast = lognormal MEAN.

    pooled=True : one model for both depots (depot x weekday intercepts, depot trend; calendar effects shared) - the
                  festival effects are estimated from twice the data.
    extra       : extra design columns, e.g. festival-specific ramps.
    trend       : 'none' | 'full' | 'damped' (phi per week).
    """

    def __init__(self, extra=(), pooled=True, gb=True, trend="damped", phi=0.9, alpha=1e-2, depot_slopes=False, gb_kw=None):
        self.extra, self.pooled, self.gb, self.trend, self.phi = tuple(extra), pooled, gb, trend, phi
        self.alpha, self.depot_slopes, self.gb_kw = alpha, depot_slopes, dict(GB_KW, **(gb_kw or {}))

    # -- design -------------------------------------------------------------------------------------------------
    def _X(self, x, tt):
        X = x[BASE_COLS + list(self.extra)].astype(float).copy()
        X["ramp2"] = x.festival_ramp.values ** 2
        if self.trend != "none":
            X["t"] = tt
        if self.pooled:
            dep = (x.depot == "Peliyagoda").values.astype(float)
            X["dep"] = dep
            for c in DOW + (["t"] if self.trend != "none" else []):
                X[c + "_dep"] = X[c] * dep
            if self.depot_slopes:
                for c in ["festival_ramp", "is_payday"] + list(self.extra):
                    X[c + "_dep"] = X[c] * dep
        return X

    def _G(self, x):
        G = x[GB_COLS].astype(float).copy()
        if self.pooled:
            G["dep"] = (x.depot == "Peliyagoda").values.astype(float)
        return G

    # -- fit / predict --------------------------------------------------------------------------------------------
    def fit(self, trn):
        if not self.pooled:
            self.sub_ = {d: DailyRidgeGB(self.extra, True, self.gb, self.trend, self.phi, self.alpha, False, self.gb_kw).fit(trn[trn.depot == d]) for d in trn.depot.unique()}
            return self
        self.t_last_ = float(trn.t.max())
        y = np.log(np.clip(trn.vol.values, 1e-3, None))
        self.ridge_ = Ridge(alpha=self.alpha).fit(self._X(trn, trn.t.values), y)
        res = y - self.ridge_.predict(self._X(trn, trn.t.values))
        self.resid_var_ = pd.Series(res).groupby(trn.depot.values).var().to_dict()
        self.gb_model_ = HistGradientBoostingRegressor(**self.gb_kw).fit(self._G(trn), res) if self.gb else None
        return self

    def predict(self, fut, cut):
        if not self.pooled:
            out = np.zeros(len(fut))
            for d, m in self.sub_.items():
                mask = (fut.depot == d).values
                if mask.any():
                    out[mask] = m.predict(fut[mask], cut)
            return out
        phi = 1.0 if self.trend == "full" else self.phi
        tt = trend_future(fut.date, pd.Timestamp(cut), self.t_last_, phi) if self.trend != "none" else None
        z = self.ridge_.predict(self._X(fut, tt))
        if self.gb_model_ is not None:
            z = z + self.gb_model_.predict(self._G(fut))
        return np.exp(z + fut.depot.map(self.resid_var_).values / 2)


class TechRidge:
    """Tech: per-depot ridge on volume with weekday, payday, ramp (+ trend). Tech orders follow a depot-specific weekday
    schedule but are otherwise noisy, so the model is deliberately small and strongly regularised."""

    def __init__(self, alpha=30.0, trend=True, phi=0.9, cols=None):
        self.alpha, self.trend, self.phi = alpha, trend, phi
        self.cols = list(cols) if cols is not None else DOW + ["is_payday", "festival_ramp"]

    def _X(self, x, tt):
        X = x[self.cols].astype(float).copy()
        if self.trend:
            X["t"] = tt
        return X

    def fit(self, trn):
        self.m_, self.t_last_ = {}, {}
        for d in trn.depot.unique():
            t = trn[trn.depot == d]
            self.m_[d] = Ridge(alpha=self.alpha).fit(self._X(t, t.t.values), t.vol.values)
            self.t_last_[d] = float(t.t.max())
        return self

    def predict(self, fut, cut):
        out = np.zeros(len(fut))
        for d, m in self.m_.items():
            mask = (fut.depot == d).values
            if not mask.any():
                continue
            f = fut[mask]
            tt = trend_future(f.date, pd.Timestamp(cut), self.t_last_[d], self.phi) if self.trend else None
            out[mask] = np.clip(m.predict(self._X(f, tt)), 0, None)
        return out


class NaiveRecent:
    """Baseline: mean daily volume of the last `weeks` weeks, per depot, repeated on every future operating day."""

    def __init__(self, weeks=8):
        self.weeks = weeks

    def fit(self, trn):
        self.mu_ = trn.groupby("depot").apply(lambda g: g.tail(self.weeks * 6).vol.mean()).to_dict()
        return self

    def predict(self, fut, cut):
        return fut.depot.map(self.mu_).values


class SeasonalNaive:
    """Baseline: same ISO week of the previous year (summed over its operating days) spread over this week's operating
    days, scaled by the recent year-over-year growth. Captures festival weeks that move with the calendar only roughly."""

    def fit(self, trn):
        self.trn_ = trn
        w = trn.assign(k=trn.iso_year * 100 + trn.iso_week).groupby(["depot", "k"]).vol.sum()
        self.wk_ = w
        last = trn.date.max()
        a = trn[trn.date > last - pd.Timedelta(weeks=26)].groupby("depot").vol.sum()
        b = trn[(trn.date > last - pd.Timedelta(weeks=78)) & (trn.date <= last - pd.Timedelta(weeks=52))].groupby("depot").vol.sum()
        nb = trn[(trn.date > last - pd.Timedelta(weeks=78)) & (trn.date <= last - pd.Timedelta(weeks=52))].groupby("depot").size()
        g = (a / b).clip(0.8, 1.3)
        # without a full 26-week comparison window a year ago there is no growth estimate: assume none
        self.growth_ = {d: (float(g[d]) if (d in b.index and nb[d] >= 26 * 6 * 3) else 1.0) for d in trn.depot.unique()}
        return self

    def predict(self, fut, cut):
        out = np.zeros(len(fut))
        for (dep, yr, wk), idx in fut.groupby(["depot", "iso_year", "iso_week"]).groups.items():
            base = self.wk_.get((dep, (yr - 1) * 100 + wk), np.nan)
            if np.isnan(base):
                base = self.trn_[self.trn_.depot == dep].tail(48).vol.mean() * len(idx)
            pos = fut.index.get_indexer(idx)
            out[pos] = base * self.growth_[dep] / len(idx)
        return out


class ChilledModel:
    """Fresh chilled volume. mode: 'share' = total forecast x depot chilled share; 'direct' = same architecture fitted on
    chilled volume; 'blend' = w * direct + (1 - w) * share. Always capped at the total forecast."""

    def __init__(self, make_total, mode="blend", w=0.5):
        self.make_total, self.mode, self.w = make_total, mode, w

    def fit(self, trn):
        self.total_ = self.make_total().fit(trn)
        self.make_total = None  # a lambda cannot be pickled; the fitted model is all we need
        self.share_ = (trn.groupby("depot").ch.sum() / trn.groupby("depot").vol.sum()).to_dict()
        self.direct_ = DailyRidgeGB(PROD_EXTRA, pooled=True, gb=True, trend="damped").fit(trn.assign(vol=trn.ch)) if self.mode != "share" else None
        return self

    def predict(self, fut, cut):
        tot = self.total_.predict(fut, cut)
        share = tot * fut.depot.map(self.share_).values
        if self.mode == "share":
            return share
        direct = self.direct_.predict(fut, cut)
        c = direct if self.mode == "direct" else self.w * direct + (1 - self.w) * share
        return np.minimum(c, tot)


# ---------------------------------------------------------------------------------------------- final bundle
PROD_EXTRA = tuple(RAMP_COLS)


def make_model(brand):
    """The production model per brand (chosen in notebooks 05-05c)."""
    if brand == "Tech":
        return TechRidge()
    # Style: the rolling backtest prefers NO trend (lower MAE, RMSE and WAPE); Fresh: damped trend (full is a statistical tie)
    return DailyRidgeGB(extra=PROD_EXTRA, pooled=True, gb=True, trend="none" if brand == "Style" else "damped", phi=0.9)


class Forecaster:
    """Fitted production system: one total-volume model per brand + a chilled model for Fresh."""

    def fit(self, panel):
        self.models_ = {b: make_model(b).fit(panel[panel.brand == b]) for b in sorted(panel.brand.unique())}
        self.chilled_ = ChilledModel(lambda: make_model("Fresh"), mode="blend", w=0.5).fit(panel[panel.brand == "Fresh"])
        self.cut_ = panel.date.max() + pd.Timedelta(days=1)
        return self

    def daily(self, fut):
        """Daily forecast for every row of `fut` (needs depot, brand + feature columns). Style/Tech chilled = 0."""
        out = fut.copy()
        out["pred_total"] = np.nan
        out["pred_chilled"] = 0.0
        for b, m in self.models_.items():
            mask = out.brand == b
            if not mask.any():
                continue
            out.loc[mask, "pred_total"] = m.predict(out[mask], self.cut_)
            if b == "Fresh":
                out.loc[mask, "pred_chilled"] = self.chilled_.predict(out[mask], self.cut_)
        return out

    def weekly(self, fut_ops, test_inputs):
        """Fill the Task 2A template: sum daily forecasts over the OPERATING days of each depot x brand x ISO week."""
        rows = []
        for _, r in test_inputs.iterrows():
            f = fut_ops[(fut_ops.iso_year == r.iso_year) & (fut_ops.iso_week == r.iso_week)].assign(depot=r.depot, brand=r.brand)
            d = self.daily(f)
            rows.append((d.pred_total.sum(), d.pred_chilled.sum()))
        out = test_inputs.copy()
        out["pred_total_volume_m3"] = np.round([r[0] for r in rows], 3)
        out["pred_chilled_volume_m3"] = np.round([r[1] for r in rows], 3)
        return out
