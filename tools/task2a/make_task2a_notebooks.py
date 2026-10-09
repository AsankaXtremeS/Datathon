"""Generate the Task 2A notebooks (01 ... 06) into notebooks/task2a/.

Run from the repo root:  python tools/task2a/make_task2a_notebooks.py
Then execute them in order (jupyter nbconvert --execute --inplace).
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "task2a"

HEADER = """import sys, warnings; warnings.filterwarnings('ignore')
sys.path.append('../../tools/task2a')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib, json
from task2a_common import *
from task2a_features import *
from task2a_models import *
from task2a_validation import *
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 100)
plt.rcParams.update({'figure.dpi': 90, 'axes.grid': True, 'grid.alpha': .25})
for d in (INTERIM, METRICS, FIGURES): d.mkdir(parents=True, exist_ok=True)
o, cal = load_orders(); cal2 = enrich_calendar(cal); panel = build_panel(o, cal2)
"""


def nb(name, cells):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nbf.write(n, OUT / name)


M = lambda s: ("md", s)
C = lambda s: ("code", s)

# ------------------------------------------------------------------------------------------------------ 01
nb("01_data_audit.ipynb", [
    M("""# Task 2A - 01 Data audit
**Question:** can the supplied files be turned into a clean weekly depot x brand demand history, and what does the 10-week forecast window look like?

Booklet rules we audit against: every order counts once (incl. `deferred`, `not_run`); week = week the store *requested* the order (`order_date`); weeks come from `calendar.csv` ISO fields; only Fresh has chilled demand."""),
    C(HEADER),
    M("## 1. Sources and coverage"),
    C("""print(o.groupby('src').agg(rows=('delivery_id', 'size'), first=('date', 'min'), last=('date', 'max'), days=('date', 'nunique')))
print('duplicate delivery_id:', o.delivery_id.duplicated().sum())
print('train/test overlap days:', len(set(o[o.src == 'train'].date) & set(o[o.src == 'test'].date)))
print(pd.crosstab(o.src, o.dispatch_status))
last_train, first_test = o[o.src == 'train'].date.max(), o[o.src == 'test'].date.min()
gap = cal[(cal.date > last_train) & (cal.date < first_test)][['date', 'dow_name', 'is_operating']]
print('\\ncalendar days between train end and test start (must be non-operating):'); print(gap)"""),
    M("""**Finding.** Train ends 2026-02-14 (Sat) and the Task 1 test file starts 2026-02-16 (Mon); the only day between is a Sunday. Stacking the two gives a gap-free history up to 2026-03-28 = end of ISO week 13, so the forecast (weeks 14-23) starts immediately after the data. The test file has **no `not_run` orders** (it only holds dispatched orders), which biases its weeks slightly low - handled in notebook 02."""),
    M("## 2. Orders vs operating days"),
    C("""m = o.merge(cal[['date', 'is_operating', 'dow', 'iso_year', 'iso_week']], on='date', how='left')
print('orders with no calendar row:', m.is_operating.isna().sum())
print('orders on NON-operating days:', int((m.is_operating == 0).sum()), '| on Sundays:', int((m.dow == 6).sum()))
ops = cal2[(cal2.is_operating == 1) & (cal2.date <= o.date.max())]
z = []
for dep, b in SERIES:
    have = set(o[(o.depot == dep) & (o.brand == b)].date)
    z.append((dep, b, len(ops), int(ops.date.isin(have).sum()), int((~ops.date.isin(have)).sum())))
print(pd.DataFrame(z, columns=['depot', 'brand', 'operating_days', 'days_with_orders', 'days_without_orders']))"""),
    M("""**Finding.** No order falls on a non-operating day and nothing is shifted onto the next operating day (checked in notebook 03), so forecasts must be produced for *operating days only* and summed. Fresh orders every operating day; Style and Tech have days without orders, which are genuine zeros, not missing data."""),
    M("## 3. The order schedule is (almost) deterministic"),
    C("""d = o.groupby(['depot', 'brand', 'date']).size().reset_index(name='n'); d['dow'] = d.date.dt.dayofweek
print('orders per day by weekday (mean over days with at least one order):')
print(d.groupby(['depot', 'brand', 'dow']).n.mean().unstack().round(1))
print('\\nstd of orders/day for Fresh (tiny -> fixed schedule):'); print(d[d.brand == 'Fresh'].groupby(['depot', 'dow']).n.std().unstack().round(2))
s = o[o.brand != 'Fresh'].assign(dow=lambda x: x.date.dt.dayofweek)
print('\\nshare of orders by weekday - Style/Tech follow depot-specific weekdays:'); print(pd.crosstab([s.brand, s.depot], s.dow, normalize='index').round(2))"""),
    M("""**Finding.** Order *counts* are fixed by weekday (Fresh Peliyagoda: 79/88/77/87/75/84; std < 1). Style and Tech only order on certain weekdays per depot (e.g. Kandy Tech on Wed-Fri, Peliyagoda Tech on Tue/Fri/Sat). So a 4-day holiday week loses *specific* weekdays, and the volume model needs weekday terms. The uncertain part of demand is order **size**."""),
    M("## 4. The forecast window (ISO weeks 14-23 of 2026)"),
    C("""w = cal2[(cal2.iso_year == 2026) & cal2.iso_week.between(14, 23)]
tbl = w.groupby('iso_week').agg(first_day=('date', 'min'), operating_days=('is_operating', 'sum'), paydays=('is_payday', 'sum'), holidays=('is_holiday', 'sum'),
                                 mean_ramp=('festival_ramp', 'mean'), festival=('festival', lambda s: ','.join(s.dropna())))
tbl.round(2)"""),
    M("""**Finding.** The window is festival-heavy: week 15 is the Sinhala New Year run-up, week 16 is New Year week with only **4 operating days**, week 18 contains Vesak (May 1, **5 operating days**), week 22 has Poson plus two paydays. A model that ignores the calendar will be wrong exactly where it matters."""),
    M("## 5. Weekly history per series"),
    C("""wk = panel.assign(k=panel.iso_year * 100 + panel.iso_week).groupby(['depot', 'brand', 'k']).vol.sum().reset_index()
fig, axs = plt.subplots(2, 3, figsize=(15, 6))
for a, (dep, b) in zip(axs.ravel(), SERIES):
    g = wk[(wk.depot == dep) & (wk.brand == b)]
    a.plot(range(len(g)), g.vol.values, lw=1); a.set_title(f'{dep} {b} - weekly m3')
plt.tight_layout(); plt.savefig(FIGURES / '01_weekly_history.png'); plt.show()
print(wk.groupby(['depot', 'brand']).vol.describe().round(1))"""),
    M("""## Audit conclusions
1. Data are clean: no duplicate orders, every order maps to a calendar row, no order on a non-operating day, no gap between train and test files.
2. One known defect: the Task 1 test file lacks `not_run` orders - corrected in notebook 02.
3. Demand = fixed weekday order schedule x calendar-driven order size; Tech is sparse and noisy (many zero days), Style is weekly per outlet.
4. The 10-week window contains New Year, Vesak and Poson plus short weeks, so calendar handling is the core of this task."""),
])

# ------------------------------------------------------------------------------------------------------ 02
nb("02_target_construction.ipynb", [
    M("""# Task 2A - 02 Target construction
Defines **exactly** what the model learns from, and why. Weekly totals are built bottom-up from a **daily operating-day panel**, so that holidays and short weeks are structural, not noise."""),
    C(HEADER),
    M("""## 1. Target definition
| Target | Definition |
|---|---|
| `vol` (daily) | sum of `order_volume_m3` of all orders (any `dispatch_status`) with `order_date` = that day, per depot x brand |
| `ch` (daily) | same, only `temp_requirement == chilled` (Fresh only) |
| weekly target | sum of the daily values over the operating days of the ISO week (`iso_year`, `iso_week` from `calendar.csv`) |

**Why `order_date`, not `dispatch_date`?** Deferred orders are dispatched later, but they were demanded on `order_date` (booklet: "assign each order to the week the store requested it")."""),
    C("""chk = o.assign(od=o.date, dd=pd.to_datetime(o.dispatch_date))
cw = cal[['date', 'iso_year', 'iso_week']].assign(k=lambda x: x.iso_year * 100 + x.iso_week).set_index('date').k
chk['k_order'] = chk.od.map(cw); chk['k_dispatch'] = chk.dd.map(cw)
has = chk.dispatch_date.notna()
print('orders whose dispatch falls in a DIFFERENT ISO week than the order:', int((chk.k_order[has] != chk.k_dispatch[has]).sum()), 'of', int(has.sum()))
print(chk.groupby('dispatch_status').apply(lambda g: pd.Series({'n': len(g), 'vol_m3': g.order_volume_m3.sum()})).round(1))"""),
    M("""Using `dispatch_date` (or dropping `deferred`/`not_run`) would move or remove demand: the `deferred` and `not_run` orders in the table above still represent demand and are kept on their requested week."""),
    M("## 2. Chilled only exists for Fresh"),
    C("""print(o.groupby(['brand', 'temp_requirement']).order_volume_m3.sum().round(0).unstack().fillna(0))
assert o[(o.brand != 'Fresh') & (o.temp_requirement == 'chilled')].empty
f = panel[panel.brand == 'Fresh']
print('\\nchilled share of Fresh volume by depot:', (f.groupby('depot').ch.sum() / f.groupby('depot').vol.sum()).round(4).to_dict())
sh = f.assign(q=f.date.dt.to_period('Q')).groupby(['depot', 'q']).apply(lambda g: g.ch.sum() / g.vol.sum()).unstack(0)
print(sh.round(3).T)"""),
    M("""**Finding.** Style/Tech have no chilled orders (assert passes) so their chilled target is exactly 0. The Fresh chilled share is very stable over time (about 0.36-0.38 per depot), so chilled = total x share is justified and tested against a direct chilled model in 05."""),
    M("## 3. Correcting the missing `not_run` orders in the test file"),
    C("""nrs = not_run_share(o)
print('train not_run share of volume per series:'); print(nrs.round(4))
raw_tot = o.groupby(['depot', 'brand']).order_volume_m3.sum()
pan_tot = panel.groupby(['depot', 'brand']).vol.sum()
print('\\npanel total / raw total (>1 only through the gross-up of the 2026-02-16+ days):'); print((pan_tot / raw_tot).round(4))
rawwk = o[o.date >= '2026-02-16'].groupby(['depot', 'brand']).order_volume_m3.sum()
corr = panel[panel.date >= '2026-02-16'].groupby(['depot', 'brand']).vol.sum()
print('\\nvolume added on the test-period days (m3):'); print((corr - rawwk).round(2))"""),
    M("""The gross-up is small (largest for Kandy Tech, 2.6%) and applied **only** to days where the source file cannot contain `not_run` orders. It is derived from train, so no test labels are used."""),
    M("## 4. Reconciliation checks"),
    C("""# no operating day is lost or duplicated; panel covers every operating day up to the last observed day
ops = cal2[(cal2.is_operating == 1) & (cal2.date <= o.date.max())]
assert all(len(panel[(panel.depot == d) & (panel.brand == b)]) == len(ops) for d, b in SERIES)
assert panel.groupby(['depot', 'brand', 'date']).size().max() == 1
assert (panel.date.dt.dayofweek != 6).all() and panel.is_operating.eq(1).all()
# weekly totals equal the sum of the raw orders (before the gross-up) on untouched days
untouched = panel[panel.date < '2026-02-16'].groupby(['depot', 'brand']).vol.sum()
raw_u = o[o.date < '2026-02-16'].groupby(['depot', 'brand']).order_volume_m3.sum()
assert np.allclose(untouched, raw_u), 'panel does not reconcile with raw orders'
print('panel rows:', panel.shape, '| all reconciliation checks passed')
panel.to_pickle(INTERIM / 'daily_panel.pkl')
wk = panel.groupby(['depot', 'brand', 'iso_year', 'iso_week']).agg(total_m3=('vol', 'sum'), chilled_m3=('ch', 'sum'), operating_days=('date', 'size')).reset_index()
wk.to_csv(INTERIM / 'weekly_targets.csv', index=False); print('weekly targets:', wk.shape); wk.tail(6)"""),
    M("""## Decisions recorded
- Targets are built **daily on operating days** and summed; weekly series are a by-product.
- All orders count; week by `order_date`; chilled only for Fresh.
- Test-period `not_run` gap corrected with a train-derived factor (<= 2.6%).
- Leakage rule for all later work: a feature may be used only if it is known for the *future* date from `calendar.csv` (weekday, payday, festival ramp, holidays, festival distance). No lagged demand is used by the production model, so the 10-week horizon needs no recursive forecasting."""),
])

# ------------------------------------------------------------------------------------------------------ 03
nb("03_feature_engineering.ipynb", [
    M("""# Task 2A - 03 Feature engineering (evidence first)
Every feature is justified by an effect we can *see* in the history, and every one is known in advance from `calendar.csv`."""),
    C(HEADER),
    M("## 1. Order size is the moving part: festival run-up, by festival"),
    C("""f = panel[(panel.brand == 'Fresh') & (panel.depot == 'Peliyagoda')].copy()
f['rb'] = pd.cut(f.festival_ramp, [-.01, 0, .3, .6, .8, 1.0])
base = f[f.festival_ramp == 0].vol.mean()
rows = []
for t in FEST_TYPES:
    g = f[(f.nf == t) & (f.festival_ramp > 0.3)]
    rows.append((t, g.date.dt.year.nunique(), len(g), g.vol.mean() / base - 1, g[g.festival_ramp > .7].vol.mean() / base - 1))
r = pd.DataFrame(rows, columns=['festival', 'years_seen', 'run_up_days', 'uplift_ramp>0.3', 'uplift_ramp>0.7']).round(3)
print('baseline (ramp=0) mean daily volume: %.1f m3' % base); r"""),
    M("""**Finding.** Festivals are **not** equal: the Sinhala New Year run-up lifts Fresh daily volume far more than Vesak, Poson or Esala. One shared `festival_ramp` coefficient averages them and under-forecasts the New Year (the biggest event in our window). Hence one ramp column **per festival** (`r_new_year`, `r_vesak`, ...); the ridge penalty keeps the rarely-seen ones stable."""),
    C("""fig, axs = plt.subplots(1, 2, figsize=(14, 3.8))
for yr, col in [(2024, 'C0'), (2025, 'C1')]:
    g = f[(f.date >= f'{yr}-03-28') & (f.date <= f'{yr}-04-26')]
    axs[0].plot(g.date.dt.strftime('%m-%d'), g.vol, marker='o', ms=3, label=f'New Year {yr}', color=col)
axs[0].legend(); axs[0].tick_params(axis='x', rotation=90); axs[0].set_title('Peliyagoda Fresh daily m3: run-up then holiday gap')
for t, g in f[f.festival_ramp > 0.05].groupby('nf'):
    m = g.groupby(pd.cut(g.festival_ramp, [0, .3, .5, .7, .9, 1.0])).vol.mean()
    axs[1].plot(range(len(m)), m.values, marker='o', label=t)
axs[1].legend(fontsize=7); axs[1].set_title('mean daily m3 by ramp bucket, per festival')
plt.tight_layout(); plt.savefig(FIGURES / '03_festival_effects.png'); plt.show()"""),
    M("## 2. Weekday, payday, trend, monsoon"),
    C("""g = panel[panel.brand == 'Fresh']
print('daily m3 by weekday (Mon..Sat):'); print(g.groupby(['depot', 'dow']).vol.mean().unstack().round(0))
print('\\npayday uplift (payday / non-payday mean daily volume):'); print((g.groupby(['depot', 'is_payday']).vol.mean().unstack().pipe(lambda d: d[1] / d[0])).round(3))
print('\\nmonsoon uplift:'); print((g.groupby(['depot', 'monsoon']).vol.mean().unstack().pipe(lambda d: d[1] / d[0])).round(3))
yr = g[g.festival_ramp == 0].groupby(['depot', g.date.dt.year]).vol.mean().unstack(); print('\\nmean daily m3 outside run-ups by year:'); print(yr.round(1)); print('growth 2025->2026:', (yr[2026] / yr[2025] - 1).round(3).to_dict())"""),
    M("""**Findings.** Fri/Sat orders are larger; paydays add about 10-12%; monsoon has **no** demand effect (so it is not a feature); there is steady growth of a few % per year for Fresh, handled with a **damped trend**; trend options (none / full / damped) are compared per brand in 05c."""),
    M("## 3. Is there catch-up demand after a non-operating day?"),
    C("""gg = panel[(panel.brand == 'Fresh') & (panel.depot == 'Peliyagoda')].copy()
gg['prev_gap'] = (gg.date - gg.date.shift(1)).dt.days
after = gg[(gg.prev_gap > 1) & (gg.date.dt.dayofweek != 0)]          # day after a non-Sunday gap
norm = gg[(gg.prev_gap == 1)]
print('days directly after a holiday gap:', len(after), '| mean vol %.1f vs %.1f on ordinary days' % (after.vol.mean(), norm.vol.mean()))
print('orders/day after gap %.1f vs %.1f' % (after.n_orders.mean(), norm.n_orders.mean()))"""),
    M("""**Finding.** The day after a holiday gap looks like any other day (same order counts and sizes). Closed days are *lost* demand, not postponed, so weekly totals shrink proportionally in short weeks and no catch-up feature is needed."""),
    M("## 4. Tech: a weekday schedule plus noise"),
    C("""t = panel[panel.brand == 'Tech']
print('Tech mean daily m3 by weekday:'); print(t.groupby(['depot', 'dow']).vol.mean().unstack().round(1))
print('\\nshare of operating days with zero Tech volume:', (t.vol == 0).groupby(t.depot).mean().round(2).to_dict())
print('payday uplift:', (t.groupby(['depot', 'is_payday']).vol.mean().unstack().pipe(lambda d: d[1] / d[0])).round(2).to_dict())"""),
    M("""## Final feature set
| Group | Features | Used by |
|---|---|---|
| Weekday | `d0..d5` (Sunday never operates) | all |
| Cash cycle | `is_payday` | all |
| Festivals | `festival_ramp`, `festival_ramp^2`, `is_holiday`, one `r_<festival>` ramp per festival | Fresh, Style; Tech uses ramp only |
| Trend | years since 2024-01-01: damped (phi 0.9 per week) for Fresh and Tech, **none** for Style (chosen in 05c) | Fresh, Tech |
| Residual learner inputs | weekday, ISO week, days to next / since last festival, ramp, payday | boosting on ridge residuals |
| Depot | depot x weekday intercepts, depot trend (pooled model) | Fresh, Style |

Deliberately **not** used: monsoon (no effect), lagged demand (not known 10 weeks ahead), outlet counts (fixed).
Calendar features are saved in `data/processed` terms by `enrich_calendar`; the feature list is also written to `reports/metrics/task2a/task2a_feature_meta.json`."""),
    C("""meta = {'base_cols': BASE_COLS, 'festival_ramp_cols': RAMP_COLS, 'gb_cols': GB_COLS, 'trend': 'damped phi=0.9/week', 'targets': ['vol', 'ch']}
json.dump(meta, open(METRICS / 'task2a_feature_meta.json', 'w'), indent=1); meta"""),
])

# ------------------------------------------------------------------------------------------------------ 04
nb("04_validation_baselines.ipynb", [
    M("""# Task 2A - 04 Validation design and baselines
## Validation design
The real task: train on everything up to ISO week 13, forecast the next **10 weeks**. We reproduce that **forward in time**:
- 34 rolling origins (Mondays, every 2 weeks, 2024-10-07 ... 2026-01-12); at each, train only on days *before* the cut, forecast the following 10 ISO weeks;
- score weekly depot x brand totals (680 weekly forecasts per brand) with **MAE, RMSE, WAPE and bias** so conclusions do not depend on one metric (the official metric is not stated);
- origins before spring 2025 can only learn from one festival season, so results there are pessimistic - the final model has two seasons of each festival.
No random folds are used: they would leak neighbouring days of the same festival run-up."""),
    C(HEADER),
    C("""print(len(ORIGINS), 'origins:', ORIGINS[0].date(), '->', ORIGINS[-1].date(), '| horizon', HORIZON_WEEKS, 'weeks')
print('last window ends', (ORIGINS[-1] + pd.Timedelta(weeks=HORIZON_WEEKS) - pd.Timedelta(days=1)).date(), '(data ends', panel.date.max().date(), ')')"""),
    M("## Baselines (what a planner would do without a model)"),
    C("""res = []
for b in BRANDS:
    res.append(backtest(panel, b, lambda: NaiveRecent(8), name='naive: last 8 weeks mean'))
    res.append(backtest(panel, b, lambda: NaiveRecent(26), name='naive: last 26 weeks mean'))
    res.append(backtest(panel, b, lambda: SeasonalNaive(), name='seasonal naive (same ISO week last year x growth)'))
bt0 = pd.concat(res); bt0.to_pickle(METRICS / 'task2a_baselines_weeks.pkl')
s0 = summarize(bt0); s0.to_csv(METRICS / 'task2a_baselines.csv', index=False)
s0.pivot(index='model', columns='brand', values='WAPE').round(4)"""),
    C("""s0.sort_values(['brand', 'WAPE'])"""),
    C("""festival_split(bt0).pivot_table(index=['brand', 'model'], columns='bucket', values='WAPE').round(3)"""),
    M("""**Reading.** Constant-level baselines are acceptable in calm weeks but fail in festival run-up weeks (Fresh error about 3x larger than in other weeks). Seasonal-naive is the best baseline for Style (festival weeks repeat) but the worst for Fresh and Tech, because it copies last year's week even when the festival or the number of operating days has moved. These are the numbers the model has to beat (notebook 05)."""),
])

# ------------------------------------------------------------------------------------------------------ 05
nb("05_modeling.ipynb", [
    M("""# Task 2A - 05 Modelling: an ablation ladder
Start from the simplest explainable model and add **one idea at a time**, keeping an idea only if it improves MAE, RMSE and WAPE on the rolling backtest. (Rule agreed with the team: ridge first; boosting only if it helps consistently.)

Model family (all forecast **mean daily volume on operating days**, then sum to weeks):
- `DailyRidgeGB`: ridge on log daily volume, lognormal-mean back-transform (so forecasts are means), optional pooled depots, optional boosting on ridge residuals;
- `TechRidge`: small linear model for the sparse, noisy Tech series."""),
    C(HEADER),
    M("## 1. Fresh and Style ladder"),
    C("""NY = ('r_new_year',)
ladder = [
    ('A0 per-depot ridge, no trend',             lambda: DailyRidgeGB((), pooled=False, gb=False, trend='none')),
    ('A1 + damped trend',                         lambda: DailyRidgeGB((), pooled=False, gb=False, trend='damped')),
    ('A2 + New Year ramp',                        lambda: DailyRidgeGB(NY, pooled=False, gb=False)),
    ('A3 + a ramp per festival (per-depot)',      lambda: DailyRidgeGB(PROD_EXTRA, pooled=False, gb=False)),
    ('A4 + pooled depots',                        lambda: DailyRidgeGB(PROD_EXTRA, pooled=True, gb=False)),
    ('A5 + residual boosting (production Fresh)', lambda: DailyRidgeGB(PROD_EXTRA, pooled=True, gb=True)),
]
style_final = ('A6 Style: drop the trend (production Style)', lambda: DailyRidgeGB(PROD_EXTRA, pooled=True, gb=True, trend='none'))
res = [backtest(panel, b, mk, name=n) for b in ('Fresh', 'Style') for n, mk in ladder]
res.append(backtest(panel, 'Style', style_final[1], name=style_final[0]))
bt = pd.concat(res); bt.to_pickle(METRICS / 'task2a_ladder_weeks.pkl')
sl = summarize(bt); sl.to_csv(METRICS / 'task2a_ladder.csv', index=False)
sl.sort_values(['brand', 'model'])"""),
    C("""fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
for a, b in zip(axs, ('Fresh', 'Style')):
    g = sl[sl.brand == b].sort_values('model').reset_index(drop=True)
    a.barh(range(len(g)), g.WAPE * 100, color=['C0'] * (len(g) - 1) + ['C2']); a.set_yticks(range(len(g))); a.set_yticklabels([x[:44] for x in g.model], fontsize=8)
    a.invert_yaxis(); a.set_xlabel('WAPE % (lower is better)'); a.set_title(b)
plt.tight_layout(); plt.savefig(FIGURES / '05_ladder.png'); plt.show()"""),
    M("""**Reading (Fresh WAPE 5.2% -> 3.2%, Style 9.3% -> 7.6%).**
- **Trend** is the first big win for Fresh (5.2% -> 4.0%): demand grows a few % a year and a trend-free model is biased low (-4.3%).
- **A New Year ramp** (4.0% -> 3.6%) and then **a ramp per festival** (3.6% -> 3.25%) are the next biggest steps: festivals differ a lot in size and one shared ramp averages them.
- **Pooling the depots** is roughly neutral for Fresh (3.25% vs 3.26%) and a small gain for Style (8.43% -> 8.37%). We keep it because it gives one model per brand with equal or better accuracy, and it is what the boosting stage builds on - but it is *not* a source of accuracy and we do not claim it is.
- **Residual boosting** adds a further gain (Fresh 3.26% -> 3.18%, Style 8.4% -> 7.7%).
- **Style without a trend** is best (A6): its demand shows no reliable growth and extrapolating it only adds error; Fresh keeps the damped trend."""),
    M("""## 2. Is boosting worth its complexity? (consistency across origins)
Agreed rule: keep boosting only if it improves out-of-time accuracy consistently, not just on average."""),
    C("""a = bt[bt.model.str.startswith('A4')].groupby(['brand', 'cut']).apply(lambda g: (g.pred - g.actual).abs().sum() / g.actual.sum()).rename('ridge')
b_ = bt[bt.model.str.startswith('A5')].groupby(['brand', 'cut']).apply(lambda g: (g.pred - g.actual).abs().sum() / g.actual.sum()).rename('ridge+gb')
w = pd.concat([a, b_], axis=1)
print('origins where boosting is better:'); print((w['ridge+gb'] < w['ridge']).groupby('brand').agg(['sum', 'count']))
print('\\nmedian WAPE across origins:'); print(w.groupby('brand').median().round(4))"""),
    M("""**Reading (boosting).** Boosting is better in 21 of 34 origins for Fresh and 25 of 34 for Style, and improves the *median* origin (Fresh 3.17% -> 2.80%, Style 8.4% -> 7.9%). It is clearly better in festival windows (05b) and sometimes slightly worse in calm ones - a real but not universal gain, so it is kept as a small, shallow residual learner (depth 3, 60 trees) on top of the explainable ridge, which stays the backbone."""),
    M("## 3. Tech ladder"),
    C("""tl = [
    ('T0 constant: last 52 weeks mean',   lambda: NaiveRecent(52)),
    ('T1 weekday ridge',                  lambda: TechRidge(cols=DOW, trend=False)),
    ('T2 + payday + ramp',                lambda: TechRidge(trend=False)),
    ('T3 + damped trend (= production)',  lambda: TechRidge(trend=True)),
]
bt_t = pd.concat([backtest(panel, 'Tech', mk, name=n) for n, mk in tl]); bt_t.to_pickle(METRICS / 'task2a_tech_ladder_weeks.pkl')
st = summarize(bt_t); st.to_csv(METRICS / 'task2a_tech_ladder.csv', index=False); st"""),
    M("""**Reading.** Tech is mostly irreducible noise (a few large single-item orders), so the best model is only about 2.5% better in WAPE than a constant (30.1% -> 29.4%): the weekday schedule and payday help a little, a trend reduces bias. We tried Poisson GLMs, Poisson boosting, quarter dummies and shrinkage blends in exploration - none beat this small ridge, so complexity was not added."""),
    M("## 4. Chilled volume: share of total vs direct model"),
    C("""tot = lambda: make_model('Fresh')
cf = [
    ('share: total forecast x depot chilled share', lambda: ChilledModel(tot, 'share')),
    ('direct: same architecture fitted on chilled',  lambda: ChilledModel(tot, 'direct')),
    ('blend 30% direct + 70% share',                  lambda: ChilledModel(tot, 'blend', 0.3)),
    ('blend 50/50 (production)',                      lambda: ChilledModel(tot, 'blend', 0.5)),
]
bt_c = pd.concat([backtest(panel, 'Fresh', mk, name=n, target='ch') for n, mk in cf])
bt_c.to_pickle(METRICS / 'task2a_chilled_weeks.pkl'); sc = summarize(bt_c); sc.to_csv(METRICS / 'task2a_chilled.csv', index=False)
print(sc.to_string()); festival_split(bt_c).pivot_table(index='model', columns='bucket', values='WAPE').round(4)"""),
    M("""**Reading.** Chilled has its own run-up dynamics, so a direct model beats the plain share; the share is more stable outside festival weeks. A 50/50 blend is best on all three metrics. All variants are capped at the total forecast (`chilled <= total`). Style and Tech chilled are exactly 0."""),
    M("""## Production configuration
- **Fresh:** pooled `DailyRidgeGB` (ramp per festival, damped trend phi 0.9, residual boosting depth 3).
- **Style:** same, **without** a trend (rolling backtest prefers it on every metric, see 05c).
- **Tech:** `TechRidge` (weekday, payday, ramp, damped trend).
- **Chilled (Fresh):** 50/50 blend of a direct chilled model and total x chilled share.

Hyper-parameter sensitivity and stress tests follow in 05c; error analysis in 05b."""),
])

# ------------------------------------------------------------------------------------------------------ 05b
nb("05b_error_analysis.ipynb", [
    M("""# Task 2A - 05b Error analysis and uncertainty
Where does the production model go wrong, and how wide should the forecast bands be?"""),
    C(HEADER),
    C("""bt = pd.concat([backtest(panel, b, lambda b=b: make_model(b), name='production') for b in BRANDS]); bt.to_pickle(METRICS / 'task2a_final_weeks.pkl')
summ = summarize(bt); summ.to_csv(METRICS / 'task2a_final_summary.csv', index=False); summ"""),
    M("## 1. By forecast horizon (weeks ahead)"),
    C("""h = bt.groupby(['brand', 'horizon']).apply(lambda g: (g.pred - g.actual).abs().sum() / g.actual.sum()).unstack(0).round(3); h"""),
    M("""Fresh error is flat across horizons (about 3%): a calendar-driven model has nothing to roll forward, so it does not decay with distance. Style alternates (about 6% / 9%) because origins are 14 days apart, so odd and even horizons keep landing on the same fixed calendar weeks (some of them festival weeks); there is no systematic decay."""),
    M("## 2. Festival run-up weeks vs other weeks; short weeks"),
    C("""festival_split(bt).pivot_table(index=['brand'], columns='bucket', values=['WAPE', 'bias']).round(3)"""),
    C("""bt['week_type'] = np.where(bt.n_days < 6, 'short week (<6 operating days)', 'full week')
bt.groupby(['brand', 'week_type']).apply(lambda g: pd.Series(metric_row(g.actual, g.pred))).round(3)"""),
    M("## 3. Worst weeks"),
    C("""bt['abs_err'] = (bt.pred - bt.actual).abs(); bt['pct'] = (bt.pred - bt.actual) / bt.actual
bt[bt.brand != 'Tech'].sort_values('abs_err', ascending=False).head(12)[['brand', 'depot', 'k', 'cut', 'actual', 'pred', 'pct', 'ramp', 'n_days']].round(2)"""),
    C("""fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for a, b in zip(axs, BRANDS):
    g = bt[bt.brand == b]; a.scatter(g.actual, g.pred, s=6, alpha=.4); lim = [0, g.actual.max() * 1.05]; a.plot(lim, lim, 'k--', lw=.8)
    a.set_title(f'{b}: forecast vs actual (weekly, all origins)'); a.set_xlabel('actual m3'); a.set_ylabel('forecast m3')
plt.tight_layout(); plt.savefig(FIGURES / '05b_pred_vs_actual.png'); plt.show()"""),
    M("""**Reading.**
- The largest Fresh misses (10-15%) are the **Christmas weeks (ISO 51-52 of 2024 and the week after)** forecast from origins that had **never seen a Christmas**; this is a data-availability effect that disappears once a festival has been observed.
- **Short weeks are the weak spot:** with fewer than 6 operating days Fresh WAPE is 5.3% (bias -4.7%), Style 17.2% (bias -12.5%) and Tech 43% (bias -27%): the model removes the closed days a little too aggressively. Our real window has two such weeks (16 and 18). We tested catch-up features (05c, section 6) and they did not fix it.
- Style under-forecasts festival run-up weeks (bias -14%); Tech is noise-limited.
- Outside short weeks the forecasts are essentially unbiased (Fresh -1%, Tech -2%, Style -3%)."""),
    M("""## 4. Forecast bands (empirical, calibrated out-of-time)
Bands are built from the ratio actual/forecast of past backtests: quantiles are estimated on the **first 17 origins** and their coverage is checked on the **last 17**, so the reported coverage is honest."""),
    C("""bt['ratio'] = bt.actual / bt.pred.clip(lower=1e-6)
first, last = sorted(bt.cut.unique())[:17], sorted(bt.cut.unique())[17:]
rows = []
for b in ('Fresh', 'Style'):
    cal_, test_ = bt[(bt.brand == b) & bt.cut.isin(first)], bt[(bt.brand == b) & bt.cut.isin(last)]
    lo, hi = cal_.ratio.quantile(.1), cal_.ratio.quantile(.9)
    cov = ((test_.actual >= test_.pred * lo) & (test_.actual <= test_.pred * hi)).mean()
    rows.append((b, round(lo, 3), round(hi, 3), round(cov, 3)))
bands = pd.DataFrame(rows, columns=['brand', 'ratio_p10', 'ratio_p90', 'coverage_on_held_out_origins (target 0.80)']); print(bands)
# Tech is count-noise dominated: additive bands in m3 per week (stable across depots)
tb = bt[bt.brand == 'Tech']; err = tb.actual - tb.pred
tcal, ttest = tb[tb.cut.isin(first)], tb[tb.cut.isin(last)]
elo, ehi = (tcal.actual - tcal.pred).quantile(.1), (tcal.actual - tcal.pred).quantile(.9)
print('Tech additive band (m3/week): [%.1f, %.1f], coverage %.3f' % (elo, ehi, ((ttest.actual - ttest.pred).between(elo, ehi)).mean()))
json.dump({'Fresh': bands.iloc[0, 1:3].tolist(), 'Style': bands.iloc[1, 1:3].tolist(), 'Tech_additive': [float(elo), float(ehi)]}, open(METRICS / 'task2a_band_params.json', 'w'))"""),
    M("""**Reading.** On origins that were *not* used to set them, coverage is 80% for Style, 82% for Tech, and 91% for Fresh (conservative, because the calibration origins had less history and larger errors). They are an extra deliverable for planners (vehicle/driver planning needs a range); the submission file contains the point forecast only."""),
    M("""## 5. Tested and rejected: a multiplicative correction for Style festival weeks
Style is under-forecast in run-up weeks (bias about -14%). Rule agreed with the team: a forecast adjustment is kept **only** if it improves the validation metric consistently. We calibrate a factor on one half of the origins and test it on the other half, in both directions."""),
    C("""g = bt[bt.brand == 'Style']; cuts = sorted(g.cut.unique()); h1, h2 = cuts[:17], cuts[17:]
wp = lambda d, p: (p - d.actual).abs().sum() / d.actual.sum()
rows = []
for name, cal_cuts, test_cuts in (('calibrate on early origins, test on later', h1, h2), ('calibrate on later origins, test on early', h2, h1)):
    c = g[g.cut.isin(cal_cuts) & (g.ramp > 0.4)]; t = g[g.cut.isin(test_cuts) & (g.ramp > 0.4)]; f = c.actual.sum() / c.pred.sum()
    rows.append((name, round(f, 3), len(t), round(wp(t, t.pred), 3), round(wp(t, t.pred * f), 3)))
pd.DataFrame(rows, columns=['split', 'factor', 'test run-up weeks', 'WAPE before', 'WAPE after'])"""),
    M("""**Decision: rejected.** The correction helps in one direction and hurts in the other, and the factor is unstable (about 1.11 vs 1.23). The under-forecast is largest for origins that had seen few festivals and shrinks as history grows, so it is a data-availability effect, not a fixed bias; the final model, which has seen two seasons of each festival, should be closer to unbiased. We keep the unadjusted forecast and flag Style week 15 (New Year run-up) as the main residual risk."""),
])

# ------------------------------------------------------------------------------------------------------ 05c
nb("05c_robustness.ipynb", [
    M("""# Task 2A - 05c Robustness and stress tests
Is the model fragile? We check (1) hyper-parameter sensitivity, (2) trend choice, (3) agreement of MAE / RMSE / WAPE, (4) dependence on a single festival year, and (5) the spread across plausible model variants for the real 10 weeks."""),
    C(HEADER),
    M("## 1. Hyper-parameter sensitivity (should be a plateau, not a spike)"),
    C("""P = PROD_EXTRA
mkp = lambda b, **kw: DailyRidgeGB(P, trend='none' if b == 'Style' else 'damped', **kw)     # production trend per brand
cfgs = {}
for a in (1e-3, 1e-2, 1e-1, 1.0): cfgs[f'ridge alpha={a}'] = dict(alpha=a)
for ph in (0.5, 0.8, 0.9, 0.95, 1.0): cfgs[f'trend phi={ph} (Fresh only; Style has no trend)'] = dict(phi=ph)
for d, it in ((2, 60), (3, 60), (4, 60), (3, 30), (3, 120)): cfgs[f'gb depth={d} iter={it}'] = dict(gb_kw=dict(max_depth=d, max_iter=it))
res = [backtest(panel, b, lambda b=b, kw=kw: mkp(b, **kw), name=n) for b in ('Fresh', 'Style') for n, kw in cfgs.items()]
hp = summarize(pd.concat(res)); hp.to_csv(METRICS / 'task2a_hyperparam_sweep.csv', index=False)
hp.pivot(index='model', columns='brand', values='WAPE').round(4)"""),
    M("""**Reading.** Fresh WAPE stays between 3.15% and 3.23% across ridge strength, trend damping and boosting size - a flat plateau, so the chosen defaults are not tuned to noise. Style behaves the same except for heavy ridge shrinkage (alpha = 1: 8.5% vs 7.6%), which flattens the festival effects."""),
    M("## 2. Trend: none vs full vs damped (agreed selection rule)"),
    C("""tr = pd.concat([backtest(panel, b, lambda tr_=tr_: DailyRidgeGB(P, trend=tr_), name=f'trend={tr_}') for b in ('Fresh', 'Style') for tr_ in ('none', 'full', 'damped')])
summarize(tr).sort_values(['brand', 'WAPE'])"""),
    M("""**Selection rule (agreed): lowest error on comparable ten-week windows.**
- **Fresh:** no trend is far worse (WAPE 5.7% vs 3.2%); full trend (3.16%) and damped trend (3.18%) are a statistical tie, so we take the **damped** trend as the conservative choice for a 10-week extrapolation.
- **Style:** **no trend** wins on MAE, RMSE and WAPE (7.61% vs 7.71% damped), so Style is forecast without a trend.
- **Tech:** the damped trend reduces bias (-5.9% -> -3.4%) with a lower RMSE and a MAE within 0.2%, so it is kept."""),
    M("## 3. Do MAE, RMSE and WAPE agree on the ranking?"),
    C("""sl = pd.read_csv(METRICS / 'task2a_ladder.csv')
for m in ('MAE', 'RMSE', 'WAPE'): sl[m + '_rank'] = sl.groupby('brand')[m].rank()
sl[['brand', 'model', 'MAE_rank', 'RMSE_rank', 'WAPE_rank']].sort_values(['brand', 'WAPE_rank'])"""),
    M("""The three metrics agree on the best model (production) and on the worst (no-trend ridge) for both brands. The only disagreements are between neighbouring rungs that differ by less than 0.1 percentage point (e.g. A3 vs A4 for Fresh), which are ties. So the chosen configuration does not depend on the unknown official metric."""),
    M("## 4. How much does the forecast rely on a single New Year?"),
    C("""fops = future_ops(cal2, panel); cut = panel.date.max() + pd.Timedelta(days=1)
def forecast(trn_mask, brand='Fresh', make=None):
    m = (make or (lambda: make_model(brand)))().fit(panel[(panel.brand == brand) & trn_mask])
    f = pd.concat([fops.assign(depot=d, brand=brand) for d in DEPOTS]); f = f[f.iso_week.between(14, 23)]
    return f.assign(p=m.predict(f, cut)).groupby(['depot', 'iso_week']).p.sum()
ny24 = panel.date.between('2024-04-01', '2024-04-20'); ny25 = panel.date.between('2025-04-01', '2025-04-20')
out = pd.DataFrame({'all data': forecast(panel.date.notna()), 'without New Year 2024': forecast(~ny24), 'without New Year 2025': forecast(~ny25)})
out['spread %'] = ((out.max(axis=1) - out.min(axis=1)) / out['all data'] * 100).round(1)
out.loc[(slice(None), [15, 16]), :].round(1)"""),
    M("""Dropping either New Year leaves the week 15/16 forecast within a modest range, because the model also learns from the other festivals and the ramp shape. This quantifies the main residual risk: *the New Year effect is estimated from two observed seasons.*"""),
    M("## 5. Spread across plausible model variants (real 10 weeks)"),
    C("""variants = {'production': lambda: make_model('Fresh'), 'per-depot ridge, NY ramp': lambda: DailyRidgeGB(('r_new_year',), pooled=False, gb=False),
            'pooled ridge, all ramps': lambda: DailyRidgeGB(PROD_EXTRA, gb=False), 'production, full trend': lambda: DailyRidgeGB(PROD_EXTRA, trend='full')}
fv = pd.DataFrame({n: forecast(panel.date.notna(), make=mk) for n, mk in variants.items()})
fv['max/min'] = (fv.max(axis=1) / fv.min(axis=1)).round(3); fv.round(1)"""),
    M("""Variants agree within about 1% in every week **except week 22** (Poson plus two paydays), where the model with a single shared ramp forecasts 8-10% higher. History explains why we trust the per-festival ramps: Poson's run-up is small (about +8% in 2025) whereas the New Year's is large, and the shared-ramp model cannot tell them apart. Week 22 remains the least certain week of the window."""),
    M("""## 6. Tested and rejected: catch-up demand around closures
Short weeks (fewer than 6 operating days) are under-forecast (see 05b). Hypothesis: when a day is closed, some demand moves onto the days that remain. We added `closed_wk` (closed Mon-Sat days in the ISO week) and `post_gap` (previous Mon-Sat day closed) as features."""),
    C("""P = PROD_EXTRA
rows = []
for b in ('Fresh', 'Style'):
    for n, ex in (('production', P), ('+ closed_wk', P + ('closed_wk',)), ('+ post_gap', P + ('post_gap',)), ('+ both', P + ('closed_wk', 'post_gap'))):
        rows.append(backtest(panel, b, lambda ex=ex, b=b: DailyRidgeGB(ex, trend='none' if b == 'Style' else 'damped'), name=n))
for n, cols in (('production', None), ('+ closed_wk', DOW + ['is_payday', 'festival_ramp', 'closed_wk']), ('+ both', DOW + ['is_payday', 'festival_ramp', 'closed_wk', 'post_gap'])):
    rows.append(backtest(panel, 'Tech', lambda cols=cols: TechRidge(cols=cols), name=n))
cu = pd.concat(rows); print(summarize(cu).sort_values(['brand', 'model']).to_string())
cu['short week'] = cu.n_days < 6
cu.groupby(['brand', 'model', 'short week']).apply(lambda g: pd.Series(metric_row(g.actual, g.pred))).round(3)[['WAPE', 'bias']].unstack('short week')"""),
    M("""**Reading.** Overall error does not improve (changes of a few hundredths of a percentage point) and the short-week bias stays (Style about -12%, Tech about -25%). Only 40 short-week forecasts per brand exist, so we do **not** add features for them; instead the limitation is documented (short weeks 16 and 18 in the real window carry more uncertainty than the bands for ordinary weeks suggest)."""),
])

# ------------------------------------------------------------------------------------------------------ 06
nb("06_predict_submit.ipynb", [
    M("""# Task 2A - 06 Final fit, forecast and submission
Fit the production models on all history (through 2026-03-28), forecast the 10 weeks, validate the file, save models and bands, and demonstrate inference from the saved model."""),
    C(HEADER),
    C("""fc = Forecaster().fit(panel)
joblib.dump(fc, MODELS / 'task2a_models.joblib')
ti = pd.read_csv(RAW / 'test' / 'task2a_test_inputs.csv')
fops = future_ops(cal2, panel)
out = fc.weekly(fops, ti)
sub = out[['row_id', 'pred_total_volume_m3', 'pred_chilled_volume_m3']]
sub.to_csv(PREDICTIONS / 'submission_task2a.csv', index=False)
print(sub.shape); sub.head()"""),
    M("## Submission checks"),
    C("""tmpl = pd.read_csv(RAW / 'templates' / 'submission_task2a.csv')
assert list(sub.row_id) == list(tmpl.row_id) and len(sub) == 60, 'row_id set/order differs from template'
assert sub.notna().all().all() and (sub.drop(columns='row_id') >= 0).all().all()
assert (sub.pred_chilled_volume_m3 <= sub.pred_total_volume_m3).all()
assert (out[out.brand != 'Fresh'].pred_chilled_volume_m3 == 0).all()
used = fops[(fops.iso_year == 2026) & fops.iso_week.between(14, 23)]
assert used.is_operating.eq(1).all() and (used.dow != 6).all()
print('all checks passed | operating days per week used:', used.groupby('iso_week').size().to_dict())"""),
    M("## Forecast vs last year (same ISO weeks) and bands"),
    C("""bp = json.load(open(METRICS / 'task2a_band_params.json'))
out['p10'] = np.where(out.brand == 'Tech', (out.pred_total_volume_m3 + bp['Tech_additive'][0]).clip(lower=0), out.pred_total_volume_m3 * out.brand.map({'Fresh': bp['Fresh'][0], 'Style': bp['Style'][0], 'Tech': np.nan}))
out['p90'] = np.where(out.brand == 'Tech', out.pred_total_volume_m3 + bp['Tech_additive'][1], out.pred_total_volume_m3 * out.brand.map({'Fresh': bp['Fresh'][1], 'Style': bp['Style'][1], 'Tech': np.nan}))
out.drop(columns=['row_id']).round(1).to_csv(PREDICTIONS / 'task2a_forecast_with_bands.csv', index=False)
hist = panel[(panel.iso_year == 2025) & panel.iso_week.between(14, 23)].groupby(['depot', 'brand', 'iso_week']).vol.sum().unstack()
f_tab = out.pivot_table(index=['depot', 'brand'], columns='iso_week', values='pred_total_volume_m3')
print('Forecast 2026 (m3)'); print(f_tab.round(0)); print('\\nActual 2025, same ISO weeks'); print(hist.round(0)); print('\\nforecast / 2025 total:', round(f_tab.sum().sum() / hist.sum().sum(), 3))"""),
    C("""fig, axs = plt.subplots(2, 3, figsize=(15, 6))
for a, (dep, b) in zip(axs.ravel(), SERIES):
    h = panel[(panel.depot == dep) & (panel.brand == b) & (panel.date >= '2025-09-01')].groupby(['iso_year', 'iso_week']).vol.sum().values
    g = out[(out.depot == dep) & (out.brand == b)].sort_values('iso_week')
    x = np.arange(len(h), len(h) + len(g))
    a.plot(range(len(h)), h, label='history'); a.plot(x, g.pred_total_volume_m3, 'C3', label='forecast'); a.fill_between(x, g.p10, g.p90, color='C3', alpha=.18, label='80% band'); a.set_title(f'{dep} {b}')
axs[0, 0].legend(); plt.tight_layout(); plt.savefig(FIGURES / '06_forecast.png'); plt.show()"""),
    M("## Capacity read-out for planners (chilled share of Fresh)"),
    C("""cap = out[out.brand == 'Fresh'].pivot_table(index='iso_week', columns='depot', values='pred_chilled_volume_m3').round(0)
cap['total chilled m3/week'] = cap.sum(axis=1); cap"""),
    M("""## Inference demo - loads the saved model and prints inputs and predictions"""),
    C("""fc2 = joblib.load(MODELS / 'task2a_models.joblib')
demo = ti.iloc[[0, 1, 2, 3, 4, 5, 6, 12, 18, 54]]          # a mix of depots, brands and weeks incl. New Year (15, 16)
print('INPUT'); print(demo.to_string(index=False))
pred = fc2.weekly(fops, demo)
print('\\nPREDICTIONS'); print(pred[['row_id', 'depot', 'brand', 'iso_week', 'pred_total_volume_m3', 'pred_chilled_volume_m3']].to_string(index=False))
chk = sub.set_index('row_id').loc[pred.row_id]
assert np.allclose(chk.pred_total_volume_m3.values, pred.pred_total_volume_m3.values), 'reloaded model differs from the submission'
print('\\nreloaded model reproduces the submission rows: OK')"""),
])
print("wrote notebooks to", OUT)
