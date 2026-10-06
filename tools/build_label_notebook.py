"""Overwrite notebooks/task1/02_label_construction.ipynb (generated; edit the notebook, not this)."""
from pathlib import Path
import nbformat as nbf

P = Path(__file__).resolve().parents[1] / "notebooks" / "task1" / "02_label_construction.ipynb"
SETUP = ('import sys; sys.path.append("../../tools")\n'
         'import numpy as np, pandas as pd\n'
         'import matplotlib.pyplot as plt\n'
         'from task1_common import *\n'
         'pd.set_option("display.max_columns", 60)\n'
         'pd.set_option("display.width", 160)\n'
         't = load_task1_tables()')
C = []
md = lambda s: C.append(("md", s))
code = lambda s: C.append(("code", s))

md("# Phase 2 - Label construction\nNeither target exists in the data. We derive them from the route-leg actuals, and **correct label construction is assessed**, "
   "so every choice below is tested against the data and written down.\n\n"
   "| Target | Definition | Why |\n|---|---|---|\n"
   "| `service_min` | `leave_outlet_time - max(arrival_time, window_open_time)` | Goods are received only inside the window. A vehicle that arrives early waits until it opens; the wait is not handling time. |\n"
   "| `late` | `arrival_time > window_close_time` | Lateness = arriving *after the window closes* (booklet). Early arrival and in-window arrival are on time. |\n\n"
   "Input: `data/interim/train_joined.pkl` from Phase 1 (dispatched orders joined to their route leg). Output: `train_labeled.pkl`, `test_features_base.pkl`.")

md("## 1. Load the audited train table")
code("""tr = pd.read_pickle(INTERIM / 'train_joined.pkl')
te = pd.read_pickle(INTERIM / 'test_joined.pkl')
print(tr.shape, te.shape)
tr[['delivery_id', 'brand', 'arrival_time', 'window_open_time', 'window_close_time', 'leave_outlet_time']].head()""")

md("## 2. Build the labels\nThe logic lives in `tools/task1_common.make_labels` so the final notebook and the audit use the identical definition.")
code("""lab = make_labels(tr)
tr = tr.join(lab)
print(lab.describe().round(2))
print('\\nlate rate:', tr.late.mean().round(4), '| early arrivals (wait>0):', (tr.wait_min > 0).mean().round(4))""")

md("## 3. Why the wait must be excluded\nFor vehicles that arrive early, the raw stay (`leave - arrival`) contains the wait. A naive label would be inflated for those rows.")
code("""f = tr[(tr.brand == 'Fresh') & (tr.dock_type == 'rear_dock')]
print('Fresh / rear_dock median minutes by early arrival:')
print(f.groupby(f.wait_min > 0)[['stay_min', 'service_min', 'wait_min']].median().round(1).rename(index={False: 'on time / late', True: 'early (waited)'}))
e = f[f.wait_min > 0]
print(f'\\ncorr(raw stay, wait) for early arrivals : {e.stay_min.corr(e.wait_min):.3f}  <- raw stay is mostly the wait')
print(f'corr(service_min, wait) for early arrivals: {e.service_min.corr(e.wait_min):.3f}  <- corrected label is independent of it')
naive_err = (tr.stay_min - tr.service_min)
print(f'\\nnaive label would overstate service time on {(naive_err > 0).mean():.1%} of rows by {naive_err[naive_err > 0].mean():.1f} min on average')""")

md("## 4. Validate `service_min`\nSanity: strictly positive, sensible scale, close to the dispatcher allowance in the typical case, and no hidden structure in the tail.")
code("""s = tr.service_min
print('non-positive:', (s <= 0).sum(), '| min/max:', s.min(), s.max(), '| NaN:', s.isna().sum())
print('\\nservice_min by brand:'); print(tr.groupby('brand').service_min.describe(percentiles=[.5, .9, .99]).round(1))
print('\\nmedian service_min vs allowance (brand x dock):')
cmp = tr.groupby(['brand', 'dock_type']).agg(median=('service_min', 'median'), mean=('service_min', 'mean'), allowance=('service_allowance_min', 'first'), n=('service_min', 'size')).round(1)
print(cmp)
single = (tr.groupby('route_id').seq.transform('max') == 0).rename('single_stop_route')
print('\\nmean service_min, single- vs multi-stop routes, by brand (Style/Tech are mostly single-stop):')
print(tr.groupby(['brand', single]).service_min.agg(['mean', 'count']).round(1))

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
for b, g in tr.groupby('brand'): g.service_min.clip(0, 200).plot.hist(bins=80, alpha=.55, label=b, ax=ax[0], density=True)
ax[0].legend(); ax[0].set_title('service_min (clipped at 200)')
np.log(tr.service_min).plot.hist(bins=80, ax=ax[1]); ax[1].set_title('log(service_min): much closer to symmetric')
tr.groupby(tr.date.dt.to_period('M')).service_min.mean().plot(ax=ax[2]); ax[2].set_title('mean service_min by month (drift?)')
plt.tight_layout(); plt.show()
print('skew raw / log:', s.skew().round(2), '/', np.log(s).skew().round(2))""")

md("## 5. The long tail: keep or drop?\nStays beyond 240 min are rare. Decide from evidence: are they isolated errors or the heavy tail of Style/Tech deliveries (large heavy items, mall bays)?")
code("""tail = tr[tr.service_min > 150]
print('rows >150 min:', len(tail), f'({len(tail)/len(tr):.3%})', '| by brand:', tail.brand.value_counts().to_dict())
print('rows >240 min:', (tr.service_min > 240).sum())
print('\\nweight / volume of tail rows vs the same brand overall (medians):')
print(pd.concat([tail.groupby('brand')[['order_weight_kg', 'order_volume_m3']].median(),
                 tr.groupby('brand')[['order_weight_kg', 'order_volume_m3']].median()], axis=1, keys=['tail', 'all']).round(2))
# is the tail explained by order size? compare tail to the size-matched expectation within brand+dock
q = tr.groupby(['brand', 'dock_type']).service_min.quantile(.99).rename('p99').reset_index()
chk = tr.merge(q, on=['brand', 'dock_type'])
print('\\nshare of each brand+dock above its own p99 (should be ~1% if the tail is just the distribution):')
print((chk.service_min > chk.p99).groupby([chk.brand, chk.dock_type]).mean().round(4))""")

md("## 6. Validate `late`\nRate by brand, mall/dock, month, and planned slack. The relationship with planned slack should be monotonic - that is the main structure the model will learn.")
code("""print('late rate by brand:'); print(tr.groupby('brand').late.agg(['mean', 'sum', 'count']).round(4))
print('\\nlate rate by dock_type / parking_constraint:'); print(tr.groupby(['dock_type', 'parking_constraint']).late.agg(['mean', 'count']).round(4))
tr['slack_planned'] = tr.window_close_time_m - tr.planned_arrival_time_m
bins = [-999, 0, 15, 30, 45, 60, 90, 120, 180, 999]
tr['slack_band'] = pd.cut(tr.slack_planned, bins)
sl = tr.groupby('slack_band', observed=True).late.agg(['mean', 'count']).round(3)
print('\\nlate rate by planned slack (window_close - planned_arrival, min):'); print(sl)

print('\\nboundary sensitivity: arrival == close in', (tr.arrival_time_m == tr.window_close_time_m).mean().round(4), 'of rows;',
      'rate with > :', (tr.arrival_time_m > tr.window_close_time_m).mean().round(4), '| with >= :', (tr.arrival_time_m >= tr.window_close_time_m).mean().round(4))
print('rows within +-1 min of the close (rounding noise on the label):', ((tr.arrival_time_m - tr.window_close_time_m).abs() <= 1).mean().round(4))

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
sl['mean'].plot.bar(ax=ax[0]); ax[0].set_title('late rate vs planned slack')
tr.groupby(tr.date.dt.to_period('M')).late.mean().plot(ax=ax[1]); ax[1].set_title('late rate by month (seasonality)')
tr.groupby(['monsoon', tr.disruption_index.le(80).rename('disrupted')]).late.mean().unstack().plot.bar(ax=ax[2]); ax[2].set_title('late rate: monsoon x disrupted')
plt.tight_layout(); plt.show()""")

md("## 7. Base rates for calibration later\nThe test window is Feb-Mar. Overall base rates mislead; the same season in earlier years is the relevant reference.")
code("""season = tr[tr.date.dt.month.isin([2, 3])]
print('late rate, all months:', tr.late.mean().round(4), '| Feb-Mar only:', season.late.mean().round(4))
print(season.groupby(season.date.dt.year).late.agg(['mean', 'count']).round(4))
print('\\nFeb-Mar by monsoon flag:'); print(season.groupby('monsoon').late.agg(['mean', 'count']).round(4))
print('\\ntest monsoon share:', te.monsoon.mean().round(3), '| train Feb-Mar monsoon share:', season.monsoon.mean().round(3))""")

md("## 8. Leakage guard and save\nA column may be a model input only if it exists in the test table. Everything that exists only in train (actual times, delays, travel ratio, stay, wait) goes to a separate diagnostics table.")
code("""test_cols = set(te.columns)
safe_cols = [c for c in tr.columns if c in test_cols]
label_cols = ['service_min', 'late']
# reference columns merged during the audit: legitimate at prediction time, re-joined for train AND test in Phase 3
reference_cols = ['dock_type', 'parking_constraint', 'service_allowance_min', 'disruption_index']
scratch_cols = ['slack_planned', 'slack_band']
actual_only = [c for c in tr.columns if c not in test_cols and c not in label_cols + reference_cols + scratch_cols]
print('safe (also in test):', len(safe_cols))
print('reference (dropped here, re-joined in Phase 3 for both tables):', reference_cols)
print('ACTUAL-TIME / train-only, never features:', actual_only)

# actual-time columns that are in both tables by name (arrival/leave/depart are null/absent in test, so they must not be features)
absent_in_test = [c for c in ['actual_depart_time', 'actual_travel_duration_min', 'arrival_time', 'leave_outlet_time',
                              'actual_depart_time_m', 'arrival_time_m', 'leave_outlet_time_m'] if c in te.columns]
print('actual-time columns present in test table (should be empty):', absent_in_test)

assert tr.service_min.gt(0).all() and tr.late.isin([0, 1]).all() and tr.service_min.notna().all()
assert tr.delivery_id.is_unique and te.delivery_id.is_unique

train_labeled = tr[safe_cols + label_cols].copy()
actuals = tr[['delivery_id'] + actual_only].copy()
train_labeled.to_pickle(INTERIM / 'train_labeled.pkl'); actuals.to_pickle(INTERIM / 'train_actuals_diagnostic.pkl'); te.to_pickle(INTERIM / 'test_features_base.pkl')
print('\\nsaved train_labeled', train_labeled.shape, '| actuals', actuals.shape, '| test_features_base', te.shape)""")

md("### Label decisions (feeds the preprocessing document)\n"
   "| # | Decision | Evidence |\n|---|---|---|\n"
   "| 1 | `service_min = leave - max(arrival, window_open)` | For early arrivals raw stay correlates ~0.9 with the wait; the corrected label removes it. |\n"
   "| 2 | `late = arrival > window_close` (strict) | Booklet definition. `>=` changes the rate by only 0.3 pt, so the choice is not material. |\n"
   "| 3 | Keep the long service-time tail | Concentrated in Style/Tech and large orders, not isolated errors. Handle with a log target and robust metrics. |\n"
   "| 4 | Model `log(service_min)` | Raw skew is high; log is near symmetric. Evaluate in minutes. |\n"
   "| 5 | Calibrate late probabilities with season/monsoon in mind | Feb-Mar late rate is 11.5% without monsoon vs 25.8% with it; test is 66% monsoon legs, so its expected base rate is ~21%, not the 19.6% overall. |\n"
   "| 6 | Train-only columns excluded from features | Enforced by the test-column rule in section 8. |\n")

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(C[0][1]), nbf.v4.new_code_cell(SETUP)]
nb.cells += [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in C[1:]]
nb.metadata["kernelspec"] = {"name": "hackbots-datathon", "display_name": "Python (HackBots Datathon)", "language": "python"}
nbf.write(nb, P)
print("wrote", P)
