"""Generate the Task 2B notebooks (01 ... 07) into notebooks/task2b/.

Run from the repo root:  python tools/task2b/make_task2b_notebooks.py
Then execute them in order (jupyter nbconvert --execute --inplace).

The notebooks read the teammate's submission and solver (read-only) and add an exact (proven-optimal) solver,
tools/task2b/task2b_exact.py. They write only NEW files (prefixed `exact_` / `_exact_`).
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "task2b"

HEADER = """import sys, warnings; warnings.filterwarnings('ignore')
sys.path.append('../../tools/task2b')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, json, time
from task2b_common import *
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 120)
plt.rcParams.update({'figure.dpi': 90, 'axes.grid': True, 'grid.alpha': .25})
for d in (METRICS, FIGURES, INTERIM): d.mkdir(parents=True, exist_ok=True)
scn, fleet, veh, dt, al = load_tables()
sub_team = pd.read_csv(SUBMISSION)            # the teammate's submitted allocation (read-only)
"""

SOLVER = """import task2b_exact as X
S = X.S
"""


ONLY = sys.argv[1:]          # optional: regenerate only the notebooks whose file name starts with one of these


def nb(name, cells):
    if ONLY and not any(name.startswith(o) for o in ONLY):
        return
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nbf.write(n, OUT / name)


M = lambda s: ("md", s)
C = lambda s: ("code", s)

# ------------------------------------------------------------------------------------------------------------- 01
nb("01_data_audit.ipynb", [
    M("""# Task 2B - 01 Data audit
Scenario S1: one dispatch day at Peliyagoda, orders closed, a festival a week away, several vehicles in the workshop. We must mark every order served/deferred and give every served order a vehicle and trip.

**Questions for this notebook:** are the inputs clean, what is actually available (workshop vehicles excluded), and how does demand compare with capacity?"""),
    C(HEADER),
    M("## 1. Orders"),
    C("""print('orders:', scn.shape, '| unique order_ref:', scn.order_ref.is_unique, '| scenarios:', scn.scenario.unique(), '| depots:', scn.depot.unique())
print('outlets with more than one order:', int((scn.outlet_id.value_counts() > 1).sum()), '(so order_ref, not outlet_id, is the allocation key)')
print('missing values per column:'); print(scn.isna().sum()[lambda s: s > 0])
print()
print(scn.groupby(['brand', 'temp_requirement']).agg(orders=('order_ref', 'size'), m3=('order_volume_m3', 'sum'), kg=('order_weight_kg', 'sum')).round(1))
print('\\nparking constraints:', scn.parking_constraint.value_counts().to_dict(), '| dock types:', scn.dock_type.value_counts().to_dict())"""),
    M("""`mall_window` is empty for non-mall outlets; the two `mall_dock` orders (one Tech, one Style) can only be delivered inside their 10:00-12:00 mall window. `van_only` outlets (6 orders: three outlets x one ambient + one chilled order) cannot be served by trucks."""),
    M("## 2. Fleet: what is really available"),
    C("""f = fleet.merge(veh, on='vehicle_id')
print('fleet file rows:', len(fleet), '| status:', fleet.status.value_counts().to_dict(), '| depots:', f.depot.unique())
f['class'] = f.type + '/' + f.temp
print(f.groupby(['status', 'class']).agg(vehicles=('vehicle_id', 'size'), m3_each=('volume_cap_m3', 'mean'), kg_each=('weight_cap_kg', 'mean')).round(1))
print('\\nin workshop:', f[f.status == 'in_workshop'].vehicle_id.tolist())
print('reefers in workshop:', f[(f.status == 'in_workshop') & (f.temp == 'reefer')].vehicle_id.tolist())"""),
    M("""**Finding.** Only 28 of the 38 listed vehicles may be used. The workshop holds **5 of the fleet's reefers** (the chilled-capable vehicles), leaving 3 reefer trucks and 1 reefer van. Reefer capacity - not total fleet size - is what will limit service on a day with rising dairy/meat/produce demand."""),
    M("## 3. Demand by district and travel time"),
    C("""d = dt.set_index('district')
t = scn.assign(chilled_m3=np.where(scn.temp_requirement == 'chilled', scn.order_volume_m3, 0)).groupby('district').agg(
    orders=('order_ref', 'size'), chilled_orders=('temp_requirement', lambda s: int((s == 'chilled').sum())), m3=('order_volume_m3', 'sum'), chilled_m3=('chilled_m3', 'sum'))
t = t.join(d[['depot_to_district_freeflow_min', 'inter_stop_freeflow_min']].rename(columns={'depot_to_district_freeflow_min': 'outbound_min', 'inter_stop_freeflow_min': 'inter_stop_min'})).sort_values('outbound_min')
print(t.round(1))
fig, ax = plt.subplots(figsize=(9, 3.6)); x = np.arange(len(t))
ax.bar(x, t.m3 - t.chilled_m3, label='ambient m3'); ax.bar(x, t.chilled_m3, bottom=t.m3 - t.chilled_m3, label='chilled m3', color='C3')
ax.set_xticks(x); ax.set_xticklabels([f'{i}\\n({int(o)} min)' for i, o in zip(t.index, t.outbound_min)], fontsize=8); ax.legend(); ax.set_title('Demand by district (outbound minutes in brackets)')
plt.tight_layout(); plt.savefig(FIGURES / '01_demand_by_district.png'); plt.show()"""),
    M("""**Finding.** The far districts (Galle, Matara, Kurunegala, Puttalam: 103-173 outbound minutes) carry chilled demand too, and a Fresh vehicle has only 270 minutes before stores open - that makes time, not only volume, a constraint."""),
    M("## 4. Demand vs capacity per resource"),
    C("""avail = fleet[fleet.status == 'available'].merge(veh, on='vehicle_id')
cap = avail.groupby('temp').agg(vehicles=('vehicle_id', 'size'), m3_per_wave=('volume_cap_m3', 'sum'), kg_per_wave=('weight_cap_kg', 'sum'))
dem = scn.groupby(scn.temp_requirement.map({'chilled': 'reefer', 'ambient': 'ambient'})).agg(orders=('order_ref', 'size'), m3=('order_volume_m3', 'sum'), kg=('order_weight_kg', 'sum'))
cmp_ = cap.join(dem); cmp_['m3_two_waves'] = cmp_.m3_per_wave * 2
cmp_['demand/capacity (2 waves)'] = (cmp_.m3 / cmp_.m3_two_waves).round(2)
print(cmp_.round(1))
cmp_.to_csv(INTERIM / 'exact_demand_vs_capacity.csv'); cmp_"""),
    M("""## Audit conclusions
1. Inputs are clean: unique `order_ref`, one scenario, one depot, no missing values; vehicles in the fleet file all belong to Peliyagoda.
2. 10 vehicles are in the workshop and must never be used; 5 of them are reefers.
3. Chilled demand (181.6 m3) is **more than twice** what the 4 available reefers can carry in one wave (86.2 m3) and about equal to two full waves (172.4 m3) - before the 270-minute pre-dawn limit is even considered. Ambient demand is far below ambient capacity.
4. So the day is a **reefer-capacity day**: which chilled orders to serve is the real decision; everything else should simply be served."""),
])

# ------------------------------------------------------------------------------------------------------------- 02
nb("02_rule_engine.ipynb", [
    M("""# Task 2B - 02 Rule engine and trip-time calculation
Before optimising anything we make the booklet's rules executable and test them against the booklet's own worked examples."""),
    C(HEADER + SOLVER),
    M("""## 1. The feasibility rules (booklet p.20)
| # | Rule |
|---|---|
| R1 | all orders on one vehicle+trip share one brand and one district |
| R2 | chilled orders need a reefer (reefers may also carry ambient) |
| R3 | `van_only` outlets need a van |
| R4 | a vehicle serves only outlets of its own depot |
| R5 | whole orders: never split an order |
| R6 | per trip: total volume <= vehicle volume cap and total weight <= weight cap |
| R7 | at most 2 trips per vehicle; trip minutes fit the budgets: Fresh trips share **270 min**, Style+Tech trips share **480 min** |

`trip_minutes = depot_to_district_freeflow_min + inter_stop_freeflow_min x (orders - 1) + sum(service_allowance_min[brand, dock_type])` (no return leg)."""),
    M("## 2. Test the trip-time formula against the booklet's worked examples"),
    C("""allow = al.set_index(['brand', 'dock_type']).service_allowance_min; d = dt.set_index('district')
def minutes(district, brand, docks):
    return d.loc[district, 'depot_to_district_freeflow_min'] + d.loc[district, 'inter_stop_freeflow_min'] * (len(docks) - 1) + sum(allow[(brand, k)] for k in docks)
gampaha = minutes('Gampaha', 'Fresh', ['rear_dock', 'rear_dock', 'street'])
colombo = minutes('Colombo', 'Fresh', ['street'] * 4)
print('Gampaha trip (booklet example): %d min (booklet: 101)' % gampaha)
print('Colombo trip (booklet example): %d min (booklet: 112)' % colombo)
print('same vehicle, both Fresh trips: %d of 270 min (booklet: 213)' % (gampaha + colombo))
assert (gampaha, colombo) == (101, 112)
print('\\nservice allowances (min per stop):'); print(allow.unstack().round(0))"""),
    M("## 3. The solver's rule engine agrees with an independent re-implementation"),
    C("""t = trip_table(sub_team, scn, veh, dt, al)          # independent: straight from the CSVs
mism = 0
for _, r in t.iterrows():
    orders = [S.ORDERS[x] for x in r.orders.split()]
    mism += abs(S.trip_minutes(orders) - r.trip_min) > 1e-9
print('trips checked:', len(t), '| minute mismatches between solver rule engine and independent calculation:', int(mism))
assert mism == 0

# unit tests of the rule predicates on synthetic situations
reefer_truck = S.VEHICLES_ALL['VEH003']; amb_truck = [v for v in S.AVAILABLE['S1'] if v.type == 'truck' and v.temp == 'ambient'][0]; van = [v for v in S.AVAILABLE['S1'] if v.type == 'van'][0]
chilled = next(o for o in S.ORDERS.values() if o.chilled and o.parking == 'normal'); vo = next(o for o in S.ORDERS.values() if o.parking == 'van_only' and not o.chilled)
tests = {'chilled on ambient truck is rejected': not S.compatible(amb_truck, chilled), 'chilled on reefer truck is accepted': S.compatible(reefer_truck, chilled),
         'van_only outlet on a truck is rejected': not S.compatible(amb_truck, vo), 'van_only outlet on a van is accepted': S.compatible(van, vo)}
pd.Series(tests)"""),
    M("## 4. Which orders can an available vehicle carry at all? (unavoidable deferrals)"),
    C("""avail_v = S.AVAILABLE['S1']
rows = []
for o in S.ORDERS.values():
    ok = [v.vid for v in avail_v if S.compatible(v, o) and o.m3 <= v.m3_cap + 1e-9 and o.kg <= v.kg_cap + 1e-9 and S.trip_minutes([o]) <= S.BUDGET[S.category(o.brand)]]
    rows.append((o.ref, o.outlet, o.brand, o.district, round(o.m3, 2), len(ok)))
c = pd.DataFrame(rows, columns=['order_ref', 'outlet', 'brand', 'district', 'm3', 'vehicles_that_can_carry_it_alone'])
print(c.vehicles_that_can_carry_it_alone.describe().round(1).to_dict())
print('\\norders NO available vehicle can carry (unavoidable):'); c[c.vehicles_that_can_carry_it_alone == 0]"""),
    M("""**Finding.** Exactly one order is physically undeliverable today: S1-078 (OUT070, Style, 40.66 m3) is bigger than the largest compatible vehicle (38 m3). Orders cannot be split (R5), so this deferral is **unavoidable** regardless of policy. Everything else *can* be carried by at least one vehicle; whether it *is* depends on competition for scarce capacity."""),
    M("## 5. How many stops fit in a Fresh trip? (district reach)"),
    C("""rows = []
for dist in d.index.intersection(scn.district.unique()):
    for dock in ('rear_dock', 'street'):
        a = allow[('Fresh', dock)]; out, inter = d.loc[dist, 'depot_to_district_freeflow_min'], d.loc[dist, 'inter_stop_freeflow_min']
        rows.append((dist, dock, int(out), int(inter), int(a), int(np.floor((270 - out + inter) / (inter + a)))))
r = pd.DataFrame(rows, columns=['district', 'dock', 'outbound_min', 'inter_stop_min', 'handling_min', 'max_stops_in_270_min'])
r.pivot(index='district', columns='dock', values='max_stops_in_270_min').join(d.depot_to_district_freeflow_min.rename('outbound_min')).sort_values('outbound_min')"""),
    M("""**Finding.** A single Fresh trip to Puttalam leaves room for only a handful of stops and uses most of a vehicle's 270 minutes; nearby districts allow 9-10 stops. This is why *where* a chilled order is matters as much as how large it is."""),
])

# ------------------------------------------------------------------------------------------------------------- 03
nb("03_priorities_and_bottlenecks.ipynb", [
    M("""# Task 2B - 03 Priorities and bottlenecks
What does each order cost the business if it is not served today, and which resource actually limits service?"""),
    C(HEADER + SOLVER),
    M("""## 1. Two ways to express priority
**(a) The weighted score used by the first solver** (teammate's `solve_task2b.py`):
`value = 10 per stop + 1 x m3 + 5 if chilled + 20 if deferred yesterday + 4 x (days since last served - 1)`.

**(b) The written lexicographic policy** (the team's stated order): serve first the Fresh orders **deferred yesterday**, then **chilled** orders, then those with the highest **days since last served**; volume only as a tie-break. Each level is optimised first and then *fixed* before the next one is considered, so a lower level can never overrule a higher one.

The two coincide when no real trade-off exists and differ exactly where the day is tight; notebook 05 quantifies that."""),
    C("""orders = pd.DataFrame([dict(order_ref=o.ref, outlet=o.outlet, brand=o.brand, district=o.district, chilled=o.chilled, m3=o.m3,
                              deferred_yesterday=o.deferred_yday, days_since=o.days_since, value=S.order_value(o)) for o in S.ORDERS.values()])
print('weights:', dict(base=S.W_BASE, volume=S.W_VOLUME, chilled=S.W_CHILLED, deferred_yday=S.W_DEFERRED_YDAY, days_since=S.W_DAYS_SINCE))
print('\\nhighest-priority orders:'); print(orders.sort_values('value', ascending=False).head(8).round(1).to_string(index=False))
print('\\norders skipped yesterday:'); print(orders[orders.deferred_yesterday == 1].round(1).to_string(index=False))
print('\\ndays since last served:', orders.days_since.value_counts().sort_index().to_dict())"""),
    M("""**Finding.** Ten orders were already skipped yesterday (6 Fresh, 2 Style, 2 Tech), and five orders have gone **5 days** without a delivery. The highest priority of all is S1-083 (OUT074 Puttalam, chilled, skipped yesterday and 5 days unserved). Puttalam is also the farthest district (173 minutes outbound), which sets up the central tension of the day."""),
    M("## 2. The binding resource: reefer capacity, in volume, weight and time"),
    C("""reefers = [v for v in S.AVAILABLE['S1'] if v.temp == 'reefer']
chilled = [o for o in S.ORDERS.values() if o.chilled]
print('available reefers:', [(v.vid, v.type, v.m3_cap, v.kg_cap) for v in reefers])
vol_dem, kg_dem = sum(o.m3 for o in chilled), sum(o.kg for o in chilled)
vol_cap, kg_cap = sum(v.m3_cap for v in reefers), sum(v.kg_cap for v in reefers)
print('\\nchilled demand: %.1f m3, %.1f t (%d orders)' % (vol_dem, kg_dem / 1000, len(chilled)))
print('reefer capacity: %.1f m3 and %.1f t in ONE wave; %.1f m3 with the maximum two trips per vehicle' % (vol_cap, kg_cap / 1000, 2 * vol_cap))
print('=> even two perfect full loads per reefer carry %.0f%% of chilled volume' % (100 * 2 * vol_cap / vol_dem))"""),
    C("""# Minimum reefer minutes needed to serve EVERY chilled order (best possible partition into trips per district, ignoring who drives)
big_reefer = max(reefers, key=lambda v: v.m3_cap); van_reefer = [v for v in reefers if v.type == 'van'][0]
rows = []
for dist in sorted({o.district for o in chilled}):
    g = [o for o in chilled if o.district == dist]
    parts = X._best_partition(g, big_reefer, van_reefer, 'predawn')
    rows.append((dist, len(g), round(sum(o.m3 for o in g), 1), len(parts), sum(i[2] for _, i in parts)))
mt = pd.DataFrame(rows, columns=['district', 'chilled_orders', 'chilled_m3', 'min_trips', 'min_minutes'])
print(mt.to_string(index=False))
need_min, have_min = mt.min_minutes.sum(), len(reefers) * S.BUDGET['predawn']
print('\\nminutes needed to serve all chilled orders (a lower bound): %d | pre-dawn minutes the 4 reefers have: %d  -> %.0f%% of what is needed' % (need_min, have_min, 100 * have_min / need_min))
print('trips needed (lower bound):', int(mt.min_trips.sum()), '| trips available (4 reefers x 2):', 2 * len(reefers))"""),
    M("""**Finding - two independent bottlenecks, each sufficient on its own:**
- **Volume:** chilled demand exceeds even two full loads of every available reefer.
- **Time:** serving every chilled order needs more reefer minutes (and more trips) than the four reefers have in the 270-minute pre-dawn window.

Ambient demand is nowhere near binding (audit, notebook 01). Hence the policy question is *which chilled orders* wait, and there is a hard floor on how many must."""),
])

# ------------------------------------------------------------------------------------------------------------- 04
nb("04_exact_solver.ipynb", [
    M("""# Task 2B - 04 An exact solver
The first solver enumerates every two-trip day plan per reefer and searches the combinations by branch-and-bound. It stops at a 3,000,000-node limit (about 5 minutes on this machine), so its answer is *best found*, **not proven optimal**. Here we replace the search by exact optimisation.

## Formulation
**Stage R - reefers and chilled orders (MILP, solved with HiGHS).**
A *column* is "reefer *r* runs trip *S*", where *S* is a set of chilled orders from one district that fits the reefer's volume and weight and the 270-minute budget. Binary `x[r,S]`; constraints: each order in at most one trip; per reefer at most 2 trips and trip minutes <= 270. Priority is a list of **lexicographic levels**: each level is maximised, then fixed, then the next is optimised.

**Stage A - all other orders on ambient vehicles (exact subset DP).**
Every non-chilled order that can be served is served; for each brand+district group we find the cheapest split into feasible trips by dynamic programming over order subsets (at most 14 orders), minimising (1) stops arriving after their window closes, (2) vehicles used, (3) minutes. Trucks are interchangeable except for capacity, so trips are planned on the largest truck and then matched to concrete trucks (verified).

The two stages share no vehicles or orders, so combining their lexicographic optima gives the lexicographic optimum of the whole problem."""),
    C(HEADER + SOLVER),
    M("## 1. Stage R: the reefer problem"),
    C("""vehicles = list(S.AVAILABLE['S1']); scope = list(S.ORDERS.values())
reefers = [v for v in vehicles if v.temp == 'reefer']; chilled = [o for o in scope if o.chilled]
plan_f, lv_f, sec_f, n_f = X.reefer_stage(reefers, chilled, 'weighted', scope, late_aware=False)
print('FAST model   : %5d columns (reefer x one feasible trip) | solve %.1fs' % (n_f, sec_f))
for lv in lv_f: print('     level:', lv[1], '=', round(lv[2], 3))
plan_r, lv_r, sec_r, n_cols = X.reefer_stage(reefers, chilled, 'weighted', scope)
print('\\nCLOCK-AWARE  : %5d columns (a reefer day plan = ordered pair of trips) | solve %.1fs' % (n_cols, sec_r))
for lv in lv_r: print('     level:', lv[1], '=', round(lv[2], 3))"""),
    M("""**Result.** The exact optimum of the first solver's own objective is **539.723** - exactly the value its branch-and-bound found, in both models. So the first solver's reefer plan *value* was already optimal; what was missing was the **proof** (HiGHS reports "optimal" only when the gap is closed) and a much faster way to get it (about 3 s for the fast model vs about 5 min).

The second, **clock-aware** model goes one step further: among all plans with that same optimal value it picks the one whose reefer stops arrive after their windows close least often, because every column is a whole day plan whose clock is exact. It costs about 50 s, so sensitivity sweeps (notebook 05) use the fast model, which gives the same served set."""),
    C("""rows = []
for vid, ts in sorted(plan_r.items()):
    for k, tr in enumerate(ts, 1):
        rows.append((vid, S.VEHICLES_ALL[vid].type, k, tr[0].district, len(tr), round(sum(o.m3 for o in tr), 1), S.trip_minutes(tr), ' '.join(o.ref for o in tr)))
print('clock-aware reefer plan (weighted policy), trips listed in the order the plan runs them')
pd.DataFrame(rows, columns=['reefer', 'type', 'trip', 'district', 'stops', 'm3', 'trip_min', 'orders'])"""),
    M("## 2. Stage A: window-aware planning of everything else"),
    C("""ambient = [v for v in vehicles if v.temp == 'ambient']; other = [o for o in scope if not o.chilled and o.ref != 'S1-078']
plan_a, lv_a, sec_a, n_trips = X.ambient_stage(ambient, other, 'weighted', scope)
print('solve time %.2fs' % sec_a)
for lv in lv_a: print('  ', lv[1], '=', lv[2])
rows = []
for vid, ts in sorted(plan_a.items()):
    for k, tr in enumerate(ts, 1):
        rows.append((vid, S.VEHICLES_ALL[vid].type, k, tr[0].brand, tr[0].district, len(tr), round(sum(o.m3 for o in tr), 1), S.trip_minutes(tr)))
amb = pd.DataFrame(rows, columns=['vehicle', 'type', 'trip', 'brand', 'district', 'stops', 'm3', 'trip_min']); amb"""),
    M("""**Finding.** With 24 ambient vehicles available and only about 17 trips needed, nothing forces a long trip: the planner splits busy districts across several trucks so that **no stop arrives after its window closes**. (The first solver ran two 10-stop trips of 249-270 minutes while 15 ambient vehicles stayed in the depot, which put 4 stops past their windows.)"""),
    M("## 3. Full solve, both policies"),
    C("""res = {}
for pol in ('weighted', 'lexicographic'):
    t = time.time(); res[pol] = X.solve_exact(pol); print(pol, 'solved in %.1fs' % (time.time() - t))
pd.DataFrame({p: X.kpis(r) for p, r in res.items()})"""),
    C("""# reference: the first solver's submitted allocation, same KPIs from the CSV
st, late_team = simulate_submission(sub_team, S)
srv = [S.ORDERS[r] for r in sub_team[sub_team.decision == 'served'].order_ref]
print('first solver (submitted file): served', len(srv), '| m3 %.1f' % sum(o.m3 for o in srv), '| chilled', sum(o.chilled for o in srv),
      '| vehicles', sub_team[sub_team.decision == 'served'].vehicle_id.nunique(), '| late stops (clock simulation):', late_team)"""),
    M("""**Reading.** Solving the *same weighted score* exactly reproduces the first solver's service outcome (79 orders, 328.3 m3, 21 chilled) - confirming it was optimal for that objective - in about 4 seconds instead of 5 minutes, and with the ambient trips re-planned. The **lexicographic** policy is different: it serves all ten orders skipped yesterday, at the cost of two chilled orders. Notebook 05 explains that trade-off.

The few stops still marked late in the exact plans are *second trips of reefers*: with the booklet's 270-minute budget a second Fresh trip can start after 08:00 in a literal clock simulation (06 asserts that this is true of every remaining late stop). The booklet says its budgets already allow for the return journey and its feasibility rules do not include windows, so we report these stops rather than treat them as plan defects."""),
])

# ------------------------------------------------------------------------------------------------------------- 05
nb("05_policies_and_whatifs.ipynb", [
    M("""# Task 2B - 05 Policy trade-offs and what-ifs
This notebook turns the day into a decision: what exactly does each priority policy buy and cost, and what would releasing a workshop reefer change?
Every number comes from an exact solve."""),
    C(HEADER + SOLVER),
    M("## 1. The policy frontier"),
    C("""pols = ['weighted', 'lexicographic', 'chilled_count', 'chilled_volume', 'max_orders']
res = {p: X.solve_exact(p, late_aware=False) for p in pols}      # fast reefer model: same served sets, no clock look-ahead
fr = pd.DataFrame({p: X.kpis(r) for p, r in res.items()}).T
fr.insert(0, 'rule', [X.POLICIES[p] for p in pols])
fr.to_csv(METRICS / 'exact_policy_frontier.csv'); fr[['rule', 'served', 'm3', 'chilled_served', 'chilled_m3', 'deferred_yday_served', 'priority_value']]"""),
    C("""served_sets = {p: r.served_refs() for p, r in res.items()}
allrefs = set(S.ORDERS)
for p in pols: print('%-15s deferred: %s' % (p, sorted(allrefs - served_sets[p])))"""),
    M("""**Reading.** Five sensible rules give different answers only in the *chilled* column - everything ambient is served, apart from the physically impossible S1-078. The cheapest policy in orders is the weighted score / most-orders rule; the policy that never leaves a skipped outlet unserved twice costs two chilled orders and about 14 m3. There is no allocation that does both."""),
    M("""## 2. The deferred-yesterday trade-off, made exact
Outlet OUT074 (S1-083) was skipped yesterday and has now gone 5 days without a delivery. Serving it means a 173-minute outbound trip to Puttalam for a single 8.7 m3 stop, which uses 188 of one reefer's 270 minutes."""),
    C("""w, l = served_sets['weighted'], served_sets['lexicographic']
print('only in the weighted plan (served there, deferred under the lexicographic policy):', sorted(w - l))
print('only in the lexicographic plan:', sorted(l - w))
rows = [(r, S.ORDERS[r].outlet, S.ORDERS[r].district, round(S.ORDERS[r].m3, 1), S.ORDERS[r].deferred_yday, S.ORDERS[r].days_since) for r in sorted((w ^ l))]
pd.DataFrame(rows, columns=['order_ref', 'outlet', 'district', 'm3', 'deferred_yesterday', 'days_since'])"""),
    M("""**Trade-off in one line.** To serve the outlet that has waited longest (and was skipped yesterday), the plan trades S1-003, S1-071, S1-073 and S1-075 for S1-083 and S1-056: four chilled orders wait instead of two. The four that wait are *first-time* misses (none was skipped yesterday). Nothing in the data says which is worse for the business; that is a **human decision**. The next section shows how close the weighted score already is to making the other choice."""),
    M("## 3. Where does the weighted score flip?"),
    C("""def serves_083(w):
    r = X.solve_exact('weighted', weights=dict(deferred_yday=w), late_aware=False); return 'S1-083' in r.served_refs(), X.kpis(r)
rows = []
for wv in (0, 10, 20, 25, 30, 40, 60, 100):
    ok, k = serves_083(wv); rows.append((wv, ok, k['chilled_served'], k['served'], k['deferred_yday_served']))
flip = pd.DataFrame(rows, columns=['deferred_yday_weight', 'S1-083 served', 'chilled served', 'orders served', 'deferred-yesterday served']); flip"""),
    C("""lo, hi = 20.0, 30.0
for _ in range(7):
    mid = (lo + hi) / 2
    if serves_083(mid)[0]: hi = mid
    else: lo = mid
print('the weighted score serves S1-083 once the deferred-yesterday weight is about %.1f (first solver used 20, i.e. %.0f%% lower)' % (hi, 100 * (1 - 20 / hi)))
FLIP = hi
json.dump({'deferred_yday_flip_weight': round(hi, 2), 'first_solver_weight': 20}, open(METRICS / 'exact_flip.json', 'w'))"""),
    M("""**Finding.** The first solver's weight of 20 sits below the point where the score would have served the five-day-unserved outlet (the threshold printed above, a little under 30). A weight roughly 40% larger gives the lexicographic answer. So the written priority order ("deferred yesterday first") and the numerical weights disagree by a small margin - which is exactly why the written policy should be implemented as an explicit ordering, not as tuned weights."""),
    M("## 4. Value of releasing a workshop reefer"),
    C("""ws_reefers = [v for v in fleet[fleet.status == 'in_workshop'].vehicle_id if S.VEHICLES_ALL[v].temp == 'reefer']
rows = []
for pol in ('weighted', 'lexicographic'):
    base = X.kpis(res[pol]); base_served = res[pol].served_refs()
    rows.append((pol, 'none (today)', '', base['chilled_served'], base['chilled_m3'], base['served'], 0))
    for vid in ws_reefers:
        r = X.solve_exact(pol, extra_vehicles=(vid,), late_aware=False); k = X.kpis(r)
        rows.append((pol, vid, '%s %.1f m3' % (S.VEHICLES_ALL[vid].type, S.VEHICLES_ALL[vid].m3_cap), k['chilled_served'], k['chilled_m3'], k['served'], k['served'] - base['served']))
wi = pd.DataFrame(rows, columns=['policy', 'released reefer', 'class', 'chilled orders served', 'chilled m3 served', 'orders served', 'net orders gained'])
wi.to_csv(METRICS / 'exact_whatif_workshop_reefer.csv', index=False); wi"""),
    M("""**Finding.** Releasing **any one of the four workshop reefer trucks** (VEH001, VEH002, VEH004, VEH005) lifts both policies to the same result - 23 of 26 chilled orders and 81 orders served - so with one more truck the weighted score and the "never miss twice" policy stop disagreeing at all. The workshop reefer van (VEH035) helps less. This is the cheapest lever for tomorrow's peak and is quantified here exactly, not guessed.

## What needs a human decision
1. **Policy:** weighted score (79 orders, one outlet missed twice) or lexicographic "never miss an outlet twice" (77 orders, none missed twice). Section 3 shows the margin between them is small.
2. **Count or volume** for the "chilled" level: we rank by number of chilled orders (outlets kept in stock); ranking by chilled m3 (`chilled_volume` above) gives a different set.
3. **Which workshop reefer to repair first**, and whether a hired vehicle is an option (section 4 gives the value of each)."""),
])

# ------------------------------------------------------------------------------------------------------------- 06
nb("06_validation_and_comparison.ipynb", [
    M("""# Task 2B - 06 Validation and comparison
Every candidate allocation is checked two independent ways - the **unmodified official validator** and an **independent re-implementation of all rules** - and compared with the first solver's submitted file."""),
    C(HEADER + SOLVER),
    C("""cands = {'first solver (submitted)': sub_team}
for pol in ('weighted', 'lexicographic'):
    cands['exact: ' + pol] = X.to_submission(X.solve_exact(pol))
tmp = INTERIM / 'candidates'; tmp.mkdir(exist_ok=True)
rows, checks = [], {}
for name, s in cands.items():
    p = tmp / (name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '') + '.csv'); s.to_csv(p, index=False)
    ok, msg = official_check(p)
    ind = independent_check(pd.read_csv(p), scn, fleet, veh, dt, al)
    checks[name] = ind.set_index('check').violations
    rows.append((name, ok, int(ind.violations.sum())))
pd.DataFrame(rows, columns=['candidate', 'official validator passed', 'independent-check violations'])"""),
    C("""pd.DataFrame(checks)"""),
    M("""All three allocations are feasible under both checks. The independent re-implementation shares no code with the solver or with the official script, so agreement is meaningful."""),
    M("## Head-to-head"),
    C("""def summary(name, s):
    m = s[s.decision == 'served'].merge(scn, on=['scenario', 'order_ref', 'outlet_id'])
    st, late = simulate_submission(s, S)
    t = trip_table(s, scn, veh, dt, al)
    return dict(served=len(m), deferred=int((s.decision == 'deferred').sum()), m3_served=round(m.order_volume_m3.sum(), 1), chilled_served=int((m.temp_requirement == 'chilled').sum()),
                vehicles=m.vehicle_id.nunique(), trips=len(t), late_stops=late, max_trip_min=int(t.trip_min.max()), longest_trip_stops=int(t.stops.max()),
                deferred_yesterday_served=int(m.deferred_yesterday.sum()))
cmp_ = pd.DataFrame({n: summary(n, s) for n, s in cands.items()}); cmp_.to_csv(METRICS / 'exact_comparison.csv'); cmp_"""),
    M("""**Reading.**
- *Same service outcome, better execution:* the exact weighted plan serves exactly what the first solver served, but with no avoidable late stop and no 10-stop 4-hour ambient trips.
- *The late stops that remain* in the exact plans (the cell above asserts that every one of them is on a reefer's **second Fresh trip**) arise because a literal clock starts that trip only after the first trip's return leg. The booklet states its budgets already allow for the return, and its feasibility rules do not include windows, so these are not defects of the plan - but they are the one place where a literal clock disagrees with the booklet's abstraction, and we report them rather than hide them."""),
    M("## Which stops are late, and why"),
    C("""for name in ('first solver (submitted)', 'exact: weighted', 'exact: lexicographic'):
    st, late = simulate_submission(cands[name], S)
    t = trip_table(cands[name], scn, veh, dt, al)[['vehicle_id', 'trip_id', 'vehicle', 'district', 'stops', 'trip_min']]
    x = st[st.late == 1].merge(t, on=['vehicle_id', 'trip_id'])[['order_ref', 'vehicle_id', 'trip_id', 'vehicle', 'district', 'stops', 'depart_depot', 'arrive', 'window']]
    other = x[~((x.trip_id == 2) & x.vehicle.str.contains('reefer'))]
    print('%-26s late stops: %d | of which NOT on a reefer second trip: %d' % (name, late, len(other)))
    if name != 'first solver (submitted)':
        assert len(other) == 0, 'an exact plan has a late stop outside reefer second trips'
    print(x.to_string(index=False)); print()"""),
    M("## Deferral analysis: unavoidable vs chosen"),
    C("""rows = []
for name, s in cands.items():
    for r in s[s.decision == 'deferred'].order_ref:
        o = S.ORDERS[r]; why = S.unavoidable_reason(o, S.AVAILABLE['S1'])
        rows.append((name, r, o.outlet, o.brand, o.district, 'chilled' if o.chilled else 'ambient', round(o.m3, 1), o.deferred_yday, o.days_since, 'unavoidable' if why else 'capacity (reefer)'))
dd = pd.DataFrame(rows, columns=['candidate', 'order_ref', 'outlet', 'brand', 'district', 'type', 'm3', 'deferred_yesterday', 'days_since', 'kind'])
dd.to_csv(METRICS / 'exact_deferral_comparison.csv', index=False); dd.sort_values(['candidate', 'order_ref'])"""),
    M("""**Finding.** One deferral (S1-078) is physically unavoidable under every policy. All other deferrals are reefer-capacity deferrals, and *which* chilled orders wait is the policy choice."""),
])

# ------------------------------------------------------------------------------------------------------------- 07
nb("07_submit_and_policy.ipynb", [
    M("""# Task 2B - 07 Final allocation, schedule and policy
Writes the exact-solver allocations as **new** files (the teammate's `submission_task2b.csv` is not touched), validates them again from disk, builds the trip plan and stop schedule, and generates the one-page policy from the solved numbers."""),
    C(HEADER + SOLVER),
    C("""RECOMMENDED = 'lexicographic'      # the team's written policy: deferred yesterday > chilled > days since served
res = {p: X.solve_exact(p) for p in ('lexicographic', 'weighted')}
out_files = {}
for p, r in res.items():
    s = X.to_submission(r); f = SUBMISSION_DIR / 'predictions' / f'submission_task2b_exact_{p}.csv'; s.to_csv(f, index=False); out_files[p] = f
    ok, msg = official_check(f); print(p, '->', f.name, '|', msg.splitlines()[-1])
    assert ok and independent_check(pd.read_csv(f), scn, fleet, veh, dt, al).violations.sum() == 0
tmpl = pd.read_csv(RAW / 'templates' / 'submission_task2b.csv')
sub = pd.read_csv(out_files[RECOMMENDED])
assert list(sub.columns) == list(tmpl.columns) and list(sub.order_ref) == list(tmpl.order_ref) and list(sub.outlet_id) == list(tmpl.outlet_id)
print('\\ncolumns and order_ref/outlet_id order match the template; rows:', len(sub))"""),
    M("## Trip plan and stop schedule (recommended policy)"),
    C("""t = trip_table(sub, scn, veh, dt, al).sort_values(['vehicle_id', 'trip_id'])
st, late = simulate_submission(sub, S)
t.round(3).to_csv(METRICS / 'exact_trip_plan.csv', index=False); st.to_csv(METRICS / 'exact_stop_schedule.csv', index=False)
print('trips:', len(t), '| vehicles:', t.vehicle_id.nunique(), '| late stops (clock simulation):', late)
t[['vehicle_id', 'trip_id', 'vehicle', 'brand', 'district', 'stops', 'm3', 'm3_cap', 'kg', 'kg_cap', 'trip_min', 'bucket_used', 'bucket_budget']].round(1)"""),
    C("""fig, ax = plt.subplots(figsize=(11, 6))
vids = sorted(st.vehicle_id.unique()); col = {'Fresh': 'C0', 'Style': 'C2', 'Tech': 'C1'}
for i, v in enumerate(vids):
    for k, g in st[st.vehicle_id == v].groupby('trip_id'):
        b = S.ORDERS[g.order_ref.iloc[0]].brand
        a, z = to_min(g.depart_depot.iloc[0]), max(to_min(x) for x in g.leave)
        ax.barh(i, z - a, left=a, color=col[b], edgecolor='k', alpha=.85)
        for _, r in g.iterrows():
            ax.plot(to_min(r.arrive), i, 'rx' if r.late else 'k.', ms=5)
ax.set_yticks(range(len(vids))); ax.set_yticklabels(vids, fontsize=7); ax.set_xlabel('minutes after midnight (03:30 = 210, 08:00 = 480)'); ax.axvline(480, color='r', ls='--', lw=1)
ax.set_title('Day plan: one bar per trip (blue Fresh, green Style, orange Tech); dots = arrivals, red x = after window close'); plt.tight_layout(); plt.savefig(FIGURES / '07_day_plan.png'); plt.show()"""),
    C("""u = t.assign(m3_pct=(t.m3_util * 100).round(0), kg_pct=(t.kg_util * 100).round(0))
print('volume utilisation of trips: median %.0f%%, max %.0f%% | weight: median %.0f%%, max %.0f%%' % (u.m3_pct.median(), u.m3_pct.max(), u.kg_pct.median(), u.kg_pct.max()))
(t.groupby('vehicle').agg(trips=('trip_id', 'size'), m3=('m3', 'sum')).round(1)).T"""),
    M("## One-page policy (generated from the solved numbers)"),
    C("""from textwrap import dedent
k = {p: X.kpis(r) for p, r in res.items()}; kk = k[RECOMMENDED]; kw = k['weighted']
flip_w = json.load(open(METRICS / 'exact_flip.json'))['deferred_yday_flip_weight']
reefers = [v for v in S.AVAILABLE['S1'] if v.temp == 'reefer']; chilled = [o for o in S.ORDERS.values() if o.chilled]
served = res[RECOMMENDED].served_refs(); deferred = [S.ORDERS[r] for r in sorted(set(S.ORDERS) - served)]
rows = '\\n'.join('| %s | %s | %s | %s | %.1f | %d | %d | %s |' % (o.ref, o.outlet, o.district, 'chilled' if o.chilled else 'ambient', o.m3, o.deferred_yday, o.days_since,
                 'unavoidable: larger than any compatible vehicle' if S.unavoidable_reason(o, S.AVAILABLE['S1']) else 'reefer capacity') for o in deferred)
only_w = sorted(res['weighted'].served_refs() - served)
txt = dedent(f'''\\
# Task 2B - Peak-day allocation policy, scenario S1 (Peliyagoda) - exact solver

**Result ({RECOMMENDED} policy):** {kk['served']}/85 orders served, {kk['m3']} of 409.9 m3; {kk['chilled_served']}/26 chilled orders; all {kk['deferred_yday_served']} outlets skipped yesterday served; {kk['trips']} trips on {kk['vehicles']} vehicles; no ambient stop is planned after its window closes; {late} stop(s) on reefers' second Fresh trips arrive late only by a literal clock (the booklet's 270-minute budget already allows for the return leg). Passes the official `check_allocation.py` and an independent re-check of every rule. Solved exactly (reefer MILP, proven optimal; ambient subset DP): about a minute for the clock-aware model, about 4 seconds for the fast model.

## 1. What limited service
- **Reefer volume:** chilled demand {sum(o.m3 for o in chilled):.1f} m3 vs {sum(v.m3_cap for v in reefers):.1f} m3 for the {len(reefers)} available reefers per wave ({2 * sum(v.m3_cap for v in reefers):.1f} m3 even with two full trips each; five reefers are in the workshop).
- **Pre-dawn minutes:** each vehicle has 270 min; serving every chilled order would need more reefer minutes and trips than exist. Far districts (Puttalam 173 min, Matara 137, Galle 103 outbound) consume a large share of one vehicle's window for one or two stops.
- Ambient trucks and vans are not binding (24 vehicles for about 17 trips).

## 2. Priority policy (strict order, each level optimised and then fixed)
1. Fresh orders **deferred yesterday** - never miss an outlet twice in a row.
2. **Chilled** orders - the most chilled orders served (outlets kept in stock for the festival build-up).
3. Highest **days since last served**.
4. Volume delivered, as a tie-break.
Ambient orders are all served unless physically impossible. Trips are planned so that no stop arrives after its window closes; reefers and constraints are as in the booklet (brand+district per trip, reefer for chilled, vans for van_only, capacity, 270/480-minute budgets, whole orders, at most 2 trips).

## 3. Deferrals
| Order | Outlet | District | Type | m3 | Def. yday | Days | Why |
|---|---|---|---|---|---|---|---|
{rows}

- **Unavoidable (1):** S1-078 is larger than any compatible vehicle (cannot be split); ask the outlet to split it or hire a larger truck.
- **Chosen (reefer capacity):** which chilled orders wait is our policy decision. Serving the Puttalam outlet skipped yesterday costs a 188-minute trip, so {len(deferred) - 1} nearer/larger orders wait instead: they become first-time misses.

## 4. Cost of the choice (exact comparison)
Weighted score instead of the written order: {kw['served']} orders, {kw['m3']} m3, {kw['chilled_served']} chilled, but {kw['deferred_yday_total'] - kw['deferred_yday_served']} outlet skipped for a second day ({only_w}). The score's deferred-yesterday weight (20) is below the value (about {flip_w:.0f}) at which it would choose the same plan.

## 5. Recommendations
- Release a reefer from the workshop: the what-if analysis (notebook 05) gives the exact orders recovered for each of VEH001, VEH002, VEH004, VEH005, VEH035.
- Serve tomorrow's carried-over chilled orders first (they will be `deferred_yesterday = 1`).
- Ask OUT070 to split its 40.7 m3 Style order.
''')
(SUBMISSION_DIR / 'docs' / 'Task2B_policy_exact.md').write_text(txt, encoding='utf-8')
print(txt[:2600])"""),
    M("""## How to use these files
- `predictions/submission_task2b_exact_lexicographic.csv` follows the team's written policy; `predictions/submission_task2b_exact_weighted.csv` reproduces the first solver's outcome with the window-aware plan.
- The teammate's `submission_task2b.csv` is unchanged. To submit one of the exact files, copy it over `submission_task2b.csv` **after** the team decides the policy (human decision, see 05)."""),
])
print("wrote notebooks to", OUT)
