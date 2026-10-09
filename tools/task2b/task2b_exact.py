"""Exact (proven-optimal) allocation for Task 2B using a trip-column MILP (scipy / HiGHS).

Why this exists next to solve_task2b.py
  * The heuristic solver's reefer stage hits its 3M-node limit (about 5 minutes) and cannot prove optimality. Here every
    stage is a mixed-integer program over *trip columns* (a column = one vehicle (class) running one feasible trip), solved
    to proven optimality in seconds.
  * Policies are explicit LEXICOGRAPHIC objectives (levels solved in order, each fixed at its optimum before the next), so
    the written priority order is exactly what the solver does - no hand-tuned weights.
  * Ambient trips are planned window-aware: with 24 ambient vehicles and ~13 trips needed, there is no reason to run a
    10-stop trip that arrives after the store opens.

The rule engine (compatible / trip_ok / trip_minutes / simulate) is the teammate's solve_task2b.py, used as a library; the
result is then verified independently by task2b_common.independent_check and the unmodified official validator.

Model
  Stage R (reefers + chilled orders): x[r,S] = reefer r runs trip S (S = chilled orders of ONE district, capacity + time
      feasible). Per order <= 1, per reefer <= 2 trips and sum of trip minutes <= 270.
  Stage A (all non-chilled orders on ambient vehicles): x[c,S] = one vehicle of class c runs trip S (one brand+district).
      Per order <= 1, per class at most n_c Fresh trips and n_c daytime trips (a vehicle can run one of each), u_c = vehicles
      of the class actually used. Reefers are reserved for chilled orders (checked: every serviceable ambient order is served).
  The two stages are independent, so lexicographic optima of the stages combine into the lexicographic optimum overall.
"""
import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, lil_matrix

from task2b_common import import_solver

S = import_solver()
SCENARIO = "S1"
TOL = 1e-6

# ----------------------------------------------------------------------------------------------------------- policies


def policy_levels(name, orders, weights=None):
    """Lexicographic levels (label, {order_ref: value}) - all MAXIMISED, in priority order."""
    chilled = {o.ref: float(o.chilled) for o in orders}
    dy = {o.ref: float(o.deferred_yday) for o in orders}
    days = {o.ref: float(max(0, o.days_since - 1)) for o in orders}
    vol = {o.ref: o.m3 for o in orders}
    chvol = {o.ref: (o.m3 if o.chilled else 0.0) for o in orders}
    allone = {o.ref: 1.0 for o in orders}
    if name == "weighted":          # the teammate's single weighted score, solved exactly
        return [("priority value (weighted score)", {o.ref: S.order_value(o, weights) for o in orders})]
    if name == "lexicographic":     # the written policy: deferred yesterday > chilled > days since served
        return [("orders deferred yesterday served", dy), ("chilled orders served", chilled),
                ("days-since-served relief", days), ("volume served (m3)", vol)]
    if name == "chilled_count":
        return [("chilled orders served", chilled), ("deferred-yesterday served", dy), ("days-since-served relief", days), ("volume served (m3)", vol)]
    if name == "chilled_volume":
        return [("chilled volume served (m3)", chvol), ("deferred-yesterday served", dy), ("days-since-served relief", days), ("orders served", allone)]
    if name == "max_orders":
        return [("orders served", allone), ("volume served (m3)", vol)]
    raise ValueError(name)


POLICIES = {"weighted": "weighted score (as in solve_task2b.py), solved exactly",
            "lexicographic": "deferred-yesterday > chilled > days-since-served (written policy)",
            "chilled_count": "most chilled orders served first",
            "chilled_volume": "most chilled m3 served first",
            "max_orders": "most orders served first"}

# ----------------------------------------------------------------------------------------------------- MILP plumbing


def lex_milp(n, rows, lo, hi, levels, integrality=None, ub=None, time_limit=120):
    """Solve `levels` lexicographically. levels = [(name, c, 'max'|'min')] with c a length-n vector.
    rows: list of (dict col->coef); lo/hi: row bounds. Returns (x, [(name, optimum)], total_seconds)."""
    A = lil_matrix((len(rows), n))
    for i, r in enumerate(rows):
        for j, cf in r.items():
            A[i, j] = cf
    cons = [LinearConstraint(A.tocsr(), lo, hi)]
    integ = np.ones(n) if integrality is None else integrality
    bnd = Bounds(0, 1 if ub is None else ub)
    out, x, t0 = [], None, time.time()
    for name, c, sense in levels:
        c = np.asarray(c, float)
        obj = -c if sense == "max" else c
        res = milp(obj, constraints=cons, integrality=integ, bounds=bnd, options=dict(time_limit=time_limit))
        if res.status != 0:
            raise RuntimeError(f"level '{name}': {res.message}")
        val = float(c @ res.x)
        out.append((name, val))
        x = res.x
        # fix this level at its optimum before the next one
        row = csr_matrix(c.reshape(1, -1))
        cons.append(LinearConstraint(row, val - TOL * (1 + abs(val)), np.inf) if sense == "max" else LinearConstraint(row, -np.inf, val + TOL * (1 + abs(val))))
    return x, out, time.time() - t0


@dataclass
class Result:
    policy: str
    trips: dict = field(default_factory=dict)          # vehicle_id -> list of trips (list of S.Order), trip_id order
    levels: list = field(default_factory=list)         # [(stage, name, value)]
    seconds: float = 0.0
    columns: int = 0

    def served_refs(self):
        return {o.ref for ts in self.trips.values() for t in ts for o in t}


# ---------------------------------------------------------------------------------------------------------- Stage R


def _reefer_stage_late_aware(reefers, chilled, policy, orders_all, weights=None):
    """Chilled orders on reefers. A column is one reefer's whole DAY PLAN: one trip, or two trips in a fixed order, so the
    clock (first trip leaves 03:30, the second after the first returns) - and therefore the number of stops that arrive after
    their window closes - is exact for every column. Levels: the policy levels, then fewest late stops, trips, minutes."""
    by_d = defaultdict(list)
    for o in chilled:
        by_d[o.district].append(o)
    cols = []          # (vehicle id, trips (tuple of tuples of refs), minutes, late stops)
    for v in reefers:
        single = []
        for d, os_ in by_d.items():
            elig = [o for o in os_ if S.compatible(v, o)]
            for k in range(1, len(elig) + 1):
                for sub in itertools.combinations(elig, k):
                    sub = list(sub)
                    if S.trip_ok(v, sub) and S.trip_minutes(sub) <= S.BUDGET["predawn"] + S.EPS:
                        single.append(sub)
        for t in single:
            cols.append((v.vid, (tuple(o.ref for o in t),), S.trip_minutes(t), S.simulate([t])[1]))
        for t1, t2 in itertools.combinations(single, 2):
            if {o.ref for o in t1} & {o.ref for o in t2}:
                continue
            m = S.trip_minutes(t1) + S.trip_minutes(t2)
            if m > S.BUDGET["predawn"] + S.EPS:
                continue
            # two Fresh trips on one vehicle: run them in the order with fewer late stops
            best = min(((S.simulate([a, c])[1], a, c) for a, c in ((t1, t2), (t2, t1))), key=lambda z: z[0])
            cols.append((v.vid, (tuple(o.ref for o in best[1]), tuple(o.ref for o in best[2])), m, best[0]))
    n = len(cols)
    ridx = {o.ref: i for i, o in enumerate(chilled)}
    per = [defaultdict(float) for _ in chilled]
    for j, (vid, trips, m, l) in enumerate(cols):
        for tr in trips:
            for r in tr:
                per[ridx[r]][j] = 1
    rows, lo, hi = list(per), [0] * len(chilled), [1] * len(chilled)
    for v in reefers:                                    # at most one day plan per reefer
        rows.append({j: 1 for j, c in enumerate(cols) if c[0] == v.vid}); lo.append(0); hi.append(1)
    levels = []
    for name, vf in policy_levels(policy, orders_all, weights):
        levels.append((name, [sum(vf[r] for tr in trips for r in tr) for (_, trips, _, _) in cols], "max"))
    levels.append(("reefer stops arriving after window close (fewer is better)", [c[3] for c in cols], "min"))
    levels.append(("reefer trips (fewer is better)", [len(c[1]) for c in cols], "min"))
    levels.append(("reefer minutes (fewer is better)", [c[2] for c in cols], "min"))
    x, out, sec = lex_milp(n, rows, lo, hi, levels)
    plan = defaultdict(list)
    for j in range(n):
        if x[j] > 0.5:
            for tr in cols[j][1]:
                plan[cols[j][0]].append([S.ORDERS[r] for r in tr])
    return plan, [("reefers", a, b) for a, b in out], sec, n


def _reefer_stage_fast(reefers, chilled, policy, orders_all, weights=None):
    """Fast formulation (about 300 columns, about 3 s): a column is one reefer running ONE trip, with per-reefer constraints
    'at most 2 trips' and 'trip minutes <= 270'. Same served set as the lateness-aware model, but it does not look at the clock,
    so use it for sensitivity sweeps where only the served set matters."""
    cols = []
    by_d = defaultdict(list)
    for o in chilled:
        by_d[o.district].append(o)
    for v in reefers:
        for d, os_ in by_d.items():
            elig = [o for o in os_ if S.compatible(v, o)]
            for k in range(1, len(elig) + 1):
                for sub in itertools.combinations(elig, k):
                    sub = list(sub)
                    if S.trip_ok(v, sub) and S.trip_minutes(sub) <= S.BUDGET["predawn"] + S.EPS:
                        cols.append((v.vid, tuple(o.ref for o in sub), S.trip_minutes(sub)))
    n = len(cols)
    ridx = {o.ref: i for i, o in enumerate(chilled)}
    per = [defaultdict(float) for _ in chilled]
    for j, (vid, refs, m) in enumerate(cols):
        for r in refs:
            per[ridx[r]][j] = 1
    rows, lo, hi = list(per), [0] * len(chilled), [1] * len(chilled)
    for v in reefers:
        rows.append({j: 1 for j, c in enumerate(cols) if c[0] == v.vid}); lo.append(0); hi.append(S.MAX_TRIPS)
        rows.append({j: c[2] for j, c in enumerate(cols) if c[0] == v.vid}); lo.append(0); hi.append(S.BUDGET["predawn"])
    levels = []
    for name, vf in policy_levels(policy, orders_all, weights):
        levels.append((name, [sum(vf[r] for r in refs) for (_, refs, _) in cols], "max"))
    levels.append(("reefer trips (fewer is better)", [1.0] * n, "min"))
    levels.append(("reefer minutes (fewer is better)", [c[2] for c in cols], "min"))
    x, out, sec = lex_milp(n, rows, lo, hi, levels)
    plan = defaultdict(list)
    for j in range(n):
        if x[j] > 0.5:
            plan[cols[j][0]].append([S.ORDERS[r] for r in cols[j][1]])
    return plan, [("reefers", a, b) for a, b in out], sec, n


def reefer_stage(reefers, chilled, policy, orders_all, weights=None, late_aware=True):
    """Stage R dispatcher: lateness-aware day-plan model (default, about 50 s) or the fast model (about 3 s)."""
    f = _reefer_stage_late_aware if late_aware else _reefer_stage_fast
    return f(reefers, chilled, policy, orders_all, weights)


# ---------------------------------------------------------------------------------------------------------- Stage A


def _best_partition(group, truck_v, van_v, bucket):
    """Cheapest way to split `group` (orders of ONE brand+district) into feasible trips - exact subset DP.

    Trip cost = 1e6 x stops arriving after window close + 1e3 (one vehicle) + trip minutes, i.e. strictly ordered
    preferences: no late stops >> fewest vehicles >> fewest minutes. A trip containing a van_only outlet is a van trip,
    every other trip is planned on the largest truck (matched to a concrete truck afterwards).
    Feasibility is monotone (dropping an order never breaks a trip), so subset sizes stop growing once none fits.
    """
    n = len(group)
    info = {}                                   # mask -> (cost, late, minutes, is_van)
    for k in range(1, n + 1):
        any_ok = False
        for idx in itertools.combinations(range(n), k):
            trip = [group[i] for i in idx]
            is_van = any(o.parking == "van_only" for o in trip)
            v0 = van_v if is_van else truck_v
            mins = S.trip_minutes(trip)
            if v0 is not None and S.trip_ok(v0, trip) and mins <= S.BUDGET[bucket] + S.EPS:
                late = S.simulate([trip])[1]
                mask = sum(1 << i for i in idx)
                info[mask] = (1e6 * late + 1e3 + mins, late, mins, is_van)
                any_ok = True
        if not any_ok:
            break
    full = (1 << n) - 1
    INF = float("inf")
    dp = [INF] * (full + 1)
    choice = [0] * (full + 1)
    dp[0] = 0.0
    by_low = defaultdict(list)                  # feasible masks grouped by their lowest set bit
    for m in info:
        by_low[(m & -m).bit_length() - 1].append(m)
    for mask in range(1, full + 1):
        low = (mask & -mask).bit_length() - 1
        best, arg = INF, 0
        for sub in by_low[low]:
            if sub & mask == sub:
                c = info[sub][0] + dp[mask ^ sub]
                if c < best:
                    best, arg = c, sub
        dp[mask], choice[mask] = best, arg
    if dp[full] == INF:
        raise RuntimeError("some ambient orders cannot be served on any vehicle")
    trips, mask = [], full
    while mask:
        sub = choice[mask]
        trips.append(([group[i] for i in range(n) if sub >> i & 1], info[sub]))
        mask ^= sub
    return trips


def ambient_stage(ambient_vehicles, orders, policy, orders_all, weights=None):
    """Every serviceable non-chilled order is served; the plan minimises late stops, then vehicles, then minutes (exact DP
    per brand+district group, trucks matched to concrete vehicles afterwards and verified)."""
    t0 = time.time()
    trucks = [v for v in ambient_vehicles if v.type == "truck"]
    vans = sorted([v for v in ambient_vehicles if v.type == "van"], key=lambda v: (v.m3_cap, v.vid))
    truck_v = S.Vehicle("TRUCK*", "truck", "ambient", max(v.kg_cap for v in trucks), max(v.m3_cap for v in trucks), "Peliyagoda") if trucks else None
    van_v = S.Vehicle("VAN*", "van", "ambient", max(v.kg_cap for v in vans), max(v.m3_cap for v in vans), "Peliyagoda") if vans else None
    groups = defaultdict(list)
    for o in orders:
        groups[(o.brand, o.district)].append(o)
    trips = {"predawn": [], "daytime": []}      # (trip, is_van)
    late = vehicles = mins = 0
    for (brand, dist), g in sorted(groups.items()):
        bucket = S.category(brand)
        for trip, (cost, l, m, is_van) in _best_partition(sorted(g, key=lambda o: o.ref), truck_v, van_v, bucket):
            trips[bucket].append((trip, is_van))
            late += l; vehicles += 1; mins += m
    plan = {}
    used = {"predawn": set(), "daytime": set()}
    need = lambda t: (sum(o.m3 for o in t), sum(o.kg for o in t))
    for b in ("predawn", "daytime"):
        for trip, is_van in sorted(trips[b], key=lambda x: (not x[1], need(x[0])), reverse=True):
            m3, kg = need(trip)
            pool = [v for v in (vans if is_van else trucks) if v.vid not in used[b] and m3 <= v.m3_cap + S.EPS and kg <= v.kg_cap + S.EPS]
            pool.sort(key=lambda v: (b == "daytime" and v.vid not in used["predawn"], v.m3_cap, v.kg_cap, v.vid))
            if not pool:
                raise RuntimeError("vehicle matching failed (not enough trucks/vans for the planned trips)")
            used[b].add(pool[0].vid)
            plan.setdefault(pool[0].vid, []).append(trip)
    n_veh = len(plan)
    out = [("ambient orders served (all serviceable ones)", float(len(orders))), ("stops arriving after window close", float(late)),
           ("ambient trips planned", float(vehicles)), ("vehicles used", float(n_veh)), ("trip minutes", float(mins))]
    return plan, [("ambient", a, b) for a, b in out], time.time() - t0, vehicles


# ------------------------------------------------------------------------------------------------------------ driver


def solve_exact(policy="lexicographic", extra_vehicles=(), scenario=SCENARIO, weights=None, late_aware=True):
    """Proven-optimal allocation for `policy`. extra_vehicles: ids of workshop vehicles to treat as available (what-if).
    late_aware=False uses the fast reefer model (same served set, does not minimise late stops) for sensitivity sweeps."""
    t0 = time.time()
    vehicles = list(S.AVAILABLE[scenario]) + [S.VEHICLES_ALL[v] for v in extra_vehicles]
    scope = [o for o in S.ORDERS.values() if o.scenario == scenario]
    reefers = [v for v in vehicles if v.temp == "reefer"]
    ambient = [v for v in vehicles if v.temp == "ambient"]
    # orders no available vehicle can ever carry (alone, on an empty vehicle) are unavoidable deferrals
    def carriable(o):
        return any(S.compatible(v, o) and o.m3 <= v.m3_cap + S.EPS and o.kg <= v.kg_cap + S.EPS
                   and S.trip_minutes([o]) <= S.BUDGET[S.category(o.brand)] + S.EPS for v in vehicles)
    chilled = [o for o in scope if o.chilled and carriable(o)]
    other = [o for o in scope if not o.chilled and carriable(o)]
    plan_r, lv_r, s_r, n_r = reefer_stage(reefers, chilled, policy, scope, weights, late_aware)
    plan_a, lv_a, s_a, n_a = ambient_stage(ambient, other, policy, scope, weights)
    res = Result(policy, levels=lv_r + lv_a, seconds=time.time() - t0, columns=n_r + n_a)
    for plan in (plan_r, plan_a):
        for vid, ts in plan.items():
            ordered, _, _ = S.best_trip_order([t for t in ts if t])   # Fresh before daytime; fewest late stops
            res.trips[vid] = ordered
    return res


def to_submission(res, scenario=SCENARIO):
    """Booklet-format submission DataFrame (input row order) for a Result."""
    assign = {o.ref: (vid, k) for vid, ts in res.trips.items() for k, t in enumerate(ts, start=1) for o in t}
    sub = S.SCN[["scenario", "order_ref", "outlet_id"]].copy()
    sub["decision"] = ["served" if r in assign else "deferred" for r in sub.order_ref]
    sub["vehicle_id"] = [assign[r][0] if r in assign else "" for r in sub.order_ref]
    sub["trip_id"] = [assign[r][1] if r in assign else "" for r in sub.order_ref]
    return sub


def kpis(res):
    """Headline numbers of a Result."""
    served = [S.ORDERS[r] for r in res.served_refs()]
    scope = [o for o in S.ORDERS.values()]
    late = 0
    for vid, ts in res.trips.items():
        late += S.simulate([t for t in ts if t])[1]
    ch = [o for o in served if o.chilled]
    return dict(served=len(served), deferred=len(scope) - len(served), m3=round(sum(o.m3 for o in served), 2),
                chilled_served=len(ch), chilled_m3=round(sum(o.m3 for o in ch), 2),
                deferred_yday_served=sum(o.deferred_yday for o in served), deferred_yday_total=sum(o.deferred_yday for o in scope),
                priority_value=round(sum(S.order_value(o) for o in served), 1),
                vehicles=len([v for v, ts in res.trips.items() if ts]), trips=sum(len([t for t in ts if t]) for ts in res.trips.values()),
                late_stops_sim=late, seconds=round(res.seconds, 1))
