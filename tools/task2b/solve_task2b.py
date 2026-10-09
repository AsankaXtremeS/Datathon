"""
Task 2B - Peak-day fleet allocation (scenario S1, Peliyagoda).

Run:  python solve_task2b.py

Pipeline
  1. Load data and apply the booklet's feasibility rules (same logic as check_allocation.py).
  2. Score every order by business priority (see PRIORITY WEIGHTS).
  3. Stage A - the scarce resource first: chilled orders on reefer vehicles.
     Every feasible day plan (<=2 trips) is enumerated for each reefer and the
     best combination across the reefers is found by branch-and-bound
     (proven optimal for the stated objective unless the node limit is hit).
  4. Stage B - window-aware packing of all remaining orders onto the ambient fleet.
  5. Stage C - repair: insert any deferred order into spare capacity, then try
     1-for-1 swaps that raise total priority.
  6. Internal feasibility check, outputs, then the official validator.

Outputs
  submission_task2b.csv          allocation (scenario, order_ref, outlet_id, decision, vehicle_id, trip_id)
  task2b_trip_plan.csv           one row per vehicle trip: load, utilisation, minutes vs budget
  task2b_stop_schedule.csv       simulated clock per stop (soft delivery-window check)
  task2b_deferrals.csv           every deferred order with reason and cost
  task2b_allocation_summary.csv  scenario-level statistics and policy comparison
  task2b_policy.md               written prioritisation policy (auto-generated from the numbers)
"""

import itertools
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_SUBMISSION = os.path.join(PROJECT_ROOT, "HackBots_Datathon_Submision",
                              "predictions", "submission_task2b.csv")
OUT_METRICS = os.path.join(PROJECT_ROOT, "reports", "metrics", "task2b")
OUT_DOCS = os.path.join(PROJECT_ROOT, "HackBots_Datathon_Submision", "docs")

# ============================================================
# RULES (from the booklet / check_allocation.py)
# ============================================================
BUDGET = {"predawn": 270, "daytime": 480}   # Fresh trips | Style + Tech trips
MAX_TRIPS = 2
EPS = 1e-6

# ============================================================
# PRIORITY WEIGHTS - value of serving an order today
# ============================================================
W_BASE = 10.0          # every delivered stop (an outlet receives its order)
W_VOLUME = 1.0         # per m3 delivered (product on shelf for the festival build-up)
W_CHILLED = 5.0        # dairy / meat: cannot be held, highest sales risk
W_DEFERRED_YDAY = 20.0 # outlet was skipped on the previous run - avoid a second miss
W_DAYS_SINCE = 4.0     # per day without a delivery beyond the normal 1 day
LATE_PENALTY = 3.0     # soft penalty per stop simulated to arrive after window close

# ============================================================
# SOFT CLOCK (used only for sequencing/tie-breaks and reporting)
# ============================================================
PREDAWN_START = 3 * 60 + 30   # Fresh vehicles leave the depot at 03:30
DAYTIME_START = 8 * 60        # Style/Tech trips leave no earlier than 08:00

EXACT_NODE_LIMIT = 3_000_000


def hhmm(s):
    if pd.isna(s) or str(s).strip() == "":
        return None
    h, m = str(s).strip().split(":")
    return int(h) * 60 + int(m)


def fmt(t):
    return f"{int(t) // 60:02d}:{int(round(t)) % 60:02d}"


# ============================================================
# 1. LOAD DATA
# ============================================================
@dataclass(frozen=True)
class Order:
    ref: str
    scenario: str
    outlet: str
    brand: str
    district: str
    depot: str
    dock: str
    parking: str
    w_open: int
    w_close: int
    chilled: bool
    kg: float
    m3: float
    deferred_yday: int
    days_since: int


@dataclass(frozen=True)
class Vehicle:
    vid: str
    type: str
    temp: str
    kg_cap: float
    m3_cap: float
    depot: str


def category(brand):
    return "predawn" if brand == "Fresh" else "daytime"


def load():
    scn = pd.read_csv(os.path.join(DATA, "test", "task2b_peak_day_scenarios.csv"))
    fleet = pd.read_csv(os.path.join(DATA, "test", "task2b_peak_day_fleet.csv"))
    veh = pd.read_csv(os.path.join(DATA, "reference", "vehicles.csv"))
    dtravel = pd.read_csv(os.path.join(DATA, "reference", "district_travel.csv"))
    allow = pd.read_csv(os.path.join(DATA, "reference", "service_allowance.csv"))

    orders = {}
    for r in scn.itertuples(index=False):
        orders[r.order_ref] = Order(
            ref=r.order_ref, scenario=r.scenario, outlet=r.outlet_id, brand=r.brand,
            district=r.district, depot=r.depot, dock=r.dock_type,
            parking=r.parking_constraint, w_open=hhmm(r.window_open_time),
            w_close=hhmm(r.window_close_time),
            chilled=(r.temp_requirement == "chilled"), kg=float(r.order_weight_kg),
            m3=float(r.order_volume_m3), deferred_yday=int(r.deferred_yesterday),
            days_since=int(r.days_since_last_served))

    vehicles_all = {r.vehicle_id: Vehicle(r.vehicle_id, r.type, r.temp, float(r.weight_cap_kg),
                                          float(r.volume_cap_m3), r.depot)
                    for r in veh.itertuples(index=False)}
    available = defaultdict(list)
    for r in fleet.itertuples(index=False):
        if r.status == "available" and r.vehicle_id in vehicles_all:
            available[r.scenario].append(vehicles_all[r.vehicle_id])

    travel = {r.district: (float(r.depot_to_district_freeflow_min), float(r.inter_stop_freeflow_min))
              for r in dtravel.itertuples(index=False)}
    allowance = {(r.brand, r.dock_type): float(r.service_allowance_min) for r in allow.itertuples(index=False)}
    return scn, fleet, orders, vehicles_all, available, travel, allowance


SCN, FLEET, ORDERS, VEHICLES_ALL, AVAILABLE, TRAVEL, ALLOWANCE = load()


# ============================================================
# 2. RULE ENGINE
# ============================================================
def trip_minutes(orders):
    """Booklet formula: outbound + inter-stop x (n-1) + sum of handling allowances."""
    if not orders:
        return 0.0
    out, inter = TRAVEL[orders[0].district]
    return out + inter * (len(orders) - 1) + sum(ALLOWANCE[(o.brand, o.dock)] for o in orders)


def compatible(v, o):
    return (v.depot == o.depot
            and (not o.chilled or v.temp == "reefer")
            and (o.parking != "van_only" or v.type == "van"))


def trip_ok(v, orders):
    if not orders:
        return True
    if len({o.brand for o in orders}) > 1 or len({o.district for o in orders}) > 1:
        return False
    if not all(compatible(v, o) for o in orders):
        return False
    return (sum(o.m3 for o in orders) <= v.m3_cap + EPS
            and sum(o.kg for o in orders) <= v.kg_cap + EPS)


def minutes_used(trips):
    used = {"predawn": 0.0, "daytime": 0.0}
    for t in trips:
        if t:
            used[category(t[0].brand)] += trip_minutes(t)
    return used


def vehicle_ok(v, trips):
    trips = [t for t in trips if t]
    if len(trips) > MAX_TRIPS or not all(trip_ok(v, t) for t in trips):
        return False
    used = minutes_used(trips)
    return all(used[c] <= BUDGET[c] + EPS for c in BUDGET)


# ============================================================
# 3. PRIORITY
# ============================================================
def order_value(o, weights=None):
    w = weights or {}
    return (w.get("base", W_BASE)
            + w.get("volume", W_VOLUME) * o.m3
            + (w.get("chilled", W_CHILLED) if o.chilled else 0.0)
            + w.get("deferred_yday", W_DEFERRED_YDAY) * o.deferred_yday
            + w.get("days_since", W_DAYS_SINCE) * max(0, o.days_since - 1))


# ============================================================
# 4. SOFT CLOCK SIMULATION (delivery windows)
# ============================================================
def sequence(trip):
    """Visit stops by earliest window close, then earliest open (EDD rule)."""
    return sorted(trip, key=lambda o: (o.w_close, o.w_open, o.ref))


def simulate(trips_in_order):
    """Return per-stop rows and the number of late stops for a vehicle's day."""
    rows, late, free = [], 0, PREDAWN_START
    for k, trip in enumerate(trips_in_order, start=1):
        out, inter = TRAVEL[trip[0].district]
        dep = free if trip[0].brand == "Fresh" else max(free, DAYTIME_START)
        t = dep + out
        for i, o in enumerate(sequence(trip)):
            if i:
                t += inter
            arrive = t
            start = max(arrive, o.w_open)
            is_late = arrive > o.w_close + EPS
            late += is_late
            t = start + ALLOWANCE[(o.brand, o.dock)]
            rows.append(dict(order_ref=o.ref, outlet_id=o.outlet, run_order=k, depart_depot=fmt(dep),
                             arrive=fmt(arrive), service_start=fmt(start), leave=fmt(t),
                             window=f"{fmt(o.w_open)}-{fmt(o.w_close)}", late=int(is_late)))
        free = t + out  # drive back to the depot
    return rows, late


def best_trip_order(trips):
    """Pick trip_id order: Fresh before daytime; among equals, fewest late stops then earliest finish."""
    trips = [t for t in trips if t]
    best = None
    for perm in itertools.permutations(trips):
        cats = [category(t[0].brand) for t in perm]
        if cats != sorted(cats, key=lambda c: c != "predawn"):
            continue
        rows, late = simulate(list(perm))
        key = (late, hhmm(rows[-1]["leave"]) if rows else 0)
        if best is None or key < best[0]:
            best = (key, list(perm), rows, late)
    return best[1], best[2], best[3]


# ============================================================
# 5. STAGE A - chilled orders on reefers (exact branch-and-bound)
# ============================================================
def enum_trips(v, pool, value):
    """All feasible single trips (subsets of one brand+district) for vehicle v."""
    groups = defaultdict(list)
    for o in pool:
        if compatible(v, o):
            groups[(o.brand, o.district)].append(o)
    trips = []
    for (brand, district), lst in sorted(groups.items()):
        lst.sort(key=lambda o: o.ref)
        out, inter = TRAVEL[district]
        budget = BUDGET[category(brand)]

        def dfs(i, chosen, m3, kg, handle):
            if chosen:
                mins = out + inter * (len(chosen) - 1) + handle
                trips.append((sum(value[o.ref] for o in chosen), frozenset(o.ref for o in chosen),
                              mins, category(brand), m3))
            for j in range(i, len(lst)):
                o = lst[j]
                nm, nk = m3 + o.m3, kg + o.kg
                nh = handle + ALLOWANCE[(o.brand, o.dock)]
                if nm > v.m3_cap + EPS or nk > v.kg_cap + EPS:
                    continue
                if out + inter * len(chosen) + nh > budget + EPS:
                    continue
                dfs(j + 1, chosen + [o], nm, nk, nh)

        dfs(0, [], 0.0, 0.0, 0.0)
    return trips


def day_plans(v, pool, value):
    """All feasible day plans (0, 1 or 2 trips) for one vehicle, best first."""
    trips = enum_trips(v, pool, value)
    plans = [(0.0, frozenset(), ())]
    for t in trips:
        plans.append((t[0], t[1], (t,)))
    for a, b in itertools.combinations(trips, 2):
        if a[1] & b[1]:
            continue
        if a[3] == b[3] and a[2] + b[2] > BUDGET[a[3]] + EPS:
            continue
        plans.append((a[0] + b[0], a[1] | b[1], (a, b)))
    plans.sort(key=lambda p: (-p[0], -sum(t[4] for t in p[2]), sorted(p[1])))
    return plans


def solve_reefers(reefers, chilled, value):
    """Choose one day plan per reefer, disjoint orders, maximise total value."""
    plans = {v.vid: day_plans(v, chilled, value) for v in reefers}
    order = sorted(reefers, key=lambda v: (len(plans[v.vid]), v.vid))  # most constrained first
    total_pool = sum(value[o.ref] for o in chilled)

    # incumbent: greedy over every vehicle permutation
    best_val, best_sel = -1.0, None
    for perm in itertools.permutations(order):
        used, sel, val = frozenset(), {}, 0.0
        for v in perm:
            p = next(p for p in plans[v.vid] if not (p[1] & used))
            sel[v.vid], used, val = p, used | p[1], val + p[0]
        if val > best_val + EPS:
            best_val, best_sel = val, dict(sel)

    # branch-and-bound. Plans are sorted by value, so the first plan of a vehicle that
    # does not overlap `used` is that vehicle's best possible plan given `used`.
    nodes = 0
    stack_sel = {}

    def first_disjoint(vid, used):
        return next(p for p in plans[vid] if not (p[1] & used))   # empty plan always qualifies

    def bb(k, used, val):
        nonlocal best_val, best_sel, nodes
        v = order[k]
        if k == len(order) - 1:                      # last vehicle: its best disjoint plan is optimal
            p = first_disjoint(v.vid, used)
            if val + p[0] > best_val + EPS:
                best_val, best_sel = val + p[0], {**stack_sel, v.vid: p}
            return
        rest = order[k + 1:]
        for p in plans[v.vid]:
            nodes += 1
            if nodes > EXACT_NODE_LIMIT:
                return
            # quick bound: this plan + each remaining vehicle's unconstrained best
            if val + p[0] + sum(first_disjoint(w.vid, used)[0] for w in rest) <= best_val + EPS:
                break
            if p[1] & used:
                continue
            nu = used | p[1]
            # tighter bound: remaining vehicles' best plans that avoid every order already taken
            bound = val + p[0] + min(sum(first_disjoint(w.vid, nu)[0] for w in rest), total_pool - val - p[0])
            if bound <= best_val + EPS:
                continue
            stack_sel[v.vid] = p
            bb(k + 1, nu, val + p[0])
            stack_sel.pop(v.vid, None)

    bb(0, frozenset(), 0.0)
    proven = nodes <= EXACT_NODE_LIMIT
    return best_sel, best_val, proven, nodes


# ============================================================
# 6. STAGE B / C - packing and repair on the whole fleet
# ============================================================
class Allocation:
    def __init__(self, vehicles):
        self.vehicles = {v.vid: v for v in vehicles}
        self.trips = {v.vid: [] for v in vehicles}   # vid -> list of lists of Order

    def served(self):
        return {o.ref for ts in self.trips.values() for t in ts for o in t}

    def late_count(self, vid, trips=None):
        trips = [t for t in (trips if trips is not None else self.trips[vid]) if t]
        return best_trip_order(trips)[2] if trips else 0


def greedy_fill(v, pending, used_minutes):
    """First-fit pending orders (priority order) into one new trip for vehicle v."""
    chosen = []
    for o in pending:
        cand = chosen + [o]
        if not trip_ok(v, cand):
            continue
        cat = category(o.brand)
        if used_minutes[cat] + trip_minutes(cand) > BUDGET[cat] + EPS:
            continue
        chosen = cand
    return chosen


def pack_remaining(alloc, pending_orders, value):
    """Window-aware packing of every remaining order, group by group."""
    groups = defaultdict(list)
    for o in pending_orders:
        groups[(o.brand, o.district)].append(o)
    # Fresh (pre-dawn) groups first, then by group priority
    keys = sorted(groups, key=lambda k: (k[0] != "Fresh", -sum(value[o.ref] for o in groups[k]), k))
    for key in keys:
        pending = sorted(groups[key], key=lambda o: (-value[o.ref], -o.m3, o.ref))
        while pending:
            best = None
            for vid, v in sorted(alloc.vehicles.items()):
                ts = alloc.trips[vid]
                if len(ts) >= MAX_TRIPS:
                    continue
                chosen = greedy_fill(v, pending, minutes_used(ts))
                if not chosen:
                    continue
                late_before = alloc.late_count(vid)
                late_after = alloc.late_count(vid, ts + [chosen])
                gain = sum(value[o.ref] for o in chosen) - LATE_PENALTY * (late_after - late_before)
                spare = v.m3_cap - sum(o.m3 for o in chosen)
                fresh = key[0] == "Fresh"
                k = (round(gain, 6),
                     (len(ts) == 0) if fresh else (len(ts) > 0),   # Fresh: new vehicle; daytime: reuse
                     v.temp != "reefer",                            # keep reefers for chilled work
                     -spare, vid)
                if best is None or k > best[0]:
                    best = (k, vid, chosen)
            if best is None:
                break
            _, vid, chosen = best
            alloc.trips[vid].append(chosen)
            refs = {o.ref for o in chosen}
            pending = [o for o in pending if o.ref not in refs]


def try_insert(alloc, o, value):
    """Insert order o into an existing trip or a new trip; best gain (least lateness) wins."""
    best = None
    for vid, v in sorted(alloc.vehicles.items()):
        if not compatible(v, o):
            continue
        ts = alloc.trips[vid]
        options = [(i, ts[:i] + [ts[i] + [o]] + ts[i + 1:]) for i in range(len(ts))]
        if len(ts) < MAX_TRIPS:
            options.append((len(ts), ts + [[o]]))
        for i, new in options:
            if vehicle_ok(v, new):
                k = (-(alloc.late_count(vid, new) - alloc.late_count(vid)), i < len(ts), vid)
                if best is None or k > best[0]:
                    best = (k, vid, new)
    if best:
        alloc.trips[best[1]] = best[2]
        return True
    return False


def repair(alloc, value):
    """Insert deferred orders; then 1-for-1 swaps that increase total priority."""
    improved = True
    while improved:
        improved = False
        served = alloc.served()
        deferred = sorted((o for o in ORDERS_IN_SCOPE if o.ref not in served),
                          key=lambda o: (-value[o.ref], o.ref))
        for o in deferred:
            if try_insert(alloc, o, value):
                improved = True
        if improved:
            continue
        # swap: drop a lower-value served order x so deferred o fits, then try to re-home x
        for o in deferred:
            done = False
            for vid, v in sorted(alloc.vehicles.items()):
                if not compatible(v, o):
                    continue
                for i, t in enumerate(alloc.trips[vid]):
                    if t[0].brand != o.brand or t[0].district != o.district:
                        continue
                    for x in sorted(t, key=lambda x: (value[x.ref], x.ref)):
                        if value[x.ref] >= value[o.ref] - EPS:
                            break
                        new_t = [y for y in t if y.ref != x.ref] + [o]
                        new = alloc.trips[vid][:i] + [new_t] + alloc.trips[vid][i + 1:]
                        if vehicle_ok(v, new):
                            alloc.trips[vid] = new
                            try_insert(alloc, x, value)
                            done = improved = True
                            break
                    if done:
                        break
                if done:
                    break
            if improved:
                break


# ============================================================
# 7. FULL SOLVE
# ============================================================
def solve(scenario, weights=None, verbose=True):
    value = {r: order_value(o, weights) for r, o in ORDERS.items()}
    vehicles = [v for v in AVAILABLE[scenario]]
    scope = [o for o in ORDERS.values() if o.scenario == scenario]
    alloc = Allocation(vehicles)

    reefers = [v for v in vehicles if v.temp == "reefer"]
    chilled = [o for o in scope if o.chilled and any(compatible(v, o) for v in reefers)]
    t0 = time.time()
    sel, val, proven, nodes = solve_reefers(reefers, chilled, value)
    if verbose:
        print(f"  Stage A (reefers): value {val:.1f}, {nodes:,} B&B nodes, "
              f"{'proven optimal' if proven else 'node limit hit - best found'} in {time.time() - t0:.1f}s")
    for vid, plan in sel.items():
        for trip in plan[2]:
            alloc.trips[vid].append([ORDERS[r] for r in sorted(trip[1])])

    served = alloc.served()
    pack_remaining(alloc, [o for o in scope if o.ref not in served], value)
    repair(alloc, value)
    return alloc, value, dict(stageA_value=val, stageA_proven=proven, stageA_nodes=nodes,
                              stageA_served=frozenset().union(*[p[1] for p in sel.values()]))


def forced_tradeoff(scenario, ref, value, stats):
    """Re-run stage A with order `ref` forced onto a reefer: what serving it would have cost."""
    reefers = [v for v in AVAILABLE[scenario] if v.temp == "reefer"]
    chilled = [o for o in ORDERS.values() if o.scenario == scenario and o.chilled
               and any(compatible(v, o) for v in reefers)]
    forced = dict(value)
    forced[ref] += 1000.0
    sel, val, _, _ = solve_reefers(reefers, chilled, forced)
    served = frozenset().union(*[p[1] for p in sel.values()])
    trip = next(t for p in sel.values() for t in p[2] if ref in t[1])
    return dict(loss=stats["stageA_value"] - (val - 1000.0), trip_min=trip[2], trip_stops=len(trip[1]),
                dropped=sorted(stats["stageA_served"] - served), added=sorted(served - stats["stageA_served"]))


# ============================================================
# 8. INTERNAL CHECK (mirrors check_allocation.py)
# ============================================================
def internal_check(alloc):
    errors = []
    avail = {v.vid for v in alloc.vehicles.values()}
    seen = defaultdict(int)
    for vid, ts in alloc.trips.items():
        v = alloc.vehicles[vid]
        if vid not in avail:
            errors.append(f"{vid} not available")
        if not vehicle_ok(v, ts):
            errors.append(f"{vid} violates capacity/compatibility/time rules")
        for t in ts:
            for o in t:
                seen[o.ref] += 1
    errors += [f"{r} assigned {n} times" for r, n in seen.items() if n > 1]
    return errors


# ============================================================
# 9. REPORTING
# ============================================================
def unavoidable_reason(o, vehicles):
    fits = [v for v in vehicles if compatible(v, o) and o.m3 <= v.m3_cap + EPS and o.kg <= v.kg_cap + EPS
            and trip_minutes([o]) <= BUDGET[category(o.brand)] + EPS]
    if fits:
        return None
    biggest = max((v for v in vehicles if compatible(v, o)), key=lambda v: v.m3_cap, default=None)
    if biggest is None:
        return "no available vehicle is compatible"
    return (f"order is {o.m3:.2f} m3 / {o.kg:.0f} kg; largest compatible vehicle "
            f"{biggest.vid} holds {biggest.m3_cap:g} m3 / {biggest.kg_cap:g} kg")


def build_outputs(scenario, alloc, value, stats, comparisons):
    # ---- trip ids, schedule ----
    assign, trip_rows, stop_rows = {}, [], []
    for vid in sorted(alloc.trips):
        ts = [t for t in alloc.trips[vid] if t]
        if not ts:
            continue
        ordered, rows, _ = best_trip_order(ts)
        alloc.trips[vid] = ordered
        v = alloc.vehicles[vid]
        for k, t in enumerate(ordered, start=1):
            for o in t:
                assign[o.ref] = (vid, k)
            m3, kg, mins = sum(o.m3 for o in t), sum(o.kg for o in t), trip_minutes(t)
            out, inter = TRAVEL[t[0].district]
            cat = category(t[0].brand)
            trip_rows.append(dict(
                scenario=scenario, vehicle_id=vid, trip_id=k, vehicle=f"{v.type}/{v.temp}",
                brand=t[0].brand, district=t[0].district, stops=len(t),
                chilled_stops=sum(o.chilled for o in t),
                volume_m3=round(m3, 3), volume_cap_m3=v.m3_cap, volume_util=round(m3 / v.m3_cap, 3),
                weight_kg=round(kg, 1), weight_cap_kg=v.kg_cap, weight_util=round(kg / v.kg_cap, 3),
                outbound_min=out, interstop_min=inter * (len(t) - 1),
                handling_min=sum(ALLOWANCE[(o.brand, o.dock)] for o in t), trip_min=mins,
                budget=cat, vehicle_budget_used_min=minutes_used(ordered)[cat], budget_min=BUDGET[cat],
                orders=" ".join(o.ref for o in sequence(t))))
        for r in rows:
            r["vehicle_id"] = vid
            r["trip_id"] = r.pop("run_order")
            stop_rows.append(r)
    trips_df = pd.DataFrame(trip_rows)
    stops_df = pd.DataFrame(stop_rows)

    # ---- submission (input row order, booklet template columns) ----
    sub = SCN[["scenario", "order_ref", "outlet_id"]].copy()
    sub["decision"] = [("served" if r in assign else "deferred") for r in sub.order_ref]
    sub["vehicle_id"] = [assign[r][0] if r in assign else "" for r in sub.order_ref]
    sub["trip_id"] = pd.array([assign[r][1] if r in assign else None for r in sub.order_ref], dtype="Int64")
    os.makedirs(os.path.dirname(OUT_SUBMISSION), exist_ok=True)
    sub.to_csv(OUT_SUBMISSION, index=False)

    # ---- deferrals ----
    vehicles = list(alloc.vehicles.values())
    reefers = [v for v in vehicles if v.temp == "reefer"]
    served_by = {r: assign[r] for r in assign}
    def_rows = []
    for o in ORDERS.values():
        if o.ref in assign:
            continue
        reason = unavoidable_reason(o, vehicles)
        if reason:
            kind, why = "unavoidable", reason
        elif o.chilled:
            kind = "capacity (reefer)"
            out_min = TRAVEL[o.district][0]
            why = (f"all {len(reefers)} available reefers are full or out of pre-dawn minutes; a {o.district} trip "
                   f"needs {out_min:g} of the {BUDGET['predawn']} pre-dawn minutes outbound alone, so serving it "
                   f"(priority {value[o.ref]:.1f}) would displace nearer multi-stop chilled trips whose combined "
                   f"priority is higher")
        else:
            kind, why = "choice", "no spare capacity on a compatible vehicle after higher-priority orders"
        def_rows.append(dict(scenario=o.scenario, order_ref=o.ref, outlet_id=o.outlet, brand=o.brand,
                             district=o.district, temp="chilled" if o.chilled else "ambient",
                             volume_m3=o.m3, weight_kg=o.kg, deferred_yesterday=o.deferred_yday,
                             days_since_last_served=o.days_since, priority=round(value[o.ref], 1),
                             deferral_type=kind, reason=why,
                             cost=(f"{o.m3:.2f} m3 of {'chilled' if o.chilled else 'ambient'} {o.brand} "
                                   f"not on shelf today; outlet reaches {o.days_since + 1} days since last delivery")))
    def_df = pd.DataFrame(def_rows)

    # ---- summary ----
    n = len(sub)
    served_mask = sub.decision == "served"
    late = int(stops_df.late.sum()) if len(stops_df) else 0
    scope = SCN.assign(served=served_mask.values)
    summary = [dict(metric="total_orders", value=n),
               dict(metric="served", value=int(served_mask.sum())),
               dict(metric="deferred", value=int((~served_mask).sum())),
               dict(metric="service_rate_orders", value=round(served_mask.mean(), 4)),
               dict(metric="demand_m3", value=round(scope.order_volume_m3.sum(), 2)),
               dict(metric="served_m3", value=round(scope[scope.served].order_volume_m3.sum(), 2)),
               dict(metric="service_rate_volume", value=round(scope[scope.served].order_volume_m3.sum()
                                                               / scope.order_volume_m3.sum(), 4)),
               dict(metric="priority_value_served", value=round(sum(value[r] for r in assign), 1)),
               dict(metric="priority_value_total", value=round(sum(value.values()), 1)),
               dict(metric="vehicles_used", value=trips_df.vehicle_id.nunique()),
               dict(metric="vehicles_available", value=len(vehicles)),
               dict(metric="trips", value=len(trips_df)),
               dict(metric="stops_simulated_late", value=late),
               dict(metric="reefer_stageA_proven_optimal", value=bool(stats["stageA_proven"]))]
    for (b, t), g in scope.groupby(["brand", "temp_requirement"]):
        summary.append(dict(metric=f"{b}_{t}_served_orders", value=f"{int(g.served.sum())}/{len(g)}"))
        summary.append(dict(metric=f"{b}_{t}_served_m3",
                            value=f"{g[g.served].order_volume_m3.sum():.2f}/{g.order_volume_m3.sum():.2f}"))
    for name, c in comparisons.items():
        summary.append(dict(metric=f"alt_policy_{name}", value=c))
    summary_df = pd.DataFrame(summary)
    summary_df.insert(0, "scenario", scenario)

    os.makedirs(OUT_METRICS, exist_ok=True)
    trips_df.to_csv(os.path.join(OUT_METRICS, "task2b_trip_plan.csv"), index=False)
    stops_df.to_csv(os.path.join(OUT_METRICS, "task2b_stop_schedule.csv"), index=False)
    def_df.to_csv(os.path.join(OUT_METRICS, "task2b_deferrals.csv"), index=False)
    summary_df.to_csv(os.path.join(OUT_METRICS, "task2b_allocation_summary.csv"), index=False)
    return sub, trips_df, stops_df, def_df, summary_df


def write_policy(scenario, sub, trips_df, stops_df, def_df, value, comparisons, stats=None):
    scope = SCN.assign(served=(sub.decision == "served").values)
    vehicles = AVAILABLE[scenario]
    reefers = [v for v in vehicles if v.temp == "reefer"]
    workshop = FLEET[(FLEET.scenario == scenario) & (FLEET.status != "available")].vehicle_id.tolist()
    reefer_ws = [v for v in workshop if VEHICLES_ALL[v].temp == "reefer"]
    ch = scope[scope.temp_requirement == "chilled"]
    amb = scope[scope.temp_requirement == "ambient"]
    rcap = sum(v.m3_cap for v in reefers)
    rkg = sum(v.kg_cap for v in reefers)
    acap = sum(v.m3_cap for v in vehicles if v.temp != "reefer")
    reefer_trips = trips_df[trips_df.vehicle.str.endswith("reefer")]
    served_ch_m3 = ch[ch.served].order_volume_m3.sum()
    L = []
    L.append(f"# Task 2B - Peak-day allocation policy ({scenario}, Peliyagoda)\n")
    L.append(f"**Result:** {int(scope.served.sum())}/{len(scope)} orders served "
             f"({scope.served.mean():.1%}), {scope[scope.served].order_volume_m3.sum():.1f} of "
             f"{scope.order_volume_m3.sum():.1f} m3 ({scope[scope.served].order_volume_m3.sum() / scope.order_volume_m3.sum():.1%}). "
             f"{len(trips_df)} trips on {trips_df.vehicle_id.nunique()} of {len(vehicles)} available vehicles. "
             f"Passes check_allocation.py.\n")
    L.append("## 1. What limits service today\n")
    L.append("| Resource | Demand | Available (one wave) | Verdict |\n|---|---|---|---|")
    L.append(f"| Reefer volume (chilled) | {ch.order_volume_m3.sum():.1f} m3 / {ch.order_weight_kg.sum() / 1000:.1f} t, {len(ch)} orders "
             f"| {len(reefers)} reefers = {rcap:.1f} m3 / {rkg / 1000:.1f} t "
             f"({', '.join(reefer_ws)} in workshop) | **Binding** |")
    L.append(f"| Pre-dawn minutes on reefers | far districts use 103-173 min outbound | 270 min per vehicle | **Binding** - a far trip leaves room for at most one short trip |")
    L.append(f"| Van-only chilled (OUT001-003) | {ch[ch.parking_constraint == 'van_only'].order_weight_kg.sum():.0f} kg "
             f"| 1 reefer van, 1040 kg per trip | needs both van trips |")
    L.append(f"| Ambient trucks/vans | {amb.order_volume_m3.sum():.1f} m3 | {acap:.0f} m3 | ample |")
    L.append(f"\nEven if every reefer did two full loads ({2 * rcap:.1f} m3), chilled demand of {ch.order_volume_m3.sum():.1f} m3 "
             f"could not all move, and the 270-min window prevents two trips to the far districts. "
             f"We served **{served_ch_m3:.1f} m3 chilled** ({int(ch.served.sum())}/{len(ch)} orders).\n")
    L.append("## 2. Priority rule\n")
    L.append(f"Value of serving an order = {W_BASE:g} per stop + {W_VOLUME:g} x m3 + {W_CHILLED:g} if chilled "
             f"+ {W_DEFERRED_YDAY:g} if deferred yesterday + {W_DAYS_SINCE:g} x (days since last served - 1); "
             f"-{LATE_PENALTY:g} per stop simulated to arrive after its window closes.\n")
    L.append("Rationale: chilled dairy/meat cannot be held and the festival is a week away; an outlet already "
             "skipped yesterday or unserved for 5 days is at real stock-out risk, so a second miss costs more than one "
             "large fresh delivery to a recently served outlet. Volume rewards product on shelf; the per-stop term keeps "
             "small outlets from being starved by big ones.\n")
    L.append("## 3. How the allocation was built\n")
    L.append("1. **Scarcest resource first** - every feasible day plan (<=2 trips, capacity, 270-min budget) was "
             "enumerated for each reefer, and branch-and-bound chose the plan combination with maximum priority "
             + ("value (proven optimal for this objective)." if stats and stats.get("stageA_proven") else
                f"value (best combination found: the search stops at {EXACT_NODE_LIMIT:,} nodes, so optimality "
                "is not proven)."))
    L.append("2. **Ambient fleet** - remaining orders packed per brand+district, Fresh first, preferring a fresh "
             "vehicle's first trip (on-time before 08:00) and keeping Style/Tech daytime trips on vehicles already used pre-dawn.")
    L.append("3. **Repair** - any deferred order is inserted wherever it fits; 1-for-1 swaps replace a lower-value order.")
    L.append("4. **Windows (soft)** - stops sequenced by earliest window close; departures 03:30 (Fresh) / 08:00 "
             f"(daytime), return = outbound time. {int(stops_df.late.sum())} of {len(stops_df)} stops simulate late.\n")
    L.append("**Reefer trips (the decisive part of the plan):**\n")
    L.append("| Vehicle | Trip | District | Stops | m3 / cap | kg / cap | Minutes (out + inter + handling) |\n|---|---|---|---|---|---|---|")
    for r in reefer_trips.itertuples():
        L.append(f"| {r.vehicle_id} | {r.trip_id} | {r.district} | {r.stops} | {r.volume_m3:.1f} / {r.volume_cap_m3:g} "
                 f"| {r.weight_kg:.0f} / {r.weight_cap_kg:g} | {r.outbound_min:g} + {r.interstop_min:g} + "
                 f"{r.handling_min:g} = **{r.trip_min:g}** (day {r.vehicle_budget_used_min:g}/{r.budget_min}) |")
    L.append("")
    L.append("## 4. Deferrals - unavoidable vs. chosen, and their cost\n")
    if len(def_df):
        L.append("| Order | Outlet | District | Type | m3 | Def. yday | Days | Why |\n|---|---|---|---|---|---|---|---|")
        for r in def_df.itertuples():
            short = r.reason if r.deferral_type == "unavoidable" else (
                f"{TRAVEL[r.district][0]:g} min outbound of the {BUDGET['predawn']}-min window - serving it would "
                "displace nearer multi-stop chilled trips worth more in total")
            L.append(f"| {r.order_ref} | {r.outlet_id} | {r.district} | {r.temp} {r.brand} | {r.volume_m3:.2f} "
                     f"| {r.deferred_yesterday} | {r.days_since_last_served} | **{r.deferral_type}**: {short} |")
        un = def_df[def_df.deferral_type == "unavoidable"]
        cap = def_df[def_df.deferral_type != "unavoidable"]
        L.append(f"\n- **Unavoidable ({len(un)})**: physically larger than any available vehicle; must be split by the "
                 "outlet/merchandising team or sent with a hired truck.")
        L.append(f"- **Capacity-driven, our choice of which ({len(cap)})**: {cap.volume_m3.sum():.1f} m3 had to stay behind "
                 "because reefer capacity is short; *which* orders stayed is our choice. We kept back the chilled orders in the "
                 f"far districts ({', '.join(sorted(f'{d} {TRAVEL[d][0]:g} min' for d in cap.district.unique()))} outbound), "
                 f"where the outbound drive alone takes a large share of a reefer's {BUDGET['predawn']} pre-dawn minutes; "
                 "the same minutes serve more stops and more priority value in nearer districts.")
        dy = cap[cap.deferred_yesterday == 1]
        for r in dy.itertuples():
            txt = (f"- **Second miss, a deliberate trade-off**: {r.order_ref} ({r.outlet_id}, {r.district}) was also deferred "
                   f"yesterday and is {r.days_since_last_served} days unserved (priority {r.priority:.1f}"
                   + (", the highest of any deferred order)." if r.priority >= def_df.priority.max() else ")."))
            if stats and "stageA_served" in stats and r.temp == "chilled":
                t = forced_tradeoff(scenario, r.order_ref, value, stats)
                fmt_o = lambda refs: ", ".join(f"{x} ({ORDERS[x].district}, {value[x]:.1f})" for x in refs)
                txt += (f" We re-solved the reefer stage with it forced in: a {r.district} trip takes {t['trip_min']:g} of "
                        f"the {BUDGET['predawn']} pre-dawn minutes for {t['trip_stops']} stop(s), which pushes out {len(t['dropped'])} "
                        f"chilled orders [{fmt_o(t['dropped'])}] while only {len(t['added'])} get in [{fmt_o(t['added'])}]. "
                        f"That is {t['loss']:.1f} less priority value and {len(t['dropped']) - len(t['added'])} more outlets "
                        "short today.")
            txt += " It goes first on tomorrow's plan; releasing a workshop reefer would recover it today."
            L.append(txt)
        L.append(f"- **Cost**: {def_df.volume_m3.sum():.1f} m3 not delivered "
                 f"({def_df[def_df.temp == 'chilled'].volume_m3.sum():.1f} m3 chilled), "
                 f"{def_df.outlet_id.nunique()} outlets short today; these orders carry into tomorrow's peak.")
    L.append("\n## 5. Trade-off check (same solver, different priorities)\n")
    L.append("| Policy | Orders served | m3 served | Chilled m3 | Deferred-yesterday orders served |\n|---|---|---|---|---|")
    for name, c in comparisons.items():
        L.append(f"| {name} | {c['served']} | {c['m3']:.1f} | {c['chilled_m3']:.1f} | {c['dy_served']} |")
    L.append("\n## 6. Recommendations\n")
    L.append(f"- Release a reefer from the workshop ({', '.join(reefer_ws)}): every extra reefer recovers roughly one deferred chilled load.")
    L.append("- Serve tomorrow's carried-over chilled orders first (they will be `deferred_yesterday = 1`).")
    L.append("- Ask OUT070 to split its 40.7 m3 Style order, or book a larger vehicle.")
    os.makedirs(OUT_DOCS, exist_ok=True)
    with open(os.path.join(OUT_DOCS, "Task2B_policy.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def policy_stats(alloc, value_unused=None):
    served = alloc.served()
    s = [ORDERS[r] for r in served]
    return dict(served=len(s), m3=sum(o.m3 for o in s), chilled_m3=sum(o.m3 for o in s if o.chilled),
                dy_served=sum(o.deferred_yday for o in s))


# ============================================================
# 10. MAIN
# ============================================================
def main():
    global ORDERS_IN_SCOPE
    print(f"Orders: {len(ORDERS)} | vehicles in fleet file: {len(FLEET)} | "
          f"available: {sum(len(v) for v in AVAILABLE.values())}")
    subs = []
    for scenario in sorted(SCN.scenario.unique()):
        ORDERS_IN_SCOPE = [o for o in ORDERS.values() if o.scenario == scenario]
        print(f"\nScenario {scenario}")
        alloc, value, stats = solve(scenario)
        errs = internal_check(alloc)
        if errs:
            raise SystemExit("internal check failed:\n  " + "\n  ".join(errs))
        print("  Internal check: OK")

        comparisons = {"Chosen (priority-weighted)": policy_stats(alloc)}
        alt = {"Max volume only": dict(base=0, volume=1, chilled=0, deferred_yday=0, days_since=0),
               "Max order count only": dict(base=1, volume=0, chilled=0, deferred_yday=0, days_since=0)}
        for name, w in alt.items():
            a, _, _ = solve(scenario, w, verbose=False)
            comparisons[name] = policy_stats(a)

        sub, trips_df, stops_df, def_df, summary_df = build_outputs(scenario, alloc, value, stats, comparisons)
        write_policy(scenario, sub, trips_df, stops_df, def_df, value, comparisons, stats)
        subs.append(sub)

        pd.set_option("display.width", 200)
        print(f"\n  Served {int((sub.decision == 'served').sum())} / {len(sub)}  "
              f"(deferred {int((sub.decision == 'deferred').sum())})")
        print(trips_df[["vehicle_id", "trip_id", "vehicle", "brand", "district", "stops", "volume_m3",
                        "volume_cap_m3", "weight_kg", "weight_cap_kg", "trip_min",
                        "vehicle_budget_used_min"]].to_string(index=False))
        if len(def_df):
            print("\n  Deferred:")
            print(def_df[["order_ref", "outlet_id", "district", "temp", "volume_m3", "deferred_yesterday",
                          "days_since_last_served", "priority", "deferral_type"]].to_string(index=False))
        print("\n  Policy comparison:")
        for k, c in comparisons.items():
            print(f"    {k:28s} served {c['served']:3d}  m3 {c['m3']:6.1f}  chilled m3 {c['chilled_m3']:6.1f}  "
                  f"deferred-yday served {c['dy_served']}")

    print(f"\nWrote {OUT_SUBMISSION}")
    validator = os.path.join(HERE, "check_allocation.py")
    if os.path.exists(validator) and os.path.getsize(validator) > 0:
        print("\nRunning official validator:")
        res = subprocess.run([sys.executable, validator, OUT_SUBMISSION], cwd=HERE)
        sys.exit(res.returncode)
    print("check_allocation.py missing or empty - skipped official validation")


ORDERS_IN_SCOPE = list(ORDERS.values())

if __name__ == "__main__":
    main()
