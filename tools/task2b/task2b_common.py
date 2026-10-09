"""Shared helpers for the Task 2B notebooks (peak-day allocation, scenario S1, Peliyagoda).

Import from a notebook in notebooks/task2b/:
    import sys; sys.path.append("../../tools/task2b")
    from task2b_common import *

Nothing here modifies solve_task2b.py or check_allocation.py: the notebooks import the solver as a library and run the
official validator from a temporary copy that has the data folder it expects next to it.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
SUBMISSION_DIR = ROOT / "HackBots_Datathon_Submision"
SUBMISSION = SUBMISSION_DIR / "predictions" / "submission_task2b.csv"
METRICS = ROOT / "reports" / "metrics" / "task2b"
FIGURES = ROOT / "reports" / "figures" / "task2b"
INTERIM = ROOT / "data" / "interim" / "task2b"

BUDGET = {"Fresh": 270, "Style": 480, "Tech": 480}   # pre-dawn (Fresh) | trading day (Style + Tech combined), minutes per vehicle
MAX_TRIPS = 2
EPS = 1e-6


def load_tables():
    """The five input tables as DataFrames (orders, fleet, vehicles, district travel, service allowance)."""
    scn = pd.read_csv(RAW / "test" / "task2b_peak_day_scenarios.csv")
    fleet = pd.read_csv(RAW / "test" / "task2b_peak_day_fleet.csv")
    veh = pd.read_csv(RAW / "reference" / "vehicles.csv")
    dt = pd.read_csv(RAW / "reference" / "district_travel.csv")
    al = pd.read_csv(RAW / "reference" / "service_allowance.csv")
    return scn, fleet, veh, dt, al


def import_solver():
    """Import the teammate's solver as a library (its module-level code only loads data; nothing is written)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import solve_task2b as S
    return S


def official_check(submission_path):
    """Run the UNMODIFIED check_allocation.py on a submission.

    As shipped the validator searches for its data under `<its folder>/data`, which does not exist in this repo, so we
    run it from a temporary folder holding an untouched copy of the script plus the five CSVs it needs.
    Returns (passed: bool, output: str).
    """
    need = [("test", "task2b_peak_day_scenarios.csv"), ("test", "task2b_peak_day_fleet.csv"),
            ("reference", "vehicles.csv"), ("reference", "district_travel.csv"), ("reference", "service_allowance.csv")]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "data").mkdir()
        shutil.copy(HERE / "check_allocation.py", tmp / "check_allocation.py")
        for sub, f in need:
            shutil.copy(RAW / sub / f, tmp / "data" / f)
        r = subprocess.run([sys.executable, str(tmp / "check_allocation.py"), str(Path(submission_path).resolve())],
                           capture_output=True, text=True, cwd=tmp)
    out = (r.stdout + r.stderr).strip()
    return ("PASSED" in out and r.returncode == 0), out


def trip_table(sub, scn, veh, dt, al):
    """Independent re-computation (straight from the CSVs, not via the solver) of every trip in a submission:
    brand/district, stops, load vs capacity, trip minutes by the booklet formula, and the per-vehicle budget usage."""
    d = dt.set_index("district")
    allow = al.set_index(["brand", "dock_type"]).service_allowance_min
    v = veh.set_index("vehicle_id")
    m = sub[sub.decision == "served"].merge(scn, on=["scenario", "order_ref", "outlet_id"])
    rows = []
    for (vid, tid), g in m.groupby(["vehicle_id", "trip_id"]):
        dist, brand = g.district.iloc[0], g.brand.iloc[0]
        out, inter = d.loc[dist, "depot_to_district_freeflow_min"], d.loc[dist, "inter_stop_freeflow_min"]
        hand = sum(allow[(brand, dk)] for dk in g.dock_type)
        mins = out + inter * (len(g) - 1) + hand
        rows.append(dict(vehicle_id=vid, trip_id=int(tid), vehicle=f"{v.loc[vid, 'type']}/{v.loc[vid, 'temp']}", brand=brand, district=dist,
                         stops=len(g), chilled=int((g.temp_requirement == 'chilled').sum()), m3=g.order_volume_m3.sum(), m3_cap=v.loc[vid, "volume_cap_m3"],
                         kg=g.order_weight_kg.sum(), kg_cap=v.loc[vid, "weight_cap_kg"], outbound=out, inter_stop=inter * (len(g) - 1),
                         handling=hand, trip_min=mins, orders=" ".join(g.order_ref)))
    t = pd.DataFrame(rows)
    t["bucket"] = np.where(t.brand == "Fresh", "Fresh", "Style/Tech")
    t["bucket_used"] = t.groupby(["vehicle_id", "bucket"]).trip_min.transform("sum")
    t["bucket_budget"] = np.where(t.bucket == "Fresh", 270, 480)
    t["m3_util"] = t.m3 / t.m3_cap
    t["kg_util"] = t.kg / t.kg_cap
    return t


def independent_check(sub, scn, fleet, veh, dt, al):
    """All seven feasibility rules re-implemented from the CSVs. Returns a DataFrame with one row per rule
    (rule, violations) - an independent second opinion next to the official validator."""
    t = trip_table(sub, scn, veh, dt, al)
    v = veh.set_index("vehicle_id")
    avail = set(fleet[fleet.status == "available"].vehicle_id)
    m = sub[sub.decision == "served"].merge(scn, on=["scenario", "order_ref", "outlet_id"])
    m["vtype"] = m.vehicle_id.map(v.type); m["vtemp"] = m.vehicle_id.map(v.temp); m["vdepot"] = m.vehicle_id.map(v.depot)
    mixed = m.groupby(["vehicle_id", "trip_id"]).agg(b=("brand", "nunique"), d=("district", "nunique"))
    res = [
        ("every order decided exactly once", int((sub.order_ref.duplicated()).sum() + (~sub.decision.isin(["served", "deferred"])).sum() + (len(sub) != len(scn)))),
        ("deferred rows have blank vehicle/trip", int(((sub.decision == "deferred") & (sub.vehicle_id.notna() | sub.trip_id.notna())).sum())),
        ("only vehicles marked available", int((~m.vehicle_id.isin(avail)).sum())),
        ("R1 one brand and one district per trip", int(((mixed.b > 1) | (mixed.d > 1)).sum())),
        ("R2 chilled only on reefer", int(((m.temp_requirement == "chilled") & (m.vtemp != "reefer")).sum())),
        ("R3 van_only outlets only on vans", int(((m.parking_constraint == "van_only") & (m.vtype != "van")).sum())),
        ("R4 vehicle serves only its home depot", int((m.vdepot != m.depot).sum())),
        ("R6 volume within capacity", int((t.m3 > t.m3_cap + EPS).sum())),
        ("R6 weight within capacity", int((t.kg > t.kg_cap + EPS).sum())),
        ("R7 at most two trips per vehicle", int((t.groupby("vehicle_id").trip_id.nunique() > MAX_TRIPS).sum())),
        ("R7 trip ids are 1 or 2", int((~t.trip_id.isin([1, 2])).sum())),
        ("R7 time budget (Fresh 270 / Style+Tech 480)", int((t.bucket_used > t.bucket_budget + EPS).groupby(t.vehicle_id).any().sum())),
    ]
    return pd.DataFrame(res, columns=["check", "violations"])


def to_min(hhmm):
    h, m = str(hhmm).split(":")
    return int(h) * 60 + int(m)


def simulate_submission(sub, S):
    """Clock simulation of any submission with the solver's rules (Fresh leaves 03:30, daytime trips 08:00, stops visited by
    earliest window close, vehicle returns after the outbound time). Returns (stops DataFrame, number of late stops).
    A stop is 'late' when it arrives after its window closes."""
    served = sub[sub.decision == "served"].copy()
    served["trip_id"] = served.trip_id.astype(int)
    rows, late = [], 0
    for vid, g in served.groupby("vehicle_id"):
        trips = [[S.ORDERS[r] for r in gg.order_ref] for _, gg in sorted(g.groupby("trip_id"), key=lambda kv: kv[0])]
        r, l = S.simulate(trips)
        late += l
        for x in r:
            x["vehicle_id"] = vid
            x["trip_id"] = x.pop("run_order")
            rows.append(x)
    return pd.DataFrame(rows), late
