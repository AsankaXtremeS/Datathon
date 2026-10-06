"""Task 1 feature pipeline (planning-time information only).

Usage:
    from task1_features import build_feature_tables
    train_X, test_X, meta = build_feature_tables(train_labeled, actuals, test_base, tables)

Feature families
  1. order        size, density, deferral
  2. outlet       dock / parking / mall / window
  3. vehicle      capacity, load factor, fuel
  4. leg + route  planned slack, position, cumulative planned load/travel, stops remaining
  5. context      calendar, payday proximity, traffic speed index, road disruption, district road class
  6. history      smoothed past-only outlet / vehicle / district behaviour (train: strictly earlier dates;
                  test: all of train). Built from actuals, which is legitimate because history is known
                  before the day being predicted.
  7. simulation   roll each route forward with expected travel (traffic x disruption), expected
                  service (allowance x outlet history) and window waiting -> expected arrival and slack.
"""
import numpy as np
import pandas as pd

from task1_common import hhmm_to_min

CATEGORICAL = ["brand", "district", "depot", "temp_requirement", "vehicle_type", "vehicle_temp",
               "dock_type", "parking_constraint", "road_class", "fuel_type", "outlet_id", "vehicle_id"]
LABELS = ["service_min", "late"]

# smoothing strength (pseudo-observations) for history means
M_SMOOTH = 20
# train rows only see history at least this old, mimicking test (2-42 days after train ends, mean ~21)
HIST_GAP_DAYS = 21


# ------------------------------------------------------------------ reference joins
def _join_reference(df: pd.DataFrame, t: dict) -> pd.DataFrame:
    o = t["outlets"][["outlet_id", "dock_type", "parking_constraint", "mall_window"]]
    v = t["vehicles"].rename(columns={"type": "veh_type_ref", "temp": "veh_temp_ref", "depot": "veh_depot"})
    v = v[["vehicle_id", "weight_cap_kg", "volume_cap_m3", "fuel_type", "km_per_l", "weekly_fuel_quota_l"]]
    dt = t["district_travel"][["district", "road_class", "free_flow_kmh", "depot_to_district_km",
                               "depot_to_district_freeflow_min", "inter_stop_km", "inter_stop_freeflow_min"]]
    cal = t["calendar"].assign(date=lambda d: pd.to_datetime(d.date))
    cal = _payday_distance(cal)[["date", "is_weekend", "iso_week", "is_payday", "festival_ramp", "is_holiday",
                                 "days_since_payday", "days_to_payday"]]
    roads = t["roads"].assign(date=lambda d: pd.to_datetime(d.date))
    out = (df.merge(o, on="outlet_id", how="left")
             .merge(v, on="vehicle_id", how="left")
             .merge(dt, on="district", how="left")
             .merge(cal, on="date", how="left")
             .merge(roads, on=["district", "date"], how="left")
             .merge(t["service_allowance"], on=["brand", "dock_type"], how="left"))
    assert len(out) == len(df), "reference join changed row count"
    return out


def _payday_distance(cal: pd.DataFrame) -> pd.DataFrame:
    cal = cal.sort_values("date").copy()
    pay = cal.date.where(cal.is_payday == 1)
    cal["days_since_payday"] = (cal.date - pay.ffill()).dt.days
    cal["days_to_payday"] = (pay.bfill() - cal.date).dt.days
    return cal


# ------------------------------------------------------------------ order / leg / route features
def _basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df.date.dt.month
    df["log_units"] = np.log1p(df.order_units)
    df["log_weight"] = np.log1p(df.order_weight_kg)
    df["log_volume"] = np.log1p(df.order_volume_m3)
    df["density_kg_m3"] = df.order_weight_kg / df.order_volume_m3.clip(lower=1e-3)
    df["kg_per_unit"] = df.order_weight_kg / df.order_units.clip(lower=1)
    df["is_deferred"] = (df.dispatch_status == "deferred").astype(int)
    df["days_deferred"] = (pd.to_datetime(df.dispatch_date) - pd.to_datetime(df.order_date)).dt.days
    df["is_mall"] = df.mall_window.notna().astype(int)
    df["is_chilled"] = (df.temp_requirement == "chilled").astype(int)

    df["window_width"] = df.window_close_time_m - df.window_open_time_m
    df["plan_slack"] = df.window_close_time_m - df.planned_arrival_time_m          # key lateness driver
    df["plan_early_gap"] = (df.window_open_time_m - df.planned_arrival_time_m)     # >0: planned to wait
    df["plan_speed_kmh"] = df.distance_km / (df.planned_travel_duration_min.clip(lower=1) / 60)

    g = df.groupby("route_id")
    df["n_stops"] = g.seq.transform("count")
    df["stops_remaining"] = df.n_stops - df.seq - 1
    df["is_first_stop"] = (df.seq == 0).astype(int)
    df["route_weight"] = g.order_weight_kg.transform("sum")
    df["route_volume"] = g.order_volume_m3.transform("sum")
    df["route_units"] = g.order_units.transform("sum")
    df["load_weight_frac"] = df.route_weight / df.weight_cap_kg
    df["load_volume_frac"] = df.route_volume / df.volume_cap_m3
    df["order_share_of_route"] = df.order_weight_kg / df.route_weight
    df["route_start_m"] = g.planned_depart_time_m.transform("min")
    df["plan_elapsed"] = df.planned_arrival_time_m - df.route_start_m
    df["cum_plan_travel"] = g.planned_travel_duration_min.cumsum()
    df["cum_distance"] = g.distance_km.cumsum()
    df["route_distance"] = g.distance_km.transform("sum")
    df["cum_allowance_prev"] = g.service_allowance_min.cumsum() - df.service_allowance_min
    df["weight_still_on_board"] = df.route_weight - (g.order_weight_kg.cumsum() - df.order_weight_kg)
    df["min_slack_prev_stops"] = g.plan_slack.shift().groupby(df.route_id).cummin()
    return df


# ------------------------------------------------------------------ traffic lookups
def _speed_lookup(traffic: pd.DataFrame):
    arr = np.full((len(traffic.district.unique()), 24, 2), 100.0)
    dist_idx = {d: i for i, d in enumerate(sorted(traffic.district.unique()))}
    for r in traffic.itertuples(index=False):
        arr[dist_idx[r.district], int(r.hour), int(r.monsoon)] = r.speed_index
    def lookup(district, minute, monsoon):
        h = np.clip((np.asarray(minute) // 60).astype(int), 0, 23)
        di = np.array([dist_idx[d] for d in district])
        return arr[di, h, np.asarray(monsoon, dtype=int)]
    lookup.table, lookup.dist_idx = arr, dist_idx
    return lookup


def _traffic_features(df: pd.DataFrame, lookup) -> pd.DataFrame:
    df = df.copy()
    df["speed_idx_depart"] = lookup(df.district.values, df.planned_depart_time_m.values, df.monsoon.values)
    df["speed_idx_arrive"] = lookup(df.district.values, df.planned_arrival_time_m.values, df.monsoon.values)
    df["exp_travel_ratio"] = (100 / df.speed_idx_depart) * (100 / df.disruption_index)
    df["exp_travel"] = df.planned_travel_duration_min * df.exp_travel_ratio
    df["exp_travel_excess"] = df.exp_travel - df.planned_travel_duration_min
    df["cum_exp_travel_excess"] = df.groupby("route_id").exp_travel_excess.cumsum()
    return df


# ------------------------------------------------------------------ history (past-only)
def _past_mean(df: pd.DataFrame, key: str, val: str, prior: float, m: float = M_SMOOTH):
    """Smoothed mean of `val` per `key` using only rows on strictly earlier dates.

    Train rows (val not NaN) look back from `date - HIST_GAP_DAYS`; test rows (val NaN) use all train history.
    Returns (smoothed_mean, n_past).
    """
    src = df.loc[df[val].notna(), [key, "date", val]]
    daily = src.groupby([key, "date"])[val].agg(["sum", "count"]).reset_index().sort_values("date")
    daily["csum"] = daily.groupby(key)["sum"].cumsum()
    daily["ccnt"] = daily.groupby(key)["count"].cumsum()
    q = df[[key, "date"]].reset_index()
    q["date"] = q.date - pd.to_timedelta(np.where(df[val].notna().values, HIST_GAP_DAYS, 0), unit="D")
    q = q.sort_values("date")
    hit = pd.merge_asof(q, daily[[key, "date", "csum", "ccnt"]], on="date", by=key,
                        allow_exact_matches=False).set_index("index").reindex(df.index)
    n = hit.ccnt.fillna(0)
    mean = (hit.csum.fillna(0) + m * prior) / (n + m)
    return mean.values, n.values


def _history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Targets used for history are ratios/residuals with natural priors (0 or a fixed rate),
    so no full-data statistic leaks into the prior."""
    df = df.copy()
    specs = [
        ("outlet_id", "h_log_svc_ratio", 0.0),     # log(service / allowance): outlet handles faster/slower than plan
        ("vehicle_id", "h_log_svc_ratio", 0.0),
        ("outlet_id", "h_late", 0.2),              # outlet late rate
        ("vehicle_id", "h_late", 0.2),
        ("vehicle_id", "h_log_travel_resid", 0.0), # log(actual / expected travel): driver/vehicle speed
        ("district", "h_log_travel_resid", 0.0),
        ("vehicle_id", "h_depart_delay", 7.7),     # first-leg departure delay vs plan
        ("outlet_id", "h_arrival_resid", 0.0),     # actual arrival - simulated arrival (allowance sim)
        ("vehicle_id", "h_arrival_resid", 0.0),
    ]
    for key, val, prior in specs:
        name = f"hist_{key.split('_')[0]}_{val[2:]}"
        df[name], n = _past_mean(df, key, val, prior)
        if val == "h_log_svc_ratio":
            df[f"hist_{key.split('_')[0]}_n"] = n
    return df


# ------------------------------------------------------------------ route simulation
def _simulate(df: pd.DataFrame, lookup, service: np.ndarray, depart_delay: np.ndarray):
    """Roll each route forward. Requires df sorted by (route_id, seq).

    depart_k  = planned depart (+ expected delay) for the first leg, else leave_{k-1}
    travel_k  = planned travel x (100/speed at depart hour) x (100/disruption)
    arrive_k  = depart_k + travel_k ; start service at max(arrive_k, window_open)
    leave_k   = start_k + service_k
    """
    route = df.route_id.values
    pdep = df.planned_depart_time_m.values.astype(float)
    ptrav = df.planned_travel_duration_min.values.astype(float)
    disr = df.disruption_index.values.astype(float)
    opens = df.window_open_time_m.values.astype(float)
    dist = np.array([lookup.dist_idx[d] for d in df.district.values])
    mons = df.monsoon.values.astype(int)
    table = lookup.table
    n = len(df)
    arr = np.empty(n); wait = np.empty(n); cum_wait = np.empty(n)
    leave = 0.0; acc_wait = 0.0
    for i in range(n):
        if i == 0 or route[i] != route[i - 1]:
            dep = pdep[i] + depart_delay[i]; acc_wait = 0.0
        else:
            dep = leave
        speed = table[dist[i], min(max(int(dep // 60), 0), 23), mons[i]]
        arr[i] = dep + ptrav[i] * (100 / speed) * (100 / disr[i])
        wait[i] = max(opens[i] - arr[i], 0.0)
        cum_wait[i] = acc_wait
        acc_wait += wait[i]
        leave = arr[i] + wait[i] + service[i]
    return arr, wait, cum_wait


def _simulation_features(df: pd.DataFrame, lookup, prefix: str, service, depart_delay) -> pd.DataFrame:
    arr, wait, cum_wait = _simulate(df, lookup, service, depart_delay)
    df[f"{prefix}_arrival"] = arr
    df[f"{prefix}_slack"] = df.window_close_time_m.values - arr
    df[f"{prefix}_delay_vs_plan"] = arr - df.planned_arrival_time_m.values
    df[f"{prefix}_wait"] = wait
    df[f"{prefix}_cum_wait_prev"] = cum_wait
    return df


# ------------------------------------------------------------------ main entry
def build_feature_tables(train: pd.DataFrame, actuals: pd.DataFrame, test: pd.DataFrame, t: dict):
    """Return (train_X, test_X, meta). Both tables share one feature list; labels only in train_X."""
    lookup = _speed_lookup(t["traffic"])
    tr = train.merge(actuals, on="delivery_id", how="left").assign(_split="train")
    te = test.assign(_split="test")
    df = pd.concat([tr, te], ignore_index=True)
    df["date"] = pd.to_datetime(df.date)

    df = _join_reference(df, t)
    df = df.sort_values(["route_id", "seq"]).reset_index(drop=True)
    df = _basic_features(df)
    df = _traffic_features(df, lookup)

    # simulation 1: plan service allowances, mean historical first-leg departure delay (no history needed)
    df = _simulation_features(df, lookup, "sim_allow", df.service_allowance_min.values.astype(float),
                              np.full(len(df), 7.7))

    # history targets (train rows only; test rows stay NaN and only *receive* history)
    is_tr = df._split.eq("train")
    df["h_log_svc_ratio"] = np.where(is_tr, np.log(df.service_min / df.service_allowance_min), np.nan)
    df["h_late"] = np.where(is_tr, df.late, np.nan)
    df["h_log_travel_resid"] = np.where(is_tr, np.log(df.actual_travel_duration_min.clip(lower=1) / df.exp_travel), np.nan)
    df["h_depart_delay"] = np.where(is_tr & df.seq.eq(0), df.actual_depart_time_m - df.planned_depart_time_m, np.nan)
    df["h_arrival_resid"] = np.where(is_tr, df.arrival_time_m - df.sim_allow_arrival, np.nan)
    df = _history_features(df)

    # simulation 2: personalised - outlet-specific service, vehicle travel speed and departure delay
    svc_hist = df.service_allowance_min * np.exp(df.hist_outlet_log_svc_ratio)
    df["exp_service_hist"] = svc_hist
    df["exp_travel_hist"] = df.exp_travel * np.exp(df.hist_vehicle_log_travel_resid)
    df_sim = df.assign(disruption_index=df.disruption_index / np.exp(df.hist_vehicle_log_travel_resid))  # scale travel per vehicle
    df_sim = _simulation_features(df_sim, lookup, "sim_hist", svc_hist.values, df.hist_vehicle_depart_delay.values)
    for c in [c for c in df_sim.columns if c.startswith("sim_hist_")]:
        df[c] = df_sim[c]
    g = df.groupby("route_id")
    df["cum_exp_service_prev"] = g.exp_service_hist.cumsum() - df.exp_service_hist
    df["plan_vs_hist_service_gap_prev"] = df.cum_exp_service_prev - df.cum_allowance_prev

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    drop = set(actuals.columns) - {"delivery_id"} | {c for c in df.columns if c.startswith("h_")} | {
        "order_date", "dispatch_date", "dispatch_status", "route_id", "seq_in_route", "leg_id", "from_point",
        "to_outlet", "planned_arrival_time", "planned_arrival_time_leg", "planned_depart_time", "window_open_time",
        "window_close_time", "mall_window", "date", "_split", "delivery_id",
        # time proxies: history counts only grow (test is out of range), iso_week is year-specific and duplicates month
        "hist_outlet_n", "hist_vehicle_n", "iso_week"} | {c for c in df.columns if c.endswith("_leg")}
    features = [c for c in df.columns if c not in drop and c not in LABELS]
    meta = {"features": features, "categorical": [c for c in CATEGORICAL if c in features], "labels": LABELS}

    keep = [c for c in ["delivery_id", "date", "route_id", "seq"] if c not in features]
    train_X = df.loc[is_tr, keep + features + LABELS].reset_index(drop=True)
    test_X = df.loc[~is_tr, keep + features].reset_index(drop=True)
    return train_X, test_X, meta
