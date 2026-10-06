"""Task 1 final model: stacked route-aware pipeline (scikit-learn only).

Stage 1  service model (raw minutes)             -> pred_svc for every stop (out-of-fold on train)
Stage 2  re-simulate each route with pred_svc     -> sim_pred_arrival / sim_pred_slack
Stage 3  arrival-residual model (actual - sim)    -> pred_slack2 = sim_pred_slack - predicted residual (OOF on train)
Stage 4  lateness classifier on features + stages 1-3, bagged over 5 seeds
Final service = mean(raw-target bag, log-target bag); Style/Tech rows also averaged with a Style/Tech-only model.

Out-of-fold predictions use GroupKFold by route, so no training row sees a model fitted on its own route.
Models are stored as a dict of plain scikit-learn estimators (joblib), loadable without this module.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import task1_features as tf

SEED = 42
SVC_STAGE1 = dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=15)
SVC_FINAL = dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=15, max_features=0.5)
SVC_SEGMENT = dict(max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20)
RESID = dict(max_iter=400, learning_rate=0.05)
LATE = dict(max_iter=1500, learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=50, max_features=0.5)
N_BAG_SVC, N_BAG_LATE, N_OOF = 3, 5, 5
SEGMENT_BRANDS = ["Style", "Tech"]
STACK_FEATURES = ["pred_svc", "sim_pred_arrival", "sim_pred_slack"]


def _reg(params, seed=SEED):
    return HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=seed, **params)


def _clf(params, seed=SEED):
    return HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=seed, **params)


def _oof(make, X, y, groups):
    """Out-of-fold predictions grouped by route, plus the model refitted on all rows."""
    oof = np.zeros(len(X))
    for a, b in GroupKFold(N_OOF).split(X, groups=groups):
        oof[b] = make().fit(X.iloc[a], y.iloc[a]).predict(X.iloc[b])
    return oof, make().fit(X, y)


def simulate_with_service(df: pd.DataFrame, service: np.ndarray, lookup) -> np.ndarray:
    """Expected arrival (minutes) per row, rolling each route with the given per-stop service times.
    Uses the vehicle's historical travel-speed residual and departure delay (same as sim_hist)."""
    d = df.assign(_svc=np.asarray(service, float)).sort_values(["route_id", "seq"])
    d = d.assign(district=d.district.astype(str),
                 disruption_index=d.disruption_index / np.exp(d.hist_vehicle_log_travel_resid))
    arr, _, _ = tf._simulate(d, lookup, d._svc.values, d.hist_vehicle_depart_delay.values)
    return pd.Series(arr, index=d.index).reindex(df.index).values


def _add_stack(df, pred_svc, lookup):
    df = df.copy()
    df["pred_svc"] = pred_svc
    df["sim_pred_arrival"] = simulate_with_service(df, pred_svc, lookup)
    df["sim_pred_slack"] = df.window_close_time_m - df.sim_pred_arrival
    return df


def fit(train: pd.DataFrame, arrival_m: pd.Series, features: list, traffic: pd.DataFrame, verbose=True) -> dict:
    """train: feature table with labels; arrival_m: actual arrival minutes aligned to train rows."""
    log = print if verbose else (lambda *a, **k: None)
    lookup = tf._speed_lookup(traffic)
    tr = train.reset_index(drop=True)
    arrival_m = pd.Series(np.asarray(arrival_m, float))
    groups = tr.route_id.values

    log("stage 1: service OOF")
    svc_oof, svc1 = _oof(lambda: _reg(SVC_STAGE1), tr[features], tr.service_min, groups)
    tr = _add_stack(tr, svc_oof, lookup)

    log("stage 3: arrival-residual OOF")
    f2 = features + STACK_FEATURES
    res_oof, resid = _oof(lambda: _reg(RESID), tr[f2], arrival_m - tr.sim_pred_arrival, groups)
    tr["pred_slack2"] = tr.sim_pred_slack - res_oof

    log(f"stage 4: lateness classifier x{N_BAG_LATE}")
    f3 = f2 + ["pred_slack2"]
    late = [_clf(LATE, seed=s).fit(tr[f3], tr.late) for s in range(N_BAG_LATE)]

    log(f"final service: raw x{N_BAG_SVC}, log x{N_BAG_SVC}, segment")
    svc_raw = [_reg(SVC_FINAL, seed=s).fit(tr[features], tr.service_min) for s in range(N_BAG_SVC)]
    svc_log = [_reg(SVC_FINAL, seed=s).fit(tr[features], np.log(tr.service_min)) for s in range(N_BAG_SVC)]
    seg = tr.brand.isin(SEGMENT_BRANDS).values
    svc_seg = _reg(SVC_SEGMENT).fit(tr.loc[seg, features], tr.service_min[seg])

    return {"features": features, "f2": f2, "f3": f3, "svc_stage1": svc1, "resid": resid, "late": late,
            "svc_raw": svc_raw, "svc_log": svc_log, "svc_segment": svc_seg, "segment_brands": SEGMENT_BRANDS}


def predict(models: dict, df: pd.DataFrame, traffic: pd.DataFrame, return_parts=False):
    """Return (pred_service_min, pred_late_prob) aligned to df rows."""
    lookup = tf._speed_lookup(traffic)
    d = df.reset_index(drop=True)
    X = d[models["features"]]
    d = _add_stack(d, models["svc_stage1"].predict(X), lookup)
    d["pred_slack2"] = d.sim_pred_slack - models["resid"].predict(d[models["f2"]])
    late = np.mean([m.predict_proba(d[models["f3"]])[:, 1] for m in models["late"]], axis=0)

    raw = np.mean([m.predict(X) for m in models["svc_raw"]], axis=0)
    lg = np.mean([np.exp(m.predict(X)) for m in models["svc_log"]], axis=0)
    svc = (raw + lg) / 2
    seg = d.brand.isin(models["segment_brands"]).values
    if seg.any():
        svc[seg] = (svc[seg] + models["svc_segment"].predict(X[seg])) / 2
    svc = np.clip(svc, 1.0, None)
    late = np.clip(late, 1e-4, 1 - 1e-4)
    if return_parts:
        return svc, late, {"svc_raw": raw, "svc_log": lg, "svc_stage1": d.pred_svc.values,
                           "late_slack_sim": d.sim_pred_slack.values, "pred_slack2": d.pred_slack2.values}
    return svc, late
