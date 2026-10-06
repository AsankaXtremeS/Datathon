# Task 1 – Data Preprocessing, Label Construction and Feature Engineering

**Team HackBots · Tech-Triathlon 2026 Datathon**

Task 1 predicts, for each planned delivery, the handling time at the outlet (`pred_service_min`) and the probability of arriving after the delivery window closes (`pred_late_prob`). The code for every step is in `HackBots_FinalNotebook.ipynb` (sections 2–5).

---

## 1. Data preparation

| Step | What we did | Result |
|---|---|---|
| Join orders to route legs | `deliveries_train` ⟷ `route_legs_train` on `route_id` + `seq_in_route` = `seq` (one-to-one enforced). The same join is used for the test files. | 91,894 train and 5,014 test deliveries. Every dispatched order matches exactly one leg, and no leg is left over. |
| Consistency checks | Compared the fields that both files carry: brand, district, depot, vehicle, outlet = `to_outlet`, planned arrival, and dispatch date = leg date. Also checked the order window against `outlets.csv`. | 100% agreement. No repairs needed. |
| Time handling | Converted HH:MM clock times to minutes since midnight. Checked for midnight wrap. Checked `arrival = actual depart + actual travel`. | No wraps. The arithmetic holds within ±1 min (rounding), so minute arithmetic is exact. |
| Excluded rows | 413 `not_run` orders were never dispatched, so they have no route, vehicle or times. | Removed from Task 1 only, because no label exists. They still count as demand in Task 2A. |
| Reference coverage | Every train and test row finds its calendar, traffic (district × hour × monsoon) and road-condition (district × date) record. | 100% coverage. No imputation needed. |

**Cleaning decisions.** We found no corrupt records. Long stays and long delays were analysed rather than deleted:
- **Delays:** they track road disruption. The mean delay is about 168 min when `disruption_index` ≤ 60, against about 39 min on clear days.
- **Service times above 150 min:** these 89 rows belong to orders 2–3× heavier than their brand's typical order. Every brand-and-dock group has almost exactly 1% of rows above its own 99th percentile.

Both are real behaviour that the model must learn, so all rows were kept.

---

## 2. Label construction

Neither target is supplied, so both are derived from the training route-leg actuals.

| Label | Definition |
|---|---|
| `service_min` | `leave_outlet_time − max(arrival_time, window_open_time)` |
| `late` | `arrival_time > window_close_time` |

**Why `max(arrival, window_open)`.** Outlets receive goods only inside their window. A vehicle that arrives early waits until the window opens, and that wait is not handling time.
- 4.4% of deliveries arrive early, with a mean wait of 20 min.
- For those deliveries, the raw stay (`leave − arrival`) correlates **0.81** with the wait (0.92 for Fresh rear-dock outlets), but the corrected label correlates only −0.11.
- Example: Fresh rear-dock median raw stay is 28 min when early against 15 min when on time. The corrected label gives 12 min and 15 min.
- A naive label would overstate service time on 4.4% of rows by about 20 min.

**Why strict `>` for lateness.** The booklet defines lateness as arrival *after the window closes*. Only 0.3% of rows arrive exactly at the close (`>` gives 19.58%, `>=` gives 19.88%), so the choice is not material.

**Validation of the labels**
- No non-positive or missing service times (range 2–428 min). Medians sit close to the dispatcher allowance (Fresh 15 vs 15–16).
- The late rate falls steadily with planned slack: 98% when already past close, 79% at 15–30 min, 25% at 60–90 min, 1.5% above 180 min.
- Brand rates: Fresh 20.4%, Style 5.5%, Tech 2.5%.

**Leakage rule.** A column may be a model input only if it also exists in the test table. Actual times (`actual_*`, `arrival_time`, `leave_outlet_time`) exist only in training, so they enter the features **only as past aggregates** (history).

---

## 3. Feature engineering (101 features, all available before departure)

Two findings drive the design:
1. The dispatcher's plan uses **exactly** the service allowance as dwell time at every stop and assumes clear roads (planned dwell − allowance = 0 on every leg).
2. Actual travel ≈ planned travel × (100 / `speed_index`) × (100 / `disruption_index`): log correlation 0.98, median ratio 1.00.

The plan is therefore optimistic in a predictable way, and we **re-simulate every route realistically**.

| Family | Features | Rationale |
|---|---|---|
| Order | log units, weight and volume; density; kg per unit; deferred flag and days deferred | Service time scales with size (correlation about 0.55). |
| Outlet | dock type, parking constraint, mall flag, window open, close and width, `outlet_id` (categorical) | Outlet explains 53% of service-time variance. |
| Vehicle | weight and volume capacity, route load factor, fuel profile, `vehicle_id` | One driver per vehicle; vehicle explains 13% of variance. |
| Leg and route | planned slack (`window_close − planned_arrival`), stop position, stops remaining, cumulative planned travel and distance, weight still on board, tightest slack at earlier stops | Delay accumulates along a route: median 24 min at the first stop, 56 min by the sixth. |
| Context | monsoon, day of week, payday distance, festival ramp, holiday, road class; traffic speed index at the planned hour; road disruption; expected travel ratio | Travel delay is driven by traffic × disruption. |
| History (past-only) | smoothed outlet and vehicle service ratio (service ÷ allowance); outlet and vehicle late rate; vehicle and district travel-speed residual; vehicle first-leg departure delay; outlet and vehicle arrival residual | Outlet handling and driver behaviour are stable over time. |
| **Route simulation** | Each route is rolled forward stop by stop: expected travel = planned × traffic × disruption × vehicle speed residual; wait if early; service = allowance × outlet history; first departure + vehicle's usual delay. Output: expected arrival, slack, wait. | Simulated slack AUC is **0.954**, against 0.859 for planned slack. History-based expected service correlates 0.71 with actual (log scale), against 0.45 for the allowance. |

**Safeguards**
- **History lag:** history for training rows uses only data at least **21 days** old, because test rows are 2–42 days after training ends. History is smoothed toward fixed priors (0 for log ratios, 0.2 late rate, 7.7 min departure delay), so no future or full-data statistic leaks in.
- **Dropped time proxies:** history counts (they only grow, so test values fall outside the training range) and ISO week (duplicates month).
- **Adversarial validation**, train vs test with folds grouped by route: AUC 0.66. The remaining difference is real (test roads are more disrupted, mean 89.9 vs 92.7) and is modelled explicitly.
- **Missing values** are structural only (first stop has no earlier stops; early-2024 rows have no earlier payday). HistGradientBoosting handles them natively.

---

## 4. Rationale summary

- **Labels** follow the booklet's operating rules: wait until the window opens, and late means after the close.
- **Features** turn the dispatcher's optimistic plan into a realistic ETA using only information available at planning time.
- **Validation** reproduces test conditions: forward-in-time folds with history frozen at the cutoff.
- **Scores:** service MAE **3.58 min** against 6.61 for the dispatcher's allowance; lateness AUC **0.979** and log-loss **0.136** against 0.318 for a planned-slack model.
