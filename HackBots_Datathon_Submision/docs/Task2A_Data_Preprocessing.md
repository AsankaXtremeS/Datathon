# Task 2A - Data preparation, label construction, features and model

**Team HackBots - Tech-Triathlon 2026 Datathon.**
Code: all Task 2A code is reproduced in `HackBots_FinalNotebook.ipynb` (Task 2A sections). Notebook numbers below (01-06) refer to the development notebooks in our repository.
Outputs: `predictions/submission_task2a.csv` (+ `task2a_forecast_with_bands.csv`), `models/task2a_models.joblib`, `docs/architecture/task2a_architecture.png`.

## 1. Data preparation (notebooks 01-02)

| Step | What we did | Result |
|---|---|---|
| Stack orders | `deliveries_train` + `task1_test_inputs`, every order once, including `deferred` and `not_run` (they are demand). | History 2024-01-01 to 2026-03-28 = ISO week 13 of 2026. No overlap or gap with the forecast, which starts in week 14. |
| Week assignment | By `order_date`, ISO year/week from `calendar.csv`. Using `dispatch_date` would move deferred orders into other weeks. | Matches the booklet rule. |
| Daily panel | Sum volume per depot x brand x **operating day**; operating days with no orders count as 0. | 4,170 rows (6 series x 695 days). No order falls on a non-operating day and Sundays never operate. |
| Chilled | Volume of `temp_requirement = chilled` orders. Style and Tech have none, so their target is exactly 0. | Fresh chilled share is stable at about 0.36-0.38. |
| Not-run gross-up | The Task 1 test file has no `not_run` orders, so the days from 2026-02-16 are scaled by `1/(1 - train not_run share)` per series. | 0% to 2.6% depending on the series; derived from train only. |
| Reconciliation | Asserts: one row per series x day; panel total equals raw order total on untouched days. | Pass. |

## 2. What drives demand (notebooks 01, 03)
- Order **counts** follow a fixed weekday schedule (Fresh Peliyagoda 79/88/77/87/75/84 Mon-Sat, std < 1). Style and Tech order only on depot-specific weekdays. Demand therefore moves through order **size**.
- Size rises with `festival_ramp`, on paydays (about +10-12%), on Fri/Sat, and with slow growth for Fresh. Monsoon has no demand effect.
- **Festivals differ in size**: the Sinhala New Year run-up lifts Fresh volume far more than Vesak, Poson or Esala, so each festival gets its own ramp feature.
- Closed days are lost demand, not postponed: the day after a closure looks like any other day.
- The forecast window is festival-heavy: week 15 New Year run-up, week 16 (4 operating days), week 18 Vesak (5 operating days), week 22 Poson plus two paydays.

## 3. Features (all known in advance from `calendar.csv`)
Weekday dummies, `is_payday`, `is_holiday`, `festival_ramp` and its square, one ramp per festival (`r_new_year`, `r_vesak`, `r_poson`, ...), damped trend (years since 2024-01-01), and for the residual learner: days to next / since last festival and ISO week. No lagged demand is used, so a 10-week horizon needs no recursion.

## 4. Model (notebook 05)
Every model forecasts **mean daily volume on operating days**; the weekly forecast is the sum over the operating days of each ISO week, so holidays and short weeks are handled structurally.
- **Fresh:** one ridge on log daily volume for both depots (depot x weekday intercepts, depot trend), back-transformed with the lognormal mean, plus a small boosting model (depth 3, 60 trees) on the ridge residuals. Damped trend (phi 0.9 per week).
- **Style:** the same, **without a trend** (the backtest prefers it on every metric).
- **Tech:** per-depot ridge on volume with weekday, payday, ramp and damped trend, strongly regularised (the series is mostly noise).
- **Chilled (Fresh):** 50/50 blend of a direct chilled model and total x chilled share, capped at total. Style / Tech chilled = 0.

## 5. Validation (notebooks 04, 05, 05b, 05c)
34 rolling origins (every 2 weeks from 2024-10-07), 10-week horizon, forward in time; scored on weekly depot x brand totals with MAE, RMSE, WAPE and bias.

| Series | Final model MAE / RMSE / WAPE | Best baseline WAPE |
|---|---|---|
| Fresh total | 23.6 / 32.2 / **3.18%** | 6.0% (last-26-week mean) |
| Style total | 8.5 / 13.8 / **7.61%** | 9.4% (seasonal naive) |
| Tech total | 7.7 / 9.4 / **29.4%** | 30.3% (last-26-week mean) |
| Fresh chilled | 9.6 / 12.9 / **3.54%** | share-only 3.84% |

Fresh ablation (WAPE): no trend 5.2% -> damped trend 4.0% -> New Year ramp 3.6% -> ramp per festival 3.25% -> pooled depots 3.26% -> residual boosting 3.18%. Pooling the depots is neutral for Fresh and a small gain for Style; the large gains come from the trend, festival-specific ramps and boosting.

Selection rules followed: ridge first, boosting kept only if consistently helpful (better in 21/34 Fresh and 25/34 Style origins, better median, clear gain in festival windows); trend chosen per brand by lowest rolling error (Fresh damped ~ full; Style none; Tech damped); mean-based forecasts; all three metrics agree on best and worst models.

Robustness (05c): hyper-parameter sweeps form a plateau (Fresh WAPE 3.15-3.23%); dropping either observed New Year changes the week 15-16 forecast by about 1%; plausible model variants agree within about 1% except week 22.

Tested and rejected: festival-specific effects shared across all festivals (week 22 over-forecast by 8-10%), catch-up demand features around closures, weekday-specific chilled shares, Poisson and Poisson-boosting models for Tech, and a multiplicative correction for Style festival weeks (unstable across split directions).

## 6. Known limits
- **Short weeks** (fewer than 6 operating days) are under-forecast: Fresh bias -4.7%, Style -12.5%, Tech -27%. The real window has two (weeks 16 and 18).
- **Style festival run-up weeks** are under-forecast in the backtest (about -14%), mostly from origins that had seen few festivals; Style week 15 is the main residual risk.
- **Tech** is mostly noise: about 30% WAPE for every method tried.
- The New Year effect rests on two observed seasons; leaving either out moves the forecast by about 1%.
- 80% forecast bands are out-of-time calibrated (coverage Style 81%, Tech 82%, Fresh 91%, i.e. conservative for Fresh).
