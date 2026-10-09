Hello, we are Team HackBots. This is our Datathon demo, covering all three tasks. We will walk through our notebooks and explain how we solved each one.

Across all three tasks we followed one rule: find the structure in the data first, model only what is left, and keep an idea only if it wins on forward-in-time validation.

Task one has no labels, so we built them. Service time is departure minus the later of arrival and window opening, because an early vehicle waits, and waiting is not handling. Late means arriving after the window closes.

*Open: 02_label_construction → 3. Why the wait must be excluded*

The key insight: the dispatcher's plan uses standard handling times and assumes empty roads, yet real travel follows planned time, traffic and road disruption almost exactly. So instead of feeding the plan to a model, we re-simulate every route stop by stop. That alone lifts lateness ranking quality from about point eight six to point nine five, before any model is trained.

*Open: 03_feature_engineering → 2. Two facts behind the simulation*

Actual times exist only in training, so we lag every history feature by twenty-one days, matching the gap to the test period, and validate on folds that move forward in time. We then compared our stacked gradient boosting pipeline with several alternatives on identical folds.

*Open: 05c_model_comparison → 4. Lateness: all metrics*

Service-time error is three point six minutes, against six point six for the dispatcher's own allowance. Lateness scores point nine eight, and the probabilities are well calibrated.

*Open: 05b_model_evaluation → 8. Summary and save*

In Task 2A, order counts follow a fixed weekday schedule, so demand moves only through order size. We therefore forecast every operating day and add them up, which handles holidays and short weeks by construction.

*Open: 01_data_audit → 3. The order schedule is (almost) deterministic*

Festivals are not equal. The New Year run-up dwarfs Vesak and Poson, so each festival gets its own ramp.

*Open: 03_feature_engineering → 1. Order size is the moving part: festival run-up, by festival*

We built the model as a ladder, adding one idea at a time and keeping it only if every error metric improved across thirty-four rolling backtests. Trend, per-festival ramps and a small boosting layer over a simple ridge earned their place. Tech, which is mostly noise, stays simple.

*Open: 05_modeling → 1. Fresh and Style ladder*

Fresh is forecast within about three percent a week, against six for the best baseline. Chilled is within three and a half, and Style within eight. The weak spot is short holiday weeks, and we say so plainly.

*Open: 05b_error_analysis → 2. Festival run-up weeks vs other weeks; short weeks*

Task 2B is a choice, not a prediction, so we first found what limits the day. Chilled demand is more than double what the four available refrigerated trucks can carry per wave, and serving it all needs more driving minutes than they have before stores open. Everything else fits, so the real decision is which chilled orders wait.

*Open: 03_priorities_and_bottlenecks → 2. The binding resource: reefer capacity, in volume, weight and time*

We solve it exactly. Refrigerated trips are an integer program over whole day plans, so the clock is exact, and everything else is an exact search that keeps every stop inside its delivery window. It runs in seconds to a minute, proves the plan optimal, and leaves just one stop arriving late by a literal clock.

*Open: 04_exact_solver → 1. Stage R: the reefer problem*

Priority is where judgement matters. We keep the written order as strict levels: outlets skipped yesterday, then chilled orders, then days since served. A weighted score only matches that order once its weight passes about twenty-eight, so strict levels are the safer encoding. Measured exactly, serving every skipped outlet costs two chilled orders. That is a business decision, so we present both plans.

*Open: 05_policies_and_whatifs → 3. Where does the weighted score flip?*

One repaired refrigerated truck removes the dilemma entirely: either policy then serves twenty-three of twenty-six chilled orders. Every plan passes the official validator and our independent check.

*Open: 05_policies_and_whatifs → 4. Value of releasing a workshop reefer*
*Open: 06_validation_and_comparison → Head-to-head*

In every task the pattern was the same: understand the structure, keep only what validates, and state the limits plainly. Thank you.
