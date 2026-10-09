TASK 1 VIDEO

Hello, we are Team HackBots. This is Task one, predicting service time and lateness for every planned delivery. We'll walk through our notebooks one by one, in the order they appear in the outline.

Phase one, data audit and cleaning. Every dispatched order joins one-to-one to its route leg, every row finds its calendar, traffic and road record, and we found nothing corrupt. The four hundred and thirteen never-run orders carry no labels, so they leave Task one. The long service times stay, because they are real orders, not errors.

Phase two, label construction. Neither target exists in the data, so we derive them from the actuals. Service time is leave time minus the later of arrival and window opening, because an early vehicle simply waits, and waiting is not handling. Late means arriving strictly after the window closes. We tested both choices against the data before keeping them.

Phase three, feature engineering. Everything here is known before the vehicle leaves. The key finding is that the dispatcher's plan assumes empty roads and standard handling, while real travel follows planned time, traffic and road disruption almost exactly. So we re-simulate every route stop by stop, and that lifts lateness ranking from about point eight six to point nine five before any model is trained. History is lagged twenty-one days to match the gap to the test period.

Phase four, validation design and baselines. We want a score we can trust, so we use three six-week folds that always move forward in time, including a monsoon-heavy season like the test window, which any Colombo to Kandy driver would recognise. Each fold rebuilds its features with history frozen at the cutoff, exactly as the test will see it. Simple models set the bar that the real model has to beat.

Phase five, modelling. It is gradient boosting throughout, stacked in stages. A service model feeds a second route simulation, an arrival-correction model refines the slack, and a bagged classifier produces the lateness probability. Training predictions are out-of-fold and grouped by route, so no stacked feature ever sees its own row.

Phase five B, model evaluation, split into lateness and service time and scored on unseen later weeks. Service-time error is three point six minutes against six point six for the dispatcher's own allowance. Lateness scores point nine eight, and its log-loss is less than half that of a planned-slack model.

Phase five C, model comparison. Five families on identical features and folds: linear, random forest, extra trees, a neural network and gradient boosting. Boosting won, and blending in the runners-up gained under one percent, so we kept the simpler single family.

Phase five D, post-hoc calibration. Validation showed small, stable biases: service time about point six of a minute low, and lateness about point four of a point low. We corrected them with parameters fitted on some folds and tested on the held-out one, never on the rows being scored.

Phase six, prediction and submission. We predict the five thousand and fourteen test deliveries, keep the template's ids and order exactly, and check that predicted lateness behaves like validation: probability falls as simulated slack grows, and the overall rate suits a monsoon-heavy window. Thank you.

---

TASK 2A VIDEO

Hello, we are Team HackBots. This is Task 2A, forecasting weekly demand for each depot and brand, ten weeks ahead. Let's go through the notebooks in order.

Notebook one, data audit. The training and test files stitch together with no gap, and history ends exactly where the forecast begins. The big discovery is that order counts follow a fixed weekday schedule, so demand moves only through order size. And the window is festival-heavy: Avurudu, Vesak and Poson, including two short weeks.

Notebook two, target construction. Every order counts once, deferred and never-run included, and weeks follow the order date. Chilled demand exists only for Fresh, so Style and Tech chilled is exactly zero. We build a daily panel of operating days and sum it to weeks, so holidays are structural. The test file lacks never-run orders, so we gross those days up using shares measured on training.

Notebook three, feature engineering, evidence first. Paydays and weekdays matter, and monsoon does not. Festivals are not equal: the New Year run-up dwarfs Vesak and Poson, so each festival gets its own ramp. Closed days are lost demand, not postponed, so no catch-up feature is needed.

Notebook four, validation design and baselines. Thirty-four rolling origins, each forecasting ten weeks, always forward in time, scored with several metrics because the official one is unknown. Recent averages and same-week-last-year set the bar the model must beat.

Notebook five, modelling, built as a ladder. We add one idea at a time and keep it only if every metric improves: trend, then per-festival ramps, then a small boosting layer over a simple ridge. Fresh weekly error falls from about five percent to three point two along the way. Style does better with no trend at all, while Fresh keeps a damped one. Chilled blends a direct model with total times chilled share. Tech stays simple, because it is mostly noise.

Notebook five B, error analysis and uncertainty. Fresh sits near three percent a week, chilled near three and a half, Style near eight and Tech near thirty. Error stays flat across the ten-week horizon, because the model runs on the calendar rather than on recent lags. Short holiday weeks are the weak spot, and we say so plainly. Forecast bands are calibrated out of time, so their coverage is honest.

Notebook five C, robustness and stress tests. The settings sit on a flat plateau, so nothing is tuned to noise. Dropping either observed New Year moves the forecast by about one percent, and sensible model variants agree within a percent or so everywhere except Poson week, which is our least certain week. Ideas that did not help, like catch-up demand, are recorded and rejected.

Notebook six, final fit and forecast. We forecast the ten weeks over each week's operating days and sanity-check against last year: New Year week peaks, and the four-day holiday week dips. A reloaded model reproduces the submission exactly, and we add eighty percent bands and a chilled-capacity read-out per depot, which is what a planner needs for vehicles and drivers. Thank you.

---

TASK 2B VIDEO

Hello, we are Team HackBots. This is Task 2B, the peak-day allocation for the Peliyagoda depot in a festival week, when demand is higher than the fleet can carry. There is no model here; it is a decision, and we'll show how we made it defensible.

Notebook one, data audit. Only twenty-eight of thirty-eight vehicles are usable, and five of the ten in the workshop are refrigerated. Chilled demand is more than double what the four available refrigerated trucks carry in one wave, while ambient demand sits far below ambient capacity.

Notebook two, rule engine and trip-time calculation. We turn the booklet's rules into code, with trip time as outbound time plus inter-stop time plus handling, inside two hundred and seventy minutes for Fresh. We test it on the booklet's own worked examples, one hundred and one and one hundred and twelve minutes. An independent re-implementation agrees on every trip. One order is simply larger than any vehicle, so that deferral is unavoidable whatever we do.

Notebook three, priorities and bottlenecks. Two things limit the day. Volume: even two full loads per reefer carry ninety-five percent of chilled demand. And time: serving every chilled order needs about thirteen hundred driving minutes, against ten eighty available before stores open. So the real question is which chilled orders wait.

Notebook four, the exact solver. Refrigerated trips are an integer program over whole day plans, so the clock is exact. Everything else is an exact search for each district that keeps every stop inside its delivery window. Trucks differ only by capacity, so we plan trips once and then match them to real trucks, and that match is verified. It proves the plan optimal, runs in seconds to a minute, and leaves just one stop arriving late by a literal clock.

Notebook five, policy trade-offs and what-ifs. We keep the written priority order as strict levels: outlets skipped yesterday first, then chilled orders, then days since served. A weighted score would only match that order past a weight of about twenty-eight. Serving every skipped outlet costs two chilled orders, so we present both plans as a business decision. Both are exact solves, so the difference between them is the policy, not the search. And one repaired refrigerated truck removes the dilemma: twenty-three of twenty-six chilled orders under either policy.

Notebook six, validation and comparison. Every plan passes the official validator and our own independent check of all the rules. Deferrals are classified honestly: one is unavoidable, the rest come from reefer capacity, and which ones wait is our stated policy choice.

Notebook seven, final allocation and policy. The trip plan, the stop schedule and the one-page policy are generated straight from the solved numbers, so the write-up can never drift from the plan. Under the written policy, seventy-seven of eighty-five orders are served, including every outlet that was skipped yesterday. Thank you.
