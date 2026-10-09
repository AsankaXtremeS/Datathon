# Demo video – Task 1 segment (outline)

The full video is 3–5 minutes and must cover model architecture, preprocessing, label construction and challenges. Suggested time for Task 1: **about 2 minutes**, leaving room for Task 2A and 2B.

| Time | Show | Say |
|---|---|---|
| 0:00–0:15 | Booklet Task 1 | "For every planned delivery we predict handling time and the probability of arriving after the window closes." |
| 0:15–0:45 | Final notebook §3 (label evidence output) | "Neither label exists. Service time = leave − max(arrival, window open), because an early vehicle waits for the window and that wait isn't handling. For early arrivals the raw stay correlates 0.81 with the wait, and our label removes it. Late means arrival strictly after the close." |
| 0:45–1:15 | Architecture diagram, band 1 | "Key finding: the dispatcher's plan assumes clear roads and standard handling times. Actual travel follows traffic × road disruption almost exactly, so we re-simulate every route realistically. That alone lifts lateness AUC from 0.86 to 0.95." |
| 1:15–1:40 | Architecture diagram, band 2 | "The model is a stacked HistGradientBoosting pipeline. Predicted service times feed a second route simulation, then an arrival-correction model, then a bagged lateness classifier. It beat logistic regression, random forest, extra trees and a neural net on identical folds." |
| 1:40–2:00 | §7 results table and confusion matrix | "On three forward-in-time folds: service error 3.6 min against 6.6 for the dispatcher's allowance; lateness AUC 0.98, F1 0.81, accuracy 94%, well calibrated." |
| Challenges (one line each) | — | Avoiding leakage, since actual times exist only in training (history lagged 21 days to mimic the test gap) · a monsoon-heavy test window · Style/Tech's heavy-tailed service times · a small median bias from the log-target model, fixed with fold-validated calibration. |

Upload as an **unlisted** YouTube video.
