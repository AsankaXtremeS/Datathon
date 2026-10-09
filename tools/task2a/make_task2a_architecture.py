"""Draw the Task 2A architecture diagram -> HackBots_Datathon_Submision/docs/architecture/task2a_architecture.png

Reflects the production design in task2a_models.py (make_model / Forecaster) and the validation in notebooks 04-05c.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from task2a_common import SUBMISSION_DIR

W, H = 16, 11.4
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, title, body, color, fs=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=color, ec="#333", lw=1.1))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=fs, linespacing=1.4)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#333"))


ax.text(W / 2, H - 0.25, "Task 2A - weekly depot x brand demand forecast (total and chilled m3): architecture", ha="center", fontsize=14.5, fontweight="bold")

# ---- row 1: data -> target -> features
y1, h1 = 8.65, 2.1
box(0.2, y1, 4.6, h1, "1  Raw data", "deliveries_train.csv + task1_test_inputs.csv\n(orders 2024-01-01 .. 2026-03-28,\nincl. deferred and not_run)\ncalendar.csv (operating days, festivals, paydays)\ntask2a_test_inputs.csv (60 rows: ISO weeks 14-23)", "#f4f4f4")
box(5.7, y1, 4.6, h1, "2  Audit and target build  (nb 01-02)", "every order counted once; week by order_date\ndaily OPERATING-day panel per depot x brand\n(orders per day = fixed weekday schedule)\nnot_run gross-up on test-file days (0-2.6%)\nreconciliation asserts vs raw totals", "#fff4cc")
box(11.2, y1, 4.6, h1, "3  Calendar features  (nb 03)", "weekday, payday, holiday, festival ramp (+ square)\nONE RAMP PER FESTIVAL (New Year, Vesak, Poson, ...)\ndays to / since festival, ISO week\nyears since 2024-01 (trend)\nall known in advance: no lagged demand", "#fff4cc")
arrow(4.8, y1 + h1 / 2, 5.7, y1 + h1 / 2)
arrow(10.3, y1 + h1 / 2, 11.2, y1 + h1 / 2)

# ---- row 2: models (mean daily forecast on operating days)
y2, h2 = 5.0, 2.75
ax.text(0.25, y2 + h2 + 0.12, "4  Models  (nb 05)  -  each forecasts MEAN DAILY volume on future operating days", fontsize=10.5, fontweight="bold")
box(0.2, y2, 3.85, h2, "Fresh total", "pooled ridge on log volume,\nboth depots (depot x weekday\nintercepts, depot trend)\n+ boosting on ridge residuals\n(depth 3, 60 trees)\ndamped trend (phi 0.9 / week)\nlognormal-mean back-transform", "#d9ecff", 8.1)
box(4.15, y2, 3.85, h2, "Style total", "same model as Fresh,\nbut NO trend\n(backtest: lower MAE, RMSE\nand WAPE without it)\n\nrun-up weeks carry extra\nuncertainty (week 15)", "#d9ecff", 8.1)
box(8.1, y2, 3.85, h2, "Tech total", "per-depot ridge on volume:\nweekday (depot-specific\nschedule), payday, ramp,\ndamped trend\nstrongly regularised:\nthe series is mostly noise\n(about 30% WAPE)", "#d9ecff", 8.1)
box(12.05, y2, 3.75, h2, "Fresh chilled", "50/50 blend of\n(a) direct model fitted on\n    chilled volume\n(b) total forecast x depot\n    chilled share\ncapped at total forecast\nStyle / Tech chilled = 0", "#d9ecff", 8.1)
ax.text(8.0, y1 - 0.3, "daily panel + calendar features feed every model", ha="center", fontsize=8, color="#555")
for xm in (2.1, 6.05, 10.0, 13.9):
    arrow(xm, y1, xm, y2 + h2 + 0.45)

# ---- row 3: weekly aggregation (full width)
ya, ha = 3.45, 1.15
box(0.2, ya, 15.6, ha, "5  Weekly aggregation", "weekly forecast = SUM of daily forecasts over the OPERATING days of each ISO week in calendar.csv  -  (4-day New Year week 16, 5-day Vesak week 18; Sundays and closed days are never forecast)", "#d9f2d9", 8.4)
for xm in (2.1, 6.05, 10.0, 13.9):
    arrow(xm, y2, xm, ya + ha)

# ---- row 4: validation (left) and outputs (right)
y4, h4 = 0.45, 2.6
box(0.2, y4, 9.3, h4, "Validation loop  (nb 04, 05, 05b, 05c)  -  chooses and stress-tests the design above",
    "34 rolling origins, 10-week horizon, forward in time; fit only on days before the cut; MAE, RMSE, WAPE, bias\n"
    "baselines: last-8 / 26-week mean, seasonal naive\n"
    "Fresh WAPE ladder: no trend 5.2% > damped trend 4.0% > New Year ramp 3.6% > ramp per festival 3.25%\n"
    "  > pooled depots 3.26% > residual boosting 3.18%\n"
    "final: Fresh 3.2%, Style 7.6%, Tech 29.4%, Fresh chilled 3.5% WAPE\n"
    "robustness: flat hyper-parameter plateau; leave-one-New-Year-out ~1%\n"
    "error analysis by horizon, festival weeks, short weeks (under-forecast)\n"
    "tested and rejected: catch-up features, Style run-up correction\n"
    "out-of-time calibrated 80% bands (coverage Style 81%, Tech 82%, Fresh 91%)", "#fde2e2", 7.9)
box(9.8, y4, 6.0, h4, "6  Outputs  (nb 06)", "submission_task2a.csv\n(60 rows, template order)\n\ntask2a_forecast_with_bands.csv\n(80% bands, for planners)\n\ntask2a_models.joblib\n(Forecaster: fit once, predict any\ndepot / brand / ISO week)", "#d9f2d9", 8.3)
arrow(12.8, ya, 12.8, y4 + h4)

ax.text(W / 2, 0.12, "Deployment: Forecaster.fit(panel) once per data refresh (seconds, CPU only, no pretrained models or external APIs); Forecaster.weekly(future_ops, requests) serves any depot / brand / ISO week from the calendar.",
        ha="center", fontsize=8, color="#444")
out = SUBMISSION_DIR / "docs" / "architecture" / "task2a_architecture.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
