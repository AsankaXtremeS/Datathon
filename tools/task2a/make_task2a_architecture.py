"""Draw the Task 2A architecture diagram -> HackBots_Datathon_Submision/docs/architecture/task2a_architecture.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from task2a_common import SUBMISSION_DIR

fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 15); ax.set_ylim(0, 8.2); ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=color, ec="#333", lw=1.1))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top", fontsize=8.3, linespacing=1.35)


def arrow(x1, y1, x2, y2, txt=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#333"))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, txt, ha="center", fontsize=7.5, color="#555")


ax.text(7.5, 7.95, "Task 2A - weekly depot x brand demand forecast: architecture", ha="center", fontsize=14, fontweight="bold")

box(0.2, 5.0, 2.9, 2.5, "1  Raw data", "deliveries_train.csv\ntask1_test_inputs.csv\n(orders to 2026-03-28,\nincl. deferred / not_run)\ncalendar.csv\ntask2a_test_inputs.csv", "#f4f4f4")
box(3.5, 5.0, 3.0, 2.5, "2  Audit + target build", "every order once\nweek by order_date\ndaily OPERATING-day panel\n(depot x brand x day)\nnot_run gross-up on test days\nreconciliation asserts", "#fff4cc")
box(6.9, 5.0, 3.0, 2.5, "3  Calendar features", "weekday, payday, holiday\nfestival ramp + ramp per festival\ndays to / since festival\nISO week, years (trend)\n(all known in advance:\nno lagged demand)", "#fff4cc")
box(10.3, 5.0, 4.5, 2.5, "4  Models (daily mean forecast)", "Fresh, Style: pooled-depot ridge on log volume\n+ residual gradient boosting, damped trend,\nlognormal-mean back-transform\nTech: per-depot ridge (weekday, payday, ramp, trend)\nChilled (Fresh) = total x depot chilled share\nStyle / Tech chilled = 0", "#d9ecff")
arrow(3.1, 6.25, 3.5, 6.25); arrow(6.5, 6.25, 6.9, 6.25); arrow(9.9, 6.25, 10.3, 6.25)

box(10.3, 2.2, 4.5, 2.2, "5  Weekly aggregation", "sum daily forecasts over the OPERATING days\nof each ISO week in calendar.csv\n(4-day New Year week, 5-day Vesak week)\nonly Mon-Sat operating days are ever forecast", "#d9f2d9")
arrow(12.55, 5.0, 12.55, 4.4)

box(6.6, 0.2, 4.0, 1.9, "6  Outputs", "submission_task2a.csv (60 rows)\ntask2a_forecast_with_bands.csv (80% bands)\ntask2a_models.joblib (Forecaster)", "#d9f2d9")
arrow(11.2, 2.2, 9.6, 2.1)

box(0.2, 0.2, 5.9, 4.3, "Validation loop (notebooks 04, 05, 05b, 05c)", "34 rolling origins, 10-week horizon, forward in time\nfit only on days before the cut\nscore weekly totals: MAE, RMSE, WAPE, bias\n\nbaselines: last-8/26-week mean, seasonal naive\nablation ladder: trend -> New Year ramp -> pooling\n-> ramp per festival -> residual boosting\nrobustness: hyper-parameter plateau, trend choice,\nleave-one-New-Year-out, model-variant spread\nerror analysis by horizon / festival / short weeks\nout-of-time calibrated 80% forecast bands", "#fde2e2")
arrow(6.1, 3.2, 10.3, 5.2, "choose design")

ax.text(7.5, 0.0, "Deployment: Forecaster.fit(panel) once per refresh (seconds, CPU only); Forecaster.weekly(future_ops, requests) serves any depot / brand / ISO week from the calendar.",
        ha="center", fontsize=8.3, color="#444")
out = SUBMISSION_DIR / "docs" / "architecture" / "task2a_architecture.png"
plt.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
