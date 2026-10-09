# Demo Video Outline – Task 2A (Depot Demand Forecast)

The demo video is a single 3–5 minute unlisted YouTube video covering the whole project. This outline is the Task 2A segment, **about 60–75 seconds**. Suggested split: Task 1 about 2:00, Task 2A about 1:05, Task 2B about 1:20, plus a short intro.

Figures below come from `Task2A_Data_Preprocessing.md`.

| Time | Show | Say |
|---|---|---|
| 0:00–0:10 | Booklet Task 2A | "We forecast weekly total and chilled volume for each depot and brand, ten weeks ahead." |
| 0:10–0:25 | `Task2A_Data_Preprocessing.md` §1 | "Every order counts once, including deferred and not-run ones, because they are still demand. Orders are assigned to a week by `order_date`. We build a daily panel of operating days, so short holiday weeks are handled by construction." |
| 0:25–0:45 | `task2a_architecture.png` | "Order counts follow a fixed weekday schedule, so demand moves through order size: paydays, festival run-ups and slow growth. Fresh and Style use a pooled-depot ridge on log daily volume with one ramp per festival, plus a small boosting model on the residuals. Tech uses a strongly regularised ridge. Fresh chilled volume blends a direct model with total × chilled share. Daily forecasts are summed over each week's operating days." |
| 0:45–1:00 | Final notebook backtest table | "On 34 rolling origins with a 10-week horizon, weekly WAPE is 3.2% for Fresh (best baseline 6.0%), 7.6% for Style and 3.5% for chilled. Tech is mostly noise, at about 29% WAPE for every method we tried." |
| 1:00–1:05 | Challenges (one line) | Festival-heavy forecast window (New Year, Vesak, Poson), short weeks, a not-run gap in the test-period history (grossed up from training data), and only two observed New Year seasons. |
