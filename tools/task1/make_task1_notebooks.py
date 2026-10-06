"""Generate the Task 1 notebook skeletons (run once; won't overwrite existing notebooks)."""
from pathlib import Path
import nbformat as nbf

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "task1"
SETUP = ('import sys; sys.path.append("../../tools/task1")\n'
         'import numpy as np, pandas as pd\n'
         'import matplotlib.pyplot as plt\n'
         'from task1_common import *\n'
         'pd.set_option("display.max_columns", 60)\n'
         't = load_task1_tables()')

NBS = {
 "01_data_audit": ("Phase 1 - Data audit and cleaning", [
  ("md", "## 1. Load and shape\nRow counts, dtypes, date ranges for train vs test."),
  ("code", "{k: v.shape for k, v in t.items()}"),
  ("md", "## 2. Nulls and duplicates"),
  ("code", "for k in ['deliveries_train','legs_train','orders_test','legs_test']:\n    print(k, t[k].isna().sum()[lambda s: s>0].to_dict(), 'dup ids:', t[k].iloc[:,0].duplicated().sum())"),
  ("md", "## 3. Order <-> leg join integrity\nOne-to-one on (route_id, seq_in_route); planned arrival must agree between the two files."),
  ("code", "tr = join_orders_to_legs(t['deliveries_train'], t['legs_train'])\nte = join_orders_to_legs(t['orders_test'], t['legs_test'])\nprint(len(tr), len(te))\nprint('planned arrival agrees:', (tr.planned_arrival_time == tr.planned_arrival_time_leg).mean(), (te.planned_arrival_time == te.planned_arrival_time_leg).mean())"),
  ("md", "## 4. Time-column sanity\nMinutes conversion, midnight wrap, actual-vs-planned gaps (earlier scan: -113 to +930 min, suspicious)."),
  ("code", "# TODO: convert *_time cols with hhmm_to_min; inspect arrival - planned_arrival, depart gaps, leave - arrival"),
  ("md", "## 5. Outliers\nService-time tail (max 428 min earlier), extreme delays. Decide clip/drop/keep and record why."),
  ("code", "# TODO"),
  ("md", "## 6. Train vs test shift\nCompare brand/district/vehicle/dow/monsoon mix and planned-time distributions; test is 2026-02-16..03-28."),
  ("code", "# TODO"),
  ("md", "## 7. Reference tables\nCoverage of outlets, traffic (district x hour x monsoon), road_conditions (district x date) for every train/test row."),
  ("code", "# TODO"),
  ("md", "## 8. Findings and cleaning decisions\n(record here; feeds the preprocessing document)"),
 ]),
 "02_label_construction": ("Phase 2 - Label construction", [
  ("md", "service_min = leave_outlet_time - max(arrival_time, window_open_time)\n\nlate = arrival_time > window_close_time"),
  ("code", "# TODO: build labels, handle midnight wrap, save data/interim/train_labeled.parquet-or-csv"),
  ("md", "## Label validation\nNo non-positive service times; compare with service_allowance; late rate by brand / mall / dock."),
  ("code", "# TODO"),
 ]),
 "03_feature_engineering": ("Phase 3 - Feature engineering", [
  ("md", "Planning-time features only (no actual times). Historical per-outlet/vehicle stats must use past data only."),
  ("code", "# TODO: order, outlet, route/plan, calendar, traffic, road features; save to data/processed/"),
 ]),
 "04_validation_baselines": ("Phase 4 - Validation design and baselines", [
  ("md", "Forward-in-time split. Baselines: allowance lookup (service), brand base rate (late)."),
  ("code", "# TODO: split, baseline MAE/RMSE, log-loss/Brier/AUC"),
 ]),
 "05_modeling": ("Phase 5 - Modelling", [
  ("md", "scikit-learn only (no pretrained models). Regressor for service time, classifier for lateness, then calibration."),
  ("code", "# TODO: train, tune with time-aware CV, calibrate, save models to HackBots_Datathon_Submision/models"),
 ]),
 "06_predict_submit": ("Phase 6 - Prediction and submission", [
  ("md", "Refit on all training data, predict test, validate and write submission_task1.csv."),
  ("code", "# TODO: predict, validate_submission(sub, t['template']), write to PREDICTIONS"),
 ]),
}

OUT.mkdir(parents=True, exist_ok=True)
for name, (title, cells) in NBS.items():
    p = OUT / f"{name}.ipynb"
    if p.exists():
        print("skip", p.name); continue
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(f"# {title}"), nbf.v4.new_code_cell(SETUP)]
    nb.cells += [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
    nb.metadata["kernelspec"] = {"name": "hackbots-datathon", "display_name": "Python (HackBots Datathon)", "language": "python"}
    nbf.write(nb, p); print("wrote", p.name)
