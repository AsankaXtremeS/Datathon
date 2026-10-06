# HackBots – Rootcode Tech-Triathlon 2026 Datathon

Task 1 service time + lateness, Task 2A demand forecast, Task 2B peak-day allocation.

> **Confidential:** the competition datasets must never be committed or shared. `data/raw/`, `data/interim/` and `data/processed/` are git-ignored.

## Team setup guide

### 1. Prerequisites
- Python **3.10 or newer** (check with `python --version`)
- Git
- VS Code with the Python and Jupyter extensions (optional but recommended)

### 2. Clone the repo
```
git clone <repo-url>
cd <repo-folder>
```

### 3. Create the virtual environment and install packages

**Windows (PowerShell)**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and call `.venv\Scripts\python` directly.

### 4. Register the Jupyter kernel (once per machine)
```
python -m ipykernel install --user --name hackbots-datathon --display-name "Python (HackBots Datathon)"
```

### 5. Add the data
The data is not in git. Get the competition CSVs from the team's shared drive and place them like this (17 files, names unchanged):

| Folder | Files |
|---|---|
| `data/raw/train/` | deliveries_train.csv, route_legs_train.csv |
| `data/raw/test/` | task1_test_inputs.csv, route_legs_test.csv, task2a_test_inputs.csv, task2b_peak_day_scenarios.csv, task2b_peak_day_fleet.csv |
| `data/raw/reference/` | outlets.csv, service_allowance.csv, calendar.csv, traffic_speed.csv, road_conditions.csv, district_travel.csv, vehicles.csv |
| `data/raw/templates/` | submission_task1.csv, submission_task2a.csv, submission_task2b.csv |

### 6. Start working
- **JupyterLab:** `jupyter lab`
- **VS Code:** open the folder, open a notebook, and pick the **"Python (HackBots Datathon)"** kernel. The interpreter default is set in `.vscode/settings.json` (Windows path; change it to `.venv/bin/python` on macOS/Linux).

### Conventions
- Use relative paths only (e.g. `data/raw/train/deliveries_train.csv`).
- Put notebooks in the matching `notebooks/taskX/` folder.
- Never edit files in `data/raw/`; write derived data to `data/interim/` or `data/processed/`.
- To add a package: `pip install <pkg>`, then add its pinned version to `requirements.txt`.

## Folder structure

```
data/                    # confidential competition data - never committed
  raw/                   # original CSVs, untouched
    train/               # training deliveries and route legs
    test/                # test inputs and peak-day scenarios/fleet
    reference/           # outlets, calendar, traffic, roads, vehicles, etc.
    templates/           # submission templates
  interim/               # intermediate cleaned data
  processed/             # model-ready feature tables
notebooks/               # exploration and modelling notebooks
  task1/                 # service time + lateness
  task2a/                # demand forecast
  task2b/                # peak-day allocation
tools/                   # helper scripts
reports/                 # analysis outputs
  figures/               # charts
  metrics/               # evaluation metrics
HackBots_Datathon/       # final submission package
  models/                # trained model files
  predictions/           # submission CSVs
  docs/                  # documentation
    architecture/        # architecture diagrams
```

## Deliverables checklist

- [ ] Architecture diagrams
- [ ] Data preprocessing document
- [ ] Model files
- [ ] Final notebook (HackBots_FinalNotebook.ipynb)
- [ ] submission_task2b.csv + prioritization policy
- [ ] submission_task1.csv + submission_task2a.csv
- [ ] Demo video link
- [ ] AI tool disclosure
