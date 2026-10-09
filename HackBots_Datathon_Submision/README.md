# HackBots – Tech-Triathlon 2026 Datathon submission

Demo video (unlisted YouTube): **[TEAM: paste link here]**

## Deliverables (booklet p. 22)

| Deliverable | Where |
|---|---|
| Final notebook | `HackBots_FinalNotebook.ipynb`: label construction, preprocessing, training and evaluation for Task 1 and Task 2A, the Task 2B allocation with the official feasibility check, and a final cell that loads the saved models and prints Task 1 and Task 2A inference inputs and predictions |
| Model files | `models/task1_models.joblib`, `models/task2a_models.joblib` |
| Prediction files | `predictions/submission_task1.csv`, `predictions/submission_task2a.csv` (supplied template columns and identifiers; Task 1 keeps the original row order) |
| Peak-day allocation | `predictions/submission_task2b.csv` + written policy `docs/Task2B_policy.md` |
| Architecture diagrams | `docs/architecture/task1_architecture.png`, `task2a_architecture.png`, `task2B_architecture.png` (models, preprocessing pipeline, proposed deployment) |
| Data preprocessing documents | `docs/Task1_Data_Preprocessing.md`, `docs/Task2A_Data_Preprocessing.md`, `docs/Task2B_Data_Preprocessing.md` |
| AI tool disclosure | `docs/AI_Tool_Disclosure.md` |
| Demo video | Unlisted YouTube link at the top of this README |
| Extra | `predictions/task2a_forecast_with_bands.csv`: Task 2A forecast with 80% bands |

## Running the notebook
Python 3 with pandas, numpy, scikit-learn and joblib. Open the notebook from this folder. It finds the competition data either in `../data/raw/` (our repository layout) or in the booklet folders (`Training Data/`, `Test Data/`, `General Data/`, `Submission Templates/`) placed inside or next to this folder. Data files are not included, per the competition's data-confidentiality terms.

Only scikit-learn models trained from scratch were used: no pre-trained models, proprietary APIs or AutoML tools.
