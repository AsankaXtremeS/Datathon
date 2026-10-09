# AI Tool Disclosure – Datathon (Tasks 1, 2A and 2B)

**Team HackBots · Tech-Triathlon 2026**

> **TEAM: complete the three `[TEAM: …]` items and delete this line before submitting.**

## Tools used

| Tool | Used for |
|---|---|
| Claude Code (Anthropic), an AI coding assistant in VS Code | Development assistant: exploratory analysis, writing pipeline and notebook code, running experiments, drafting documentation |
| [TEAM: list any other tools, e.g. ChatGPT, Copilot, or "none"] | |

No AI tool or pretrained model is part of the **submitted models or allocation**. Task 1 uses scikit-learn `HistGradientBoosting` estimators, and Task 2A uses scikit-learn ridge regressions with small `HistGradientBoosting` residual models. Every model was trained from scratch on the competition data. Task 2B uses no model: the allocation comes from our own branch-and-bound and packing code. We used no proprietary model APIs and no AutoML or low-code modelling tools, as the competition rules require.

## AI-assisted work

| Area | How the AI tool was used |
|---|---|
| Task 1 – data audit | Generated the audit checks (join integrity, time arithmetic, outliers, train/test shift) and summarised the findings |
| Task 1 – labels | Proposed the label definitions and checked them against the data (wait-time correlation, boundary sensitivity) |
| Task 1 – features | Wrote the feature pipeline, the route simulation and past-only history features, and ran the leakage and adversarial checks |
| Task 1 – validation and modelling | Designed the forward-in-time folds, ran the experiments (losses, tuning, stacking, bagging, model-family comparison, calibration) and wrote the final pipeline |
| Task 2A and 2B | [TEAM: describe how AI tools were or were not used to build the Task 2A forecaster and the Task 2B solver] |
| Final notebook | Merged the team's Task 2A and 2B code into the final notebook and added the inference cell for both models |
| Task 2B policy text | Checked the generated policy against the data and corrected its explanation of the chilled deferrals; the allocation itself is unchanged |
| Documentation | Drafted this disclosure, the Task 1 preprocessing document, the Task 1 architecture diagram, the README and the notebook explanations |

## Human work and decisions

- Framed the problem and split Task 1 into phases, then reviewed each phase's results before approving the next.
- Directed the approach: asked for a model-family comparison before accepting HistGradientBoosting, for the evaluation notebook with confusion matrices, and for the service-time bias correction.
- Ran the notebooks, checked the outputs and reported errors for fixing.
- Set the project structure and the rule that raw data files are never modified.
- [TEAM: who built Task 2A and Task 2B, and the key decisions you made there]

## How we used the tool

- The team gave instructions phase by phase and reviewed every result before moving on.
- Every claim in the notebooks is backed by code that runs on the data. Numbers come from executed cells, not from AI text.
- Results were validated on held-out, forward-in-time data. Changes were kept only when they improved scores on unseen folds.
- The Task 2B allocation is checked with the organisers' `check_allocation.py`.
