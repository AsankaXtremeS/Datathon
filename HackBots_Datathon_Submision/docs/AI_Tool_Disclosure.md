# AI Tool Disclosure – Datathon (Task 1)

**Team HackBots · Tech-Triathlon 2026**

> **TEAM: review and complete every `[TEAM: …]` item before submitting.** This disclosure must accurately describe how *your team* worked.

## Tools used

| Tool | Used for |
|---|---|
| Claude Code (Anthropic), an AI coding assistant in VS Code | Development assistant: exploratory analysis, writing pipeline and notebook code, running experiments, drafting documentation |
| [TEAM: list any other tools, e.g. ChatGPT, Copilot, or "none"] | |

No AI tool or pretrained model is part of the **submitted predictive models**. Every model is a scikit-learn `HistGradientBoosting` estimator trained from scratch on the competition data. We used no proprietary model APIs and no AutoML or low-code modelling tools, as the competition rules require.

## AI-assisted work

| Area | How the AI tool was used |
|---|---|
| Data audit | Generated the audit checks (join integrity, time arithmetic, outliers, train/test shift) and summarised the findings |
| Label construction | Proposed and verified the label definitions with data evidence (wait-time correlation, boundary sensitivity) |
| Feature engineering | Wrote the feature pipeline, the route simulation and past-only history features, and ran the leakage and adversarial checks |
| Validation and modelling | Designed the forward-in-time folds, ran the experiments (losses, tuning, stacking, bagging, model-family comparison, calibration) and wrote the final pipeline |
| Documentation | Drafted this disclosure, the preprocessing document, the architecture diagram and the notebook explanations |

## Human work and decisions

[TEAM: describe honestly what the team did. Examples:]
- [TEAM: problem framing and the plan to split Task 1 into phases]
- [TEAM: reviewing each phase's outputs and approving or redirecting the approach, e.g. asking for the model comparison, adding confusion matrices, asking for the bias correction]
- [TEAM: running the notebooks and checking the results]
- [TEAM: decisions such as keeping the long-tail service times, using a strict `>` for lateness, and the folder structure]
- [TEAM: anything you wrote, changed or verified yourselves]

## How we used the tool

- The team gave instructions phase by phase and reviewed every result before moving on.
- Every claim in the notebooks is backed by code that runs on the data. Numbers come from executed cells, not from AI text.
- Results were validated on held-out, forward-in-time data. Corrections were kept only when they improved scores on unseen folds.
- [TEAM: add how you checked AI-generated code or conclusions]
