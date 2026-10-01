# Autonomous Data Science Agent

An academic research prototype for a **controlled autonomous machine-learning workflow**. Give the system a tabular CSV file and a natural-language objective; its decision layer turns deterministic tool output into a bounded modelling plan, evaluates results, and proposes the next approved experiment.

The agent is an orchestration and decision layer, not a code-execution environment. It can select from an allowlisted set of profiling, preprocessing, training, and evaluation tools, but it cannot run arbitrary code or promote a model without human approval.

## What the prototype supports

- Binary and multiclass classification
- Regression
- Initial model families: Logistic Regression, Ridge/Linear Regression, Random Forest, and XGBoost
- Iterative feedback loop: **analyse → decide → train → evaluate → improve**
- Bounded experiments, explicit acceptance criteria, and stopping conditions
- Local JSON experiment tracking by default, with optional MLflow tracking
- FastAPI serving, Streamlit dashboard, Docker deployment, DVC pipeline/data versioning, and optional Evidently reporting

## Design at a glance

```text
CSV + objective
      │
      ▼
Deterministic tools: profile → preprocess → train → evaluate
      │ structured, validated results
      ▼
Agent: identify task/target → plan allowlisted experiment → assess result
      │
      ├─ acceptance criteria met ──► candidate model → human approval → promoted model
      │
      └─ criteria not met ─────────► next bounded improvement experiment / stop
```

Monitoring is deliberately conservative: it produces a drift report and a **retraining proposal**. It does not replace the deployed model or start uncontrolled production retraining.

## Planned layout

```text
src/autods/           # installable Python package
  api.py               # FastAPI app (`autods.api:app`)
  cli.py               # controlled CLI (`python -m autods.cli`)
  agent.py             # decision/orchestration layer
  profiling.py         # deterministic dataset profiling
  preprocessing.py     # deterministic preparation and split logic
  training.py          # allowlisted model training/evaluation
  orchestrator.py      # bounded experiment loop and promotion policy
  monitoring.py        # drift checks and retraining proposals
streamlit_app.py       # Streamlit dashboard
data/raw/              # input CSVs (normally DVC-managed)
artifacts/             # local plans, reports, and JSON experiment log
models/                # local candidate/promoted model files
tests/
```

## Quick start (local development)

Prerequisites: Python 3.10+ (3.11 recommended) and a CSV dataset. Install the optional extras used by the complete local prototype:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dashboard,mlops,boosting]"
```

Start the API in one terminal:

```powershell
python -m uvicorn autods.api:app --app-dir src --reload --host 127.0.0.1 --port 8000
```

Start the dashboard in a second terminal:

```powershell
python -m streamlit run streamlit_app.py
```

Open the dashboard URL printed by Streamlit and upload a CSV there. The dashboard runs the same bounded local workflow and saves its uploaded file under `datasets/`. The API supports an equivalent CSV upload and controlled-run workflow; its interactive contract is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

The API upload endpoint returns a server-generated dataset identifier; use that identifier in `/runs` and `/monitor/drift` rather than a filesystem path. API promotion additionally requires `AUTODS_APPROVAL_TOKEN` to be configured and supplied in the `X-Approval-Token` header. The local dashboard is intentionally a single-user demonstration surface.

Use an objective that expresses the intended outcome and, where useful, the acceptance threshold—for example: `Predict customer churn and prefer macro F1 of at least 0.80.` The system records the inferred task, target, metric, candidate plan, results, and stop reason.

For a scriptable run, use the same controlled workflow through the CLI:

```powershell
python -m autods.cli run --dataset data/raw/dataset.csv --objective "Predict churn with F1 at least 0.75." --target churn --output artifacts/summary.json
```

## Tracking and model promotion

The default tracking backend is local JSON: one readable audit record is persisted per run under `artifacts/runs/<run-id>/run.json`. This keeps the initial demo self-contained.

To mirror a CLI run to MLflow, install the `mlops` extra (included in the command above), start a local SQLite-backed MLflow server in one terminal, then configure the standard MLflow URI in a second terminal and add `--use-mlflow`:

```powershell
$agentPython = 'C:\\Users\\sande\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe' # or python in an approved virtual environment
& $agentPython -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

```powershell
$agentPython = 'C:\\Users\\sande\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe' # or python in an approved virtual environment
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
& $agentPython -m autods.cli run --dataset data/raw/dataset.csv --objective "Predict customer churn" --target churn --use-mlflow
```

MLflow is optional; point `MLFLOW_TRACKING_URI` at an existing tracking service when available. With no URI set, the CLI uses a local SQLite database under `artifacts/mlflow.db`. Regardless of backend, a successful experiment yields a **candidate**, not an automatically deployed model. Review its metrics, acceptance criteria, and artifacts in the dashboard/API, then explicitly approve promotion. Only an approved promoted model is eligible for FastAPI serving.

## Docker

The supplied Compose setup runs the FastAPI API and Streamlit dashboard, persisting uploaded datasets, artifacts, and local models on the host:

```powershell
$env:AUTODS_APPROVAL_TOKEN = "replace-with-a-long-local-secret"
docker compose up --build
```

Compose binds its demo ports to `127.0.0.1` and requires the approval token above; it is not intended as an internet-facing authentication solution. It uses the JSON tracker by default. For an MLflow-tracked run, invoke the controlled CLI with `--use-mlflow` and set `MLFLOW_TRACKING_URI` for that process.

- API: [http://localhost:8000/docs](http://localhost:8000/docs)
- Dashboard: [http://localhost:8501](http://localhost:8501)

## DVC workflow

After initializing Git, initialize DVC once and add the input data:

```powershell
git init
dvc init
dvc add data/raw/dataset.csv
git add .
git commit -m "Initialize controlled AutoDS prototype"
```

`dvc.yaml` contains a reproducible example stage. Edit its dataset path and objective to suit the experiment, then run:

```powershell
dvc repro
```

The stage calls the package CLI rather than arbitrary agent-generated code. Generated run artifacts are intentionally outside the DVC cache; experiment metadata belongs in the configured tracking backend.

## Guardrails

- The agent receives structured tool results and emits validated plans; it does not execute free-form Python.
- Training uses only allowlisted preprocessing steps and model families.
- The loop observes configured experiment budgets, acceptance criteria, and stopping conditions.
- Promotion is a separate, manual approval action.
- Promotion saves a feature-only drift baseline alongside the approved model; drift detection creates a reviewed retraining proposal rather than silently changing production.

This is intentionally a small, explainable prototype—not a general-purpose AutoML platform.
