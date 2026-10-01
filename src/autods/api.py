"""Minimal FastAPI serving surface for approved models and controlled runs."""

from __future__ import annotations

import hmac
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from .monitoring import DriftBaseline, compare_to_baseline, retraining_request
from .orchestrator import AutonomousDataScienceOrchestrator
from .registry import LocalModelRegistry, RegistryError
from .schemas import RunConfig
from .tracking import LocalExperimentTracker

APP_ROOT = Path(os.environ.get("AUTODS_HOME", ".")).resolve()
DATASET_ROOT = Path(os.environ.get("AUTODS_DATASET_DIR", APP_ROOT / "datasets")).resolve()
ARTIFACT_ROOT = Path(os.environ.get("AUTODS_ARTIFACT_DIR", APP_ROOT / "artifacts")).resolve()
MODEL_ROOT = Path(os.environ.get("AUTODS_MODEL_DIR", APP_ROOT / "models")).resolve()
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = FastAPI(
    title="Autonomous Data Science Agent",
    version="0.1.0",
    description="Controlled tabular ML experiments; promotion always needs human approval.",
)


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunRequest(StrictRequest):
    dataset_name: str
    objective: str = Field(min_length=3, max_length=2_000)
    target_column: str | None = Field(default=None, max_length=255)
    acceptance_threshold: float | None = Field(default=None, ge=0)
    max_experiments: int = Field(default=3, ge=1, le=5)


class ApprovalRequest(StrictRequest):
    approved_by: str = Field(min_length=1, max_length=200)


class PredictionRequest(StrictRequest):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)
    include_probabilities: bool = False


class MonitoringRequest(StrictRequest):
    current_dataset_name: str
    drift_threshold: float = Field(default=0.20, gt=0, le=1)


def _safe_dataset_path(name: str) -> Path:
    # Dataset names, unlike generic paths, prevent reading arbitrary server files.
    if not re.fullmatch(r"[a-f0-9]{32}\.csv", name):
        raise HTTPException(status_code=400, detail="dataset_name must be an API-issued dataset identifier")
    path = (DATASET_ROOT / name).resolve()
    if path.parent != DATASET_ROOT.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="Uploaded dataset was not found")
    return path


def _require_approval_token(provided_token: str | None) -> None:
    expected_token = os.environ.get("AUTODS_APPROVAL_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="API promotion is disabled until AUTODS_APPROVAL_TOKEN is configured",
        )
    if not provided_token or not hmac.compare_digest(provided_token, expected_token):
        raise HTTPException(status_code=403, detail="A valid approval token is required for promotion")


def _orchestrator() -> AutonomousDataScienceOrchestrator:
    return AutonomousDataScienceOrchestrator(artifact_root=ARTIFACT_ROOT, model_root=MODEL_ROOT)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "autonomous-data-science-agent"}


@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, str]:
    filename = Path(file.filename or "dataset.csv").name
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are accepted")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Dataset exceeds the 200 MB prototype limit")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_name = f"{uuid.uuid4().hex}.csv"
    destination = DATASET_ROOT / dataset_name
    staging = DATASET_ROOT / f".{dataset_name}.upload"
    try:
        staging.write_bytes(content)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            staging.unlink()
    return {
        "dataset_name": dataset_name,
        "message": "Dataset uploaded as an immutable identifier; it has not been executed as code.",
    }


@app.post("/runs")
def start_run(request: RunRequest) -> dict[str, Any]:
    dataset_path = _safe_dataset_path(request.dataset_name)
    summary = _orchestrator().run(
        dataset_path=dataset_path,
        objective=request.objective,
        config=RunConfig(
            max_experiments=request.max_experiments,
            target_column=request.target_column,
            acceptance_threshold=request.acceptance_threshold,
            use_mlflow=os.environ.get("AUTODS_TRACKING_BACKEND", "json").lower() == "mlflow",
        ),
    )
    return summary.to_dict()


@app.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    """List all completed runs from the local tracker."""
    runs_dir = ARTIFACT_ROOT / "runs"
    if not runs_dir.is_dir():
        return []
    results = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        run_file = run_dir / "run.json"
        if run_file.is_file():
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                results.append({
                    "run_id": data.get("run_id"),
                    "started_at": data.get("started_at"),
                    "completed_at": data.get("completed_at"),
                    "status": data.get("status"),
                    "objective": data.get("objective"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    return results


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise HTTPException(status_code=400, detail="Invalid run identifier")
    try:
        return LocalExperimentTracker(ARTIFACT_ROOT / "runs").get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/models/production")
def production_model_metadata() -> dict[str, Any]:
    try:
        return LocalModelRegistry(MODEL_ROOT).production_metadata()
    except RegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/models/pending")
def list_pending_proposals() -> list[dict[str, Any]]:
    """List all pending model proposals awaiting human approval."""
    registry = LocalModelRegistry(MODEL_ROOT)
    pending_dir = registry.pending_root
    if not pending_dir.is_dir():
        return []
    results = []
    for proposal_dir in sorted(pending_dir.iterdir(), reverse=True):
        metadata_file = proposal_dir / "proposal.json"
        if metadata_file.is_file():
            try:
                data = json.loads(metadata_file.read_text(encoding="utf-8"))
                results.append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return results


@app.post("/promotions/{proposal_id}/approve")
def approve_promotion(
    proposal_id: str,
    request: ApprovalRequest,
    x_approval_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_approval_token(x_approval_token)
    try:
        return LocalModelRegistry(MODEL_ROOT).approve(proposal_id, request.approved_by)
    except RegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        model = LocalModelRegistry(MODEL_ROOT).load_production()
    except RegistryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response: dict[str, Any] = {"predictions": model.predict(request.records)}
    if request.include_probabilities:
        response["probabilities"] = model.predict_proba(request.records)
    return response


@app.post("/monitor/drift")
def monitor_drift(request: MonitoringRequest) -> dict[str, Any]:
    current = _safe_dataset_path(request.current_dataset_name)
    registry = LocalModelRegistry(MODEL_ROOT)
    try:
        baseline = DriftBaseline(**registry.production_baseline())
        source_run_id = registry.production_metadata()["metadata"].get("run_id")
    except (RegistryError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    report = compare_to_baseline(baseline, current, threshold=request.drift_threshold)
    payload = {
        "report": report.to_dict(),
        "retraining_request": retraining_request(report, source_run_id),
    }
    report_path = ARTIFACT_ROOT / "monitoring" / f"{uuid.uuid4().hex}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload


class RetrainRequest(StrictRequest):
    dataset_name: str
    source_run_id: str | None = None
    objective: str = Field(min_length=3, max_length=2_000)
    target_column: str | None = Field(default=None, max_length=255)
    acceptance_threshold: float | None = Field(default=None, ge=0)
    max_experiments: int = Field(default=3, ge=1, le=5)


@app.post("/retrain")
def controlled_retrain(
    request: RetrainRequest,
    x_approval_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Run a controlled retraining experiment using new data.
    
    This does NOT automatically replace the production model.
    The result is a new run that may produce a pending proposal for human review.
    """
    _require_approval_token(x_approval_token)
    dataset_path = _safe_dataset_path(request.dataset_name)
    summary = _orchestrator().run(
        dataset_path=dataset_path,
        objective=request.objective,
        config=RunConfig(
            max_experiments=request.max_experiments,
            target_column=request.target_column,
            acceptance_threshold=request.acceptance_threshold,
        ),
    )
    result = summary.to_dict()
    result["source_run_id"] = request.source_run_id
    result["retraining"] = True
    return result
