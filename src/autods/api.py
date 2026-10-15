"""Minimal FastAPI serving surface for approved models and controlled runs."""

from __future__ import annotations

import hmac
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from .agent import LLMConfigurationError, LLMResponseError, create_decision_agent
from .monitoring import DriftBaseline, compare_to_baseline, retraining_request
from .orchestrator import AutonomousDataScienceOrchestrator
from .planning import PlanningError, PolicyViolation, create_controlled_plan
from .registry import LocalModelRegistry, RegistryError
from .schemas import RunConfig
from .tracking import LocalExperimentTracker
from .auth import (
    current_user,
    list_workspace_audit_events,
    require_permission,
    router as auth_router,
)
from .datasets import DatasetLibrary
from .profiling import DatasetError

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
app.include_router(auth_router)


@app.middleware("http")
async def workspace_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if not request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-store"
    return response


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunRequest(StrictRequest):
    dataset_name: str
    objective: str = Field(min_length=3, max_length=2_000)
    target_column: str | None = Field(default=None, max_length=255)
    acceptance_threshold: float | None = Field(default=None, ge=0)
    max_experiments: int = Field(default=3, ge=1, le=5)


class PlanPreviewRequest(StrictRequest):
    """Operator-controlled inputs for a plan-only agent recommendation."""

    dataset_name: str
    objective: str = Field(min_length=3, max_length=2_000)
    target_column: str | None = Field(default=None, max_length=255)
    acceptance_threshold: float | None = Field(default=None, ge=0)
    max_experiments: int = Field(default=3, ge=1, le=5)


class ApprovalRequest(StrictRequest):
    approved_by: str = Field(min_length=1, max_length=200)


class ReviewRequest(StrictRequest):
    """A reviewer decision that deliberately does not deploy a candidate."""

    decision: Literal["approve", "reject"]
    reason: str = Field(min_length=3, max_length=1_200)


class DeploymentRequest(StrictRequest):
    """Administrator release note for an already reviewer-approved candidate."""

    deployment_note: str = Field(min_length=3, max_length=1_200)


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


def _monitoring_report_path(report_id: str) -> Path:
    """Resolve a persisted monitoring report without accepting a filesystem path."""

    if not re.fullmatch(r"[a-f0-9]{32}", report_id):
        raise HTTPException(status_code=400, detail="Invalid monitoring report identifier")
    root = (ARTIFACT_ROOT / "monitoring").resolve()
    path = (root / f"{report_id}.json").resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status_code=404, detail="Monitoring report was not found")
    return path


def _monitoring_report_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Expose a small, display-ready history row rather than a report file path."""

    report = payload.get("report")
    retraining = payload.get("retraining_request")
    report_id = payload.get("report_id")
    if (
        not isinstance(report, dict)
        or not isinstance(retraining, dict)
        or not isinstance(report_id, str)
        or not re.fullmatch(r"[a-f0-9]{32}", report_id)
    ):
        return None
    drifted = report.get("drifted_columns", [])
    return {
        "report_id": report_id,
        "created_at": payload.get("created_at"),
        "dataset_name": payload.get("dataset_name"),
        "monitored_by": payload.get("monitored_by"),
        "model_version": payload.get("model_version"),
        "monitoring_scope": payload.get("monitoring_scope", "feature_drift"),
        "drift_detected": bool(report.get("drift_detected")),
        "drifted_columns": drifted if isinstance(drifted, list) else [],
        "missing_columns": report.get("missing_columns", []),
        "threshold": report.get("threshold"),
        "retraining_state": retraining.get("state"),
    }


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
async def upload_dataset(
    file: UploadFile = File(...),
    user: dict = Depends(require_permission("datasets:write")),
) -> dict[str, Any]:
    filename = Path(file.filename or "dataset.csv").name
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are accepted")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Dataset exceeds the 200 MB prototype limit")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")
    try:
        report = await run_in_threadpool(DatasetLibrary(DATASET_ROOT).add, content, filename, user["email"])
    except DatasetError as exc:
        raise HTTPException(400, detail=str(exc)[:600]) from exc
    return {**report, "message": "Dataset validated and saved unchanged. No training has started."}


@app.get("/datasets", dependencies=[Depends(current_user)])
def list_datasets(
    search: str = Query(default="", max_length=150),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict[str, Any]:
    return DatasetLibrary(DATASET_ROOT).list(search, offset, limit)


@app.get("/datasets/{dataset_name}", dependencies=[Depends(current_user)])
def inspect_dataset(dataset_name: str) -> dict[str, Any]:
    _safe_dataset_path(dataset_name)
    try:
        return DatasetLibrary(DATASET_ROOT).inspect(dataset_name)
    except DatasetError as exc:
        raise HTTPException(400, detail=str(exc)[:600]) from exc


@app.get("/datasets/{dataset_name}/preview", dependencies=[Depends(current_user)])
def preview_dataset(
    dataset_name: str,
    offset: int = Query(default=0, ge=0, le=1_000_000),
    limit: int = Query(default=20, ge=1, le=50),
) -> dict[str, Any]:
    _safe_dataset_path(dataset_name)
    try:
        return DatasetLibrary(DATASET_ROOT).preview(dataset_name, offset, limit)
    except DatasetError as exc:
        raise HTTPException(400, detail=str(exc)[:600]) from exc


@app.post("/runs", dependencies=[Depends(require_permission("runs:write"))])
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


@app.post("/plans/preview", dependencies=[Depends(require_permission("runs:write"))])
def preview_plan(request: PlanPreviewRequest) -> dict[str, Any]:
    """Generate a reviewable agent plan without training, tracking, or writing a run."""
    dataset_path = _safe_dataset_path(request.dataset_name)
    try:
        agent = create_decision_agent()
        profile, plan, decision = create_controlled_plan(
            dataset_path,
            request.objective,
            RunConfig(
                max_experiments=request.max_experiments,
                target_column=request.target_column,
                acceptance_threshold=request.acceptance_threshold,
            ),
            agent,
        )
    except (DatasetError, PlanningError, PolicyViolation, LLMConfigurationError, LLMResponseError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:1_200]) from exc
    return {
        "dataset": {
            "dataset_name": request.dataset_name,
            "rows": profile.rows,
            "columns": len(profile.columns),
            "data_hash": profile.data_hash,
        },
        "plan": plan.to_dict(),
        "initial_decision": decision.to_dict(),
        "planning_adapter": "openai_structured" if agent.__class__.__name__ == "OpenAILLMDecisionAgent" else "heuristic",
        "guardrails": [
            "This request created a declarative plan only. No preprocessing, training, or experiment tracking was started.",
            "Metrics and model recipes are checked against the fixed policy allowlist before they are returned.",
            f"Any later workflow is bounded to at most {plan.max_experiments} experiments.",
            "A successful result remains a candidate and still requires human approval before promotion.",
        ],
    }


@app.get("/runs", dependencies=[Depends(current_user)])
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
    return sorted(results, key=lambda run: run.get("started_at") or "", reverse=True)


@app.get("/runs/{run_id}", dependencies=[Depends(current_user)])
def get_run(run_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise HTTPException(status_code=400, detail="Invalid run identifier")
    try:
        return LocalExperimentTracker(ARTIFACT_ROOT / "runs").get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/models/production", dependencies=[Depends(current_user)])
def production_model_metadata() -> dict[str, Any]:
    try:
        return LocalModelRegistry(MODEL_ROOT).production_metadata()
    except RegistryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/models/pending", dependencies=[Depends(current_user)])
def list_pending_proposals() -> list[dict[str, Any]]:
    """List all pending model proposals awaiting human approval."""
    registry = LocalModelRegistry(MODEL_ROOT)
    return [
        proposal
        for proposal in registry.list_proposals()
        if proposal.get("status") == "pending_human_approval"
    ]


@app.get("/models/proposals", dependencies=[Depends(current_user)])
def list_model_proposals() -> list[dict[str, Any]]:
    """Read proposal evidence and its audit events without exposing model bytes."""

    return LocalModelRegistry(MODEL_ROOT).list_proposals()


@app.post("/model-reviews/{proposal_id}")
def review_model_proposal(
    proposal_id: str,
    request: ReviewRequest,
    user: dict = Depends(require_permission("reviews:write")),
) -> dict[str, Any]:
    """Record reviewer feedback; approval makes a candidate deployable, not deployed."""

    try:
        return LocalModelRegistry(MODEL_ROOT).review(
            proposal_id=proposal_id,
            decision=request.decision,
            reviewed_by=user["email"],
            reason=request.reason,
        )
    except RegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/deployments", dependencies=[Depends(current_user)])
def deployment_status() -> dict[str, Any]:
    """Read local endpoint and retained release metadata without making a prediction."""

    return LocalModelRegistry(MODEL_ROOT).deployment_status()


@app.post("/deployments/{proposal_id}")
def deploy_model_proposal(
    proposal_id: str,
    request: DeploymentRequest,
    user: dict = Depends(require_permission("deployments:write")),
) -> dict[str, Any]:
    """Deploy one candidate after independent reviewer approval and admin action."""

    try:
        return LocalModelRegistry(MODEL_ROOT).deploy(
            proposal_id=proposal_id,
            deployed_by=user["email"],
            deployment_note=request.deployment_note,
        )
    except RegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/promotions/{proposal_id}/approve")
def approve_promotion(
    proposal_id: str,
    request: ApprovalRequest,
    x_approval_token: str | None = Header(default=None),
    user: dict = Depends(require_permission("legacy-promotion:write")),
) -> dict[str, Any]:
    _require_approval_token(x_approval_token)
    try:
        # This legacy operation also deploys, so only an administrator may use
        # it until the separate review/deploy flow is introduced. Never trust
        # an identity supplied in the request body for the approval record.
        return LocalModelRegistry(MODEL_ROOT).approve(proposal_id, user["email"])
    except RegistryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict", dependencies=[Depends(require_permission("predictions:read"))])
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        model = LocalModelRegistry(MODEL_ROOT).load_production()
    except RegistryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response: dict[str, Any] = {"predictions": model.predict(request.records)}
    if request.include_probabilities:
        response["probabilities"] = model.predict_proba(request.records)
    return response


@app.get("/monitoring/reports", dependencies=[Depends(current_user)])
def list_monitoring_reports() -> list[dict[str, Any]]:
    """List safe monitoring-history summaries, newest first."""

    root = ARTIFACT_ROOT / "monitoring"
    if not root.is_dir():
        return []
    reports: list[dict[str, Any]] = []
    for path in root.glob("*.json"):
        if not re.fullmatch(r"[a-f0-9]{32}", path.stem):
            continue
        try:
            summary = _monitoring_report_summary(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError):
            continue
        if summary is not None:
            reports.append(summary)
    return sorted(reports, key=lambda item: str(item.get("created_at") or ""), reverse=True)


@app.get("/monitoring/reports/{report_id}", dependencies=[Depends(current_user)])
def get_monitoring_report(report_id: str) -> dict[str, Any]:
    """Return one persisted monitoring report; callers never receive its path."""

    try:
        payload = json.loads(_monitoring_report_path(report_id).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Monitoring report could not be read") from exc
    if _monitoring_report_summary(payload) is None:
        raise HTTPException(status_code=404, detail="Monitoring report was not found")
    return payload


@app.get("/alerts", dependencies=[Depends(current_user)])
def list_alerts() -> list[dict[str, Any]]:
    """Return unresolved facts from monitoring, workflow, and review records.

    This local prototype deliberately has no email, SMS, push delivery, or alert
    acknowledgement mechanism. The response is a read-only in-app inbox built
    from persisted evidence rather than fabricated operational telemetry.
    """

    alerts: list[dict[str, Any]] = []
    for report in list_monitoring_reports():
        if not report.get("drift_detected"):
            continue
        drifted = report.get("drifted_columns", [])
        columns = ", ".join(str(column) for column in drifted[:4])
        if len(drifted) > 4:
            columns += ", …"
        alerts.append(
            {
                "alert_id": f"drift-{report['report_id']}",
                "source": "monitoring",
                "severity": "warning",
                "state": "open",
                "title": "Feature drift detected",
                "message": (
                    f"{len(drifted)} monitored feature(s) exceeded the drift threshold"
                    + (f": {columns}." if columns else ".")
                ),
                "occurred_at": report.get("created_at"),
                "resource_id": report["report_id"],
                "destination": "#monitoring",
            }
        )
    for run in list_runs():
        if run.get("status") != "failed":
            continue
        alerts.append(
            {
                "alert_id": f"run-{run['run_id']}",
                "source": "workflow",
                "severity": "critical",
                "state": "open",
                "title": "Controlled workflow failed",
                "message": run.get("objective") or "A saved workflow failed before completion.",
                "occurred_at": run.get("started_at"),
                "resource_id": run["run_id"],
                "destination": "#experiment-history",
            }
        )
    for proposal in LocalModelRegistry(MODEL_ROOT).list_proposals():
        if proposal.get("status") != "pending_human_approval":
            continue
        metadata = proposal.get("metadata", {})
        model_name = str(metadata.get("model_name") or "Model candidate").replace("_", " ")
        alerts.append(
            {
                "alert_id": f"review-{proposal['proposal_id']}",
                "source": "governance",
                "severity": "info",
                "state": "open",
                "title": "Candidate awaiting human review",
                "message": f"{model_name.title()} is awaiting a reviewer decision.",
                "occurred_at": proposal.get("created_at"),
                "resource_id": proposal["proposal_id"],
                "destination": "#approval",
            }
        )
    return sorted(alerts, key=lambda alert: str(alert.get("occurred_at") or ""), reverse=True)


@app.get("/audit-logs", dependencies=[Depends(current_user)])
def list_audit_logs(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=250, ge=1, le=250),
) -> dict[str, Any]:
    """Return a safe, read-only audit view derived from persisted evidence.

    The academic prototype does not yet maintain a separate append-only audit
    database. This endpoint intentionally derives its rows from the governance
    audit events, persisted feature-drift reports, saved workflow records, and
    workspace-administration events; it neither creates audit events nor
    exposes filesystem paths or model bytes.
    """

    events: list[dict[str, Any]] = []
    registry = LocalModelRegistry(MODEL_ROOT)
    for proposal in registry.list_proposals():
        proposal_id = proposal.get("proposal_id")
        if not isinstance(proposal_id, str) or not re.fullmatch(r"[a-f0-9]{32}", proposal_id):
            continue
        metadata = proposal.get("metadata")
        model_name = (
            str(metadata.get("model_name")).replace("_", " ").title()
            if isinstance(metadata, dict) and metadata.get("model_name")
            else "Model candidate"
        )
        audit_events = proposal.get("audit_events", [])
        if isinstance(audit_events, list):
            for index, event in enumerate(audit_events):
                if not isinstance(event, dict):
                    continue
                occurred_at = event.get("at")
                action = event.get("action")
                if not isinstance(occurred_at, str) or not isinstance(action, str):
                    continue
                events.append(
                    {
                        "event_id": f"governance-{proposal_id}-{index}",
                        "source": "governance",
                        "action": action,
                        "occurred_at": occurred_at,
                        "actor": event.get("actor") if isinstance(event.get("actor"), str) else None,
                        "resource_type": "model_candidate",
                        "resource_id": proposal_id,
                        "summary": f"{model_name}: {action.replace('_', ' ')}.",
                        "reason": event.get("reason") if isinstance(event.get("reason"), str) else None,
                        "destination": "#model-registry",
                    }
                )
        # Older combined-promotion records predate governance audit events, but
        # retain an authenticated approver and timestamp in their metadata.
        if (
            proposal.get("status") == "promoted"
            and isinstance(proposal.get("approved_at"), str)
            and isinstance(proposal.get("approved_by"), str)
        ):
            events.append(
                {
                    "event_id": f"legacy-promotion-{proposal_id}",
                    "source": "governance",
                    "action": "legacy_promoted",
                    "occurred_at": proposal["approved_at"],
                    "actor": proposal["approved_by"],
                    "resource_type": "model_candidate",
                    "resource_id": proposal_id,
                    "summary": f"{model_name}: legacy combined promotion recorded.",
                    "reason": None,
                    "destination": "#model-registry",
                }
            )

    for report in list_monitoring_reports():
        report_id = report.get("report_id")
        occurred_at = report.get("created_at")
        if not isinstance(report_id, str) or not re.fullmatch(r"[a-f0-9]{32}", report_id):
            continue
        if not isinstance(occurred_at, str):
            continue
        drifted = report.get("drifted_columns", [])
        drift_count = len(drifted) if isinstance(drifted, list) else 0
        detected = bool(report.get("drift_detected"))
        events.append(
            {
                "event_id": f"monitoring-{report_id}",
                "source": "monitoring",
                "action": "feature_drift_checked",
                "occurred_at": occurred_at,
                "actor": report.get("monitored_by") if isinstance(report.get("monitored_by"), str) else None,
                "resource_type": "monitoring_report",
                "resource_id": report_id,
                "summary": (
                    f"Feature-drift check found {drift_count} drifting feature(s)."
                    if detected
                    else "Feature-drift check completed with no detected drift."
                ),
                "reason": (
                    f"{drift_count} feature(s) exceeded the configured threshold."
                    if detected
                    else "No feature exceeded the configured threshold."
                ),
                "destination": "#monitoring",
            }
        )

    for run in list_runs():
        run_id = run.get("run_id")
        occurred_at = run.get("completed_at") or run.get("started_at")
        if not isinstance(run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", run_id):
            continue
        if not isinstance(occurred_at, str):
            continue
        status = str(run.get("status") or "recorded")
        objective = run.get("objective")
        events.append(
            {
                "event_id": f"workflow-{run_id}",
                "source": "workflow",
                "action": f"workflow_{status}",
                "occurred_at": occurred_at,
                "actor": None,
                "resource_type": "workflow_run",
                "resource_id": run_id,
                "summary": f"Controlled workflow recorded with status: {status.replace('_', ' ')}.",
                "reason": objective if isinstance(objective, str) else None,
                "destination": "#experiment-history",
            }
        )

    for event in list_workspace_audit_events(limit=250):
        event_id = event.get("id")
        occurred_at = event.get("at")
        resource_id = event.get("resource_id")
        action = event.get("action")
        if not all(isinstance(value, str) for value in (event_id, occurred_at, resource_id, action)):
            continue
        if not re.fullmatch(r"[a-f0-9]{32}", event_id) or not re.fullmatch(r"[a-f0-9]{32}", resource_id):
            continue
        events.append(
            {
                "event_id": f"workspace-{event_id}",
                "source": "workspace",
                "action": action,
                "occurred_at": occurred_at,
                "actor": event.get("actor") if isinstance(event.get("actor"), str) else None,
                "resource_type": event.get("resource_type") if isinstance(event.get("resource_type"), str) else "workspace_user",
                "resource_id": resource_id,
                "summary": f"Workspace user record: {action.replace('_', ' ')}.",
                "reason": event.get("reason") if isinstance(event.get("reason"), str) else None,
                "destination": "#settings",
            }
        )

    events.sort(key=lambda event: str(event.get("occurred_at") or ""), reverse=True)
    return {"items": events[offset : offset + limit], "total": len(events)}


@app.post("/monitor/drift")
def monitor_drift(
    request: MonitoringRequest,
    user: dict = Depends(require_permission("monitoring:write")),
) -> dict[str, Any]:
    """Record a feature-drift check; never launch retraining or replace a model."""

    current = _safe_dataset_path(request.current_dataset_name)
    registry = LocalModelRegistry(MODEL_ROOT)
    try:
        baseline = DriftBaseline(**registry.production_baseline())
        production = registry.production_metadata()
        source_run_id = production["metadata"].get("run_id")
    except (RegistryError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    report = compare_to_baseline(baseline, current, threshold=request.drift_threshold)
    report_id = uuid.uuid4().hex
    payload = {
        "report_id": report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": request.current_dataset_name,
        "monitored_by": user["email"],
        "model_version": production.get("deployment", {}).get("version"),
        "monitoring_scope": "feature_drift",
        "report": report.to_dict(),
        "retraining_request": retraining_request(report, source_run_id),
    }
    report_path = ARTIFACT_ROOT / "monitoring" / f"{report_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


class RetrainRequest(StrictRequest):
    dataset_name: str
    source_run_id: str | None = None
    objective: str = Field(min_length=3, max_length=2_000)
    target_column: str | None = Field(default=None, max_length=255)
    acceptance_threshold: float | None = Field(default=None, ge=0)
    max_experiments: int = Field(default=3, ge=1, le=5)


@app.post("/retrain", dependencies=[Depends(require_permission("retraining:write"))])
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


@app.get("/workspace/overview", dependencies=[Depends(current_user)])
def workspace_overview() -> dict[str, Any]:
    """An honest, read-only summary of existing artifacts, not simulated telemetry."""
    runs = list_runs()
    pending = list_pending_proposals()
    try:
        production = LocalModelRegistry(MODEL_ROOT).production_metadata()
    except RegistryError:
        production = None
    return {
        "runs_count": len(runs),
        "review_count": sum(p.get("status") == "pending_human_approval" for p in pending),
        "failed_count": sum(r.get("status") == "failed" for r in runs),
        "production_available": production is not None,
        "recent_runs": runs[:5],
    }


# Built assets are served by the same origin as the cookie-authenticated API.
# Hash navigation avoids swallowing unknown API paths with an SPA fallback.
FRONTEND_ROOT = Path(os.environ.get("AUTODS_FRONTEND_DIR", APP_ROOT / "frontend" / "dist"))
if (FRONTEND_ROOT / "index.html").is_file():
    app.mount("/", StaticFiles(directory=FRONTEND_ROOT, html=True), name="workspace")
