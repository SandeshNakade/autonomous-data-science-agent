"""Auditable experiment tracking with a local default and optional MLflow mirror."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import DatasetProfile, ExperimentResult, MLPlan, RunSummary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


class LocalExperimentTracker:
    """Writes one readable JSON record per autonomous run.

    This makes the prototype runnable without a tracking server while retaining
    the same audit artefacts that can be mirrored to MLflow later.
    """

    def __init__(self, root: str | Path = "artifacts/runs") -> None:
        self.root = Path(root)
        self._runs: dict[str, dict[str, Any]] = {}

    def start_run(self, objective: str, profile: DatasetProfile) -> str:
        run_id = uuid.uuid4().hex
        record = {
            "run_id": run_id,
            "started_at": _utc_now(),
            "objective": objective,
            "dataset": {
                "path": profile.dataset_path,
                "sha256": profile.data_hash,
                "rows": profile.rows,
                "columns": len(profile.columns),
            },
            "plan": None,
            "events": [],
            "experiments": [],
            "final_evaluation": None,
            "status": "running",
        }
        self._runs[run_id] = record
        self._persist(run_id)
        return run_id

    def log_plan(self, run_id: str, plan: MLPlan) -> None:
        self._record(run_id)["plan"] = plan.to_dict()
        self._persist(run_id)

    def log_decision(self, run_id: str, decision: dict[str, Any]) -> None:
        self._record(run_id)["events"].append({"at": _utc_now(), "kind": "decision", **decision})
        self._persist(run_id)

    def log_experiment(self, run_id: str, result: ExperimentResult) -> None:
        self._record(run_id)["experiments"].append(result.to_dict())
        self._persist(run_id)

    def log_final_evaluation(self, run_id: str, result: ExperimentResult) -> None:
        self._record(run_id)["final_evaluation"] = result.to_dict()
        self._persist(run_id)

    def complete(self, run_id: str, summary: RunSummary) -> None:
        record = self._record(run_id)
        record["completed_at"] = _utc_now()
        record["status"] = summary.status.value
        record["summary"] = {
            "selected_experiment_id": summary.selected_experiment_id,
            "final_evaluation_id": (
                summary.final_evaluation.experiment_id if summary.final_evaluation else None
            ),
            "promotion_proposal_id": summary.promotion_proposal_id,
            "stop_reason": summary.stop_reason,
        }
        self._persist(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        path = self.root / run_id / "run.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        raise KeyError(f"Unknown run: {run_id}")

    def _record(self, run_id: str) -> dict[str, Any]:
        if run_id not in self._runs:
            raise KeyError(f"Run is not active in this process: {run_id}")
        return self._runs[run_id]

    def _persist(self, run_id: str) -> None:
        _write_json(self.root / run_id / "run.json", self._record(run_id))


class MLflowExperimentTracker(LocalExperimentTracker):
    """Local audit log plus an optional, deliberately thin MLflow integration."""

    def __init__(self, root: str | Path = "artifacts/runs", tracking_uri: str | None = None) -> None:
        super().__init__(root)
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("MLflow requested but not installed; install the 'mlops' extra") from exc
        self.mlflow = mlflow
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)
        self.mlflow.set_experiment("autonomous-data-science-agent")
        self._active_mlflow_runs: dict[str, Any] = {}

    def start_run(self, objective: str, profile: DatasetProfile) -> str:
        run_id = super().start_run(objective, profile)
        active = self.mlflow.start_run(run_name=run_id)
        self._active_mlflow_runs[run_id] = active
        self.mlflow.set_tags(
            {
                "autods.run_id": run_id,
                "autods.dataset_sha256": profile.data_hash,
                "autods.rows": str(profile.rows),
            }
        )
        return run_id

    def log_plan(self, run_id: str, plan: MLPlan) -> None:
        super().log_plan(run_id, plan)
        self.mlflow.log_params(
            {
                "target_column": plan.target_column,
                "task_type": plan.task_type.value,
                "primary_metric": plan.metric,
                "acceptance_threshold": plan.acceptance_threshold,
                "candidate_models": ",".join(plan.candidate_models),
            }
        )

    def log_experiment(self, run_id: str, result: ExperimentResult) -> None:
        super().log_experiment(run_id, result)
        if result.status == "completed":
            self.mlflow.log_metrics(
                {f"{result.experiment_id}.{name}": value for name, value in result.metrics.items()}
            )
        else:
            self.mlflow.set_tag(f"{result.experiment_id}.error", result.error or "unknown")

    def complete(self, run_id: str, summary: RunSummary) -> None:
        super().complete(run_id, summary)
        run_file = self.root / run_id / "run.json"
        self.mlflow.log_artifact(str(run_file))
        self.mlflow.set_tag("autods.status", summary.status.value)
        self.mlflow.end_run()
        self._active_mlflow_runs.pop(run_id, None)


def create_tracker(root: str | Path, use_mlflow: bool) -> LocalExperimentTracker:
    if use_mlflow:
        # MLflow 3.16+ disables its legacy file-store backend by default. A
        # SQLite database keeps the local demo self-contained while an
        # explicitly supplied server URI (for example, http://127.0.0.1:5000)
        # remains the source of truth when one is configured.
        default_database = (Path(root).parent / "mlflow.db").resolve().as_posix()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{default_database}"
        return MLflowExperimentTracker(root, tracking_uri=tracking_uri)
    return LocalExperimentTracker(root)
