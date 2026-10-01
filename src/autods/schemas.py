"""Structured contracts passed between the decision agent and deterministic tools.

The contracts intentionally contain data and allowlisted names only.  They never
carry Python expressions, shell commands, or arbitrary hyperparameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        return self in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }


class RunStatus(str, Enum):
    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    inferred_type: str
    pandas_dtype: str
    missing_fraction: float
    unique_count: int
    unique_fraction: float
    likely_identifier: bool = False


@dataclass(frozen=True)
class DatasetProfile:
    dataset_path: str
    data_hash: str
    rows: int
    columns: list[ColumnProfile]
    likely_targets: list[str]

    def column(self, name: str) -> ColumnProfile:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"Column '{name}' is not present in the profile")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessingSpec:
    numeric_imputation: str = "median"
    categorical_imputation: str = "most_frequent"
    scale_numeric: bool = True
    one_hot_encode: bool = True
    excluded_columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["excluded_columns"] = list(self.excluded_columns)
        return result


@dataclass(frozen=True)
class MLPlan:
    objective: str
    target_column: str
    task_type: TaskType
    metric: str
    acceptance_threshold: float
    candidate_models: tuple[str, ...]
    preprocessing: PreprocessingSpec
    max_experiments: int = 3
    test_size: float = 0.2
    random_state: int = 42
    reasoning: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["task_type"] = self.task_type.value
        result["candidate_models"] = list(self.candidate_models)
        result["reasoning"] = list(self.reasoning)
        return result


@dataclass(frozen=True)
class AgentDecision:
    action: str
    model_name: str | None
    strategy: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    experiment_id: str
    model_name: str
    metric: str
    score: float | None
    metrics: dict[str, float]
    status: str
    elapsed_seconds: float
    strategy: str
    error: str | None = None
    model_bundle: Any | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        # Do not feed a fitted sklearn pipeline to ``asdict``: it can be large,
        # non-JSON-serialisable, and belongs in the controlled model artifact.
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "metric": self.metric,
            "score": self.score,
            "metrics": self.metrics,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "strategy": self.strategy,
            "error": self.error,
        }


@dataclass(frozen=True)
class RunConfig:
    """Bounded controls supplied by the operator, never by generated agent text."""

    max_experiments: int = 3
    target_column: str | None = None
    acceptance_threshold: float | None = None
    random_state: int = 42
    test_size: float = 0.2
    use_mlflow: bool = False


@dataclass
class RunSummary:
    run_id: str
    status: RunStatus
    profile: DatasetProfile
    plan: MLPlan | None
    experiments: list[ExperimentResult]
    decision_trace: list[dict[str, Any]]
    selected_experiment_id: str | None
    final_evaluation: ExperimentResult | None
    promotion_proposal_id: str | None
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "profile": self.profile.to_dict(),
            "plan": self.plan.to_dict() if self.plan else None,
            "experiments": [experiment.to_dict() for experiment in self.experiments],
            "decision_trace": self.decision_trace,
            "selected_experiment_id": self.selected_experiment_id,
            "final_evaluation": self.final_evaluation.to_dict() if self.final_evaluation else None,
            "promotion_proposal_id": self.promotion_proposal_id,
            "stop_reason": self.stop_reason,
        }
