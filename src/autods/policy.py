"""Hard guardrails between an agent plan and execution tools."""

from __future__ import annotations

import math
from .schemas import MLPlan, TaskType


class PolicyViolation(ValueError):
    """Raised when a plan asks for something outside the controlled prototype."""


ALLOWED_MODELS: dict[TaskType, tuple[str, ...]] = {
    TaskType.BINARY_CLASSIFICATION: (
        "logistic_regression",
        "random_forest",
        "xgboost",
    ),
    TaskType.MULTICLASS_CLASSIFICATION: (
        "logistic_regression",
        "random_forest",
        "xgboost",
    ),
    TaskType.REGRESSION: (
        "linear_regression",
        "ridge",
        "random_forest",
        "xgboost",
    ),
}

ALLOWED_METRICS: dict[TaskType, tuple[str, ...]] = {
    TaskType.BINARY_CLASSIFICATION: ("f1", "accuracy", "roc_auc"),
    TaskType.MULTICLASS_CLASSIFICATION: ("f1_macro", "accuracy", "roc_auc_ovr"),
    TaskType.REGRESSION: ("r2", "rmse"),
}


def metric_higher_is_better(metric: str) -> bool:
    return metric != "rmse"


def validate_plan(plan: MLPlan) -> None:
    if not plan.target_column.strip():
        raise PolicyViolation("A non-empty target column is required")
    if not 1 <= plan.max_experiments <= 5:
        raise PolicyViolation("max_experiments must be between 1 and 5")
    if not 0.05 <= plan.test_size <= 0.4:
        raise PolicyViolation("test_size must be between 0.05 and 0.4")
    if plan.metric not in ALLOWED_METRICS[plan.task_type]:
        raise PolicyViolation(
            f"Metric '{plan.metric}' is not permitted for {plan.task_type.value}"
        )
    allowed_models = ALLOWED_MODELS[plan.task_type]
    if not plan.candidate_models:
        raise PolicyViolation("At least one allowlisted candidate model is required")
    if len(set(plan.candidate_models)) != len(plan.candidate_models):
        raise PolicyViolation("Candidate models must not be repeated")
    unsupported = set(plan.candidate_models) - set(allowed_models)
    if unsupported:
        raise PolicyViolation(f"Unsupported model request: {sorted(unsupported)}")
    if not math.isfinite(plan.acceptance_threshold):
        raise PolicyViolation("Acceptance threshold must be finite")
    if plan.task_type.is_classification and not 0 <= plan.acceptance_threshold <= 1:
        raise PolicyViolation("Classification acceptance thresholds must be between 0 and 1")
    if plan.metric == "rmse" and plan.acceptance_threshold < 0:
        raise PolicyViolation("RMSE acceptance threshold cannot be negative")
