"""The decision/orchestration layer.

This module deliberately produces declarative plans rather than executable code.
`HeuristicDecisionAgent` makes offline demos reproducible; a production LLM
adapter can implement the same small interface and will still be constrained by
``policy.validate_plan`` before any deterministic tool is called.
"""

from __future__ import annotations

import importlib.util
import math
import re
from typing import Protocol

from .policy import metric_higher_is_better
from .schemas import (
    AgentDecision,
    DatasetProfile,
    ExperimentResult,
    MLPlan,
    PreprocessingSpec,
    RunConfig,
    TaskType,
)


class PlanningError(ValueError):
    """The objective/profile pair cannot be safely turned into a plan."""


class DecisionAgent(Protocol):
    """Contract for a structured LLM or deterministic planning implementation."""

    def create_plan(
        self, profile: DatasetProfile, objective: str, config: RunConfig
    ) -> MLPlan: ...

    def initial_decision(self, plan: MLPlan) -> AgentDecision: ...

    def assess_results(
        self, plan: MLPlan, history: list[ExperimentResult]
    ) -> AgentDecision: ...


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _objective_target(profile: DatasetProfile, objective: str) -> str | None:
    objective_normalised = _normalise(objective)
    matches: list[str] = []
    for column in profile.columns:
        candidate = _normalise(column.name)
        if candidate and re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", objective_normalised):
            matches.append(column.name)
    if len(matches) == 1:
        return matches[0]

    # "target: churn" / "predict the churn label" are common project prompts.
    explicit = re.search(
        r"(?:target|label|outcome)\s*(?:is|:|=)?\s*([a-zA-Z0-9_ -]+)", objective,
        flags=re.IGNORECASE,
    )
    if explicit:
        wanted = _normalise(explicit.group(1)).split()
        for column in profile.columns:
            if _normalise(column.name) in " ".join(wanted):
                return column.name

    if len(profile.likely_targets) == 1:
        return profile.likely_targets[0]
    return None


def _infer_task(profile: DatasetProfile, target_name: str, objective: str) -> TaskType:
    target = profile.column(target_name)
    language = _normalise(objective)
    if any(word in language for word in ("regression", "forecast", "continuous")):
        if target.inferred_type != "numeric":
            raise PlanningError("The objective requests regression but the target is not numeric")
        return TaskType.REGRESSION
    if "classification" in language or "classify" in language:
        return (
            TaskType.BINARY_CLASSIFICATION
            if target.unique_count <= 2
            else TaskType.MULTICLASS_CLASSIFICATION
        )
    if target.inferred_type in {"categorical", "boolean"}:
        return (
            TaskType.BINARY_CLASSIFICATION
            if target.unique_count <= 2
            else TaskType.MULTICLASS_CLASSIFICATION
        )
    # Integer labels with a small number of distinct values are usually classes.
    if target.unique_count <= 2:
        return TaskType.BINARY_CLASSIFICATION
    if target.unique_count <= 20 and any(
        word in language for word in ("category", "segment", "class", "churn", "fraud")
    ):
        return TaskType.MULTICLASS_CLASSIFICATION
    return TaskType.REGRESSION


def _metric_and_default_threshold(task: TaskType, objective: str) -> tuple[str, float]:
    language = _normalise(objective)
    if task == TaskType.BINARY_CLASSIFICATION:
        if "roc" in language or "auc" in language:
            return "roc_auc", 0.75
        if "accuracy" in language:
            return "accuracy", 0.75
        return "f1", 0.75
    if task == TaskType.MULTICLASS_CLASSIFICATION:
        if "roc" in language or "auc" in language:
            return "roc_auc_ovr", 0.75
        if "accuracy" in language:
            return "accuracy", 0.70
        return "f1_macro", 0.70
    if "rmse" in language or "root mean" in language:
        return "rmse", 1.0
    return "r2", 0.60


def _threshold_from_objective(objective: str, metric: str) -> float | None:
    """Extract a plainly stated acceptance threshold, including percentages."""

    patterns: tuple[str, ...] = (
        r"(?:at least|minimum|no less than|above|over|greater than|>=)\s*(?:[a-z_ -]+?\s*)?(\d+(?:\.\d+)?)\s*%?",
        r"(?:f1|accuracy|auc|r2|rmse)\s*(?:of|>=|at least)?\s*(\d+(?:\.\d+)?)\s*%?",
    )
    if metric == "rmse":
        patterns = (
            r"(?:rmse\s*)?(?:at most|no more than|below|under|less than|<=)\s*(\d+(?:\.\d+)?)",
            r"rmse\s*(?:of|=)?\s*(\d+(?:\.\d+)?)",
            *patterns,
        )
    for pattern in patterns:
        found = re.search(pattern, objective, flags=re.IGNORECASE)
        if found:
            value = float(found.group(1))
            return value / 100 if value > 1 and value <= 100 else value
    return None


def _available_candidates(task: TaskType) -> tuple[str, ...]:
    if task.is_classification:
        candidates = ["logistic_regression", "random_forest"]
    else:
        candidates = ["ridge", "random_forest"]
    # XGBoost is an optional dependency.  The plan reports its omission in
    # rationale rather than attempting a package install or arbitrary fallback.
    if importlib.util.find_spec("xgboost") is not None:
        candidates.append("xgboost")
    return tuple(candidates)


def _best(history: list[ExperimentResult], metric: str) -> ExperimentResult | None:
    usable = [
        item
        for item in history
        if item.status == "completed" and item.score is not None and math.isfinite(item.score)
    ]
    if not usable:
        return None
    return (
        max(usable, key=lambda item: item.score)
        if metric_higher_is_better(metric)
        else min(usable, key=lambda item: item.score)
    )


def _meets_threshold(result: ExperimentResult, plan: MLPlan) -> bool:
    if result.score is None or not math.isfinite(result.score):
        return False
    return (
        result.score >= plan.acceptance_threshold
        if metric_higher_is_better(plan.metric)
        else result.score <= plan.acceptance_threshold
    )


class HeuristicDecisionAgent:
    """A transparent offline planner used for the first research prototype.

    It models the same input/output boundary expected from a structured LLM:
    factual profile + natural-language objective in, validated declarative plan
    out.  The orchestrator never trusts this output without a policy check.
    """

    def create_plan(
        self, profile: DatasetProfile, objective: str, config: RunConfig
    ) -> MLPlan:
        if not objective.strip():
            raise PlanningError("A natural-language ML objective is required")
        target_name = config.target_column or _objective_target(profile, objective)
        if target_name is None:
            raise PlanningError(
                "Target column is ambiguous. Mention the column in the objective or pass target_column."
            )
        try:
            target = profile.column(target_name)
        except KeyError as exc:
            raise PlanningError(str(exc)) from exc
        target_tokens = set(re.findall(r"[a-z0-9]+", target_name.lower()))
        if target_tokens & {"id", "uuid", "identifier", "email", "phone", "name"}:
            raise PlanningError("A likely identifier cannot be used as the target")
        if target.missing_fraction > 0.50:
            raise PlanningError("Target has more than 50% missing values")
        if target.unique_count < 2:
            raise PlanningError("Target needs at least two distinct non-null values")

        task = _infer_task(profile, target_name, objective)
        metric, default_threshold = _metric_and_default_threshold(task, objective)
        threshold = config.acceptance_threshold
        if threshold is None:
            threshold = _threshold_from_objective(objective, metric)
        if threshold is None:
            threshold = default_threshold
        candidates = _available_candidates(task)
        excluded = tuple(
            column.name
            for column in profile.columns
            if column.name != target_name and column.likely_identifier
        )
        rationale = [
            f"Selected '{target_name}' as the target from the objective/profile.",
            f"Inferred {task.value} from target type and objective wording.",
            f"Will optimise {metric} against an explicit threshold of {threshold:g}.",
            "Candidate recipes are fixed allowlisted estimators; no generated code is executable.",
        ]
        if "xgboost" not in candidates:
            rationale.append("XGBoost is not installed, so the optional recipe is unavailable.")
        if excluded:
            rationale.append(f"Excluded likely identifier columns: {', '.join(excluded)}.")
        return MLPlan(
            objective=objective,
            target_column=target_name,
            task_type=task,
            metric=metric,
            acceptance_threshold=float(threshold),
            candidate_models=candidates,
            preprocessing=PreprocessingSpec(excluded_columns=excluded),
            max_experiments=config.max_experiments,
            test_size=config.test_size,
            random_state=config.random_state,
            reasoning=tuple(rationale),
        )

    def initial_decision(self, plan: MLPlan) -> AgentDecision:
        return AgentDecision(
            action="run_experiment",
            model_name=plan.candidate_models[0],
            strategy="interpretable_baseline",
            reason=(
                f"Start with the simplest approved baseline and evaluate {plan.metric} "
                f"against {plan.acceptance_threshold:g}."
            ),
        )

    def assess_results(
        self, plan: MLPlan, history: list[ExperimentResult]
    ) -> AgentDecision:
        best = _best(history, plan.metric)
        if best and _meets_threshold(best, plan):
            return AgentDecision(
                action="accept",
                model_name=best.model_name,
                strategy="quality_gate_met",
                reason=(
                    f"{best.model_name} achieved {plan.metric}={best.score:.4f}, "
                    "meeting the declared acceptance criterion."
                ),
            )
        if len(history) >= plan.max_experiments:
            return AgentDecision(
                action="stop",
                model_name=best.model_name if best else None,
                strategy="experiment_budget_exhausted",
                reason="The bounded experiment budget was exhausted before acceptance.",
            )
        tried = {result.model_name for result in history}
        next_model = next((name for name in plan.candidate_models if name not in tried), None)
        if next_model is None:
            return AgentDecision(
                action="stop",
                model_name=best.model_name if best else None,
                strategy="approved_recipes_exhausted",
                reason="Every approved candidate recipe has been attempted.",
            )
        prior = f"Best so far is {best.score:.4f}." if best and best.score is not None else "No valid result yet."
        return AgentDecision(
            action="run_experiment",
            model_name=next_model,
            strategy="controlled_model_improvement",
            reason=(
                f"{prior} It misses the acceptance criterion, so try the next untried "
                f"approved recipe: {next_model}."
            ),
        )
