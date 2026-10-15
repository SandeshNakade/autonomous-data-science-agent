"""The decision/orchestration layer.

This module deliberately produces declarative plans rather than executable code.
`HeuristicDecisionAgent` makes offline demos reproducible; a production LLM
adapter can implement the same small interface and will still be constrained by
``policy.validate_plan`` before any deterministic tool is called.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
from typing import Any, Protocol

from .policy import ALLOWED_METRICS, metric_higher_is_better
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


class LLMConfigurationError(PlanningError):
    """An LLM-backed agent was requested without a usable configuration."""


class LLMResponseError(PlanningError):
    """An LLM response was unavailable or violated the structured contract."""


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


class StructuredLLMClient(Protocol):
    """Small provider boundary used by the LLM decision agent and its tests."""

    def generate(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        instructions: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]: ...


class OpenAIResponsesStructuredClient:
    """OpenAI Responses API client restricted to strict JSON-schema output.

    The client exposes no tools and receives profile metadata only. It therefore
    cannot ask the runtime to execute code, access a file, or invoke a command.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("OPENAI_API_KEY is required when AUTODS_AGENT_MODE=openai")
        if not model.strip():
            raise LLMConfigurationError("AUTODS_LLM_MODEL cannot be empty")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "OpenAI support is optional. Install it with: pip install -e '.[llm]'"
            ) from exc
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout_seconds,
        )

    def generate(
        self,
        *,
        schema_name: str,
        schema: dict[str, Any],
        instructions: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": json.dumps(payload, sort_keys=True)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
                max_output_tokens=700,
            )
            raw = response.output_text
        except Exception as exc:
            # API errors contain the rejected parameter and a useful explanation.
            # Preserve those fields, but never log the request, headers, or key.
            raise LLMResponseError(
                _safe_openai_error(exc, api_key=getattr(self._client, "api_key", ""))
            ) from None
        if not raw:
            raise LLMResponseError("OpenAI returned no structured output")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMResponseError("OpenAI returned invalid JSON despite the structured schema") from exc
        if not isinstance(value, dict):
            raise LLMResponseError("OpenAI structured output must be a JSON object")
        return value


def _safe_openai_error(exc: Exception, *, api_key: object = "") -> str:
    """Keep actionable provider diagnostics without dumping an SDK exception.

    OpenAI errors usually store the error object in ``body``. Some compatible
    providers wrap it under ``error`` or return plain text. HTML responses expose
    only their title, never a whole proxy page or request dump.
    """

    details = [f"OpenAI structured request failed: {type(exc).__name__}"]
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        details.append(f"HTTP {status}")
    body = getattr(exc, "body", None)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            pass
    explanation_found = False
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            for field in ("code", "param", "message"):
                value = error.get(field)
                if isinstance(value, (str, int)) and str(value).strip():
                    details.append(f"{field}: {value}")
                    explanation_found = True
        elif isinstance(error, str) and error.strip():
            details.append(f"message: {error}")
            explanation_found = True
    if not explanation_found:
        message = body if isinstance(body, str) else getattr(exc, "message", "")
        if isinstance(message, str) and message.strip():
            if re.search(r"<\s*(?:!doctype|html|head|body)\b", message, re.IGNORECASE):
                title = re.search(r"<title[^>]*>(.*?)</title>", message, re.IGNORECASE | re.DOTALL)
                if title:
                    details.append("HTML response title: " + re.sub(r"<[^>]+>", "", title[1]))
                else:
                    details.append("The endpoint returned HTML instead of a JSON API error")
            else:
                details.append(f"message: {message}")
    # Redact before truncating, including a non-standard provider key if supplied.
    message = "; ".join(details)
    if isinstance(api_key, str) and api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"(?i)\bsk-[A-Za-z0-9_.*-]+", "[REDACTED]", message)
    message = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", message)
    return " ".join(message.split())[:1200]


def _bounded_text(value: object, *, limit: int = 360) -> str:
    """Keep provider-supplied rationale readable and bounded in audit artefacts."""

    return str(value).strip().replace("\x00", " ")[:limit]


def _profile_metadata(profile: DatasetProfile) -> dict[str, Any]:
    """The provider sees aggregate schema metadata, never rows or local paths."""

    return {
        "rows": profile.rows,
        "likely_targets": profile.likely_targets,
        "columns": [
            {
                "name": column.name,
                "inferred_type": column.inferred_type,
                "missing_fraction": column.missing_fraction,
                "unique_count": column.unique_count,
                "likely_identifier": column.likely_identifier,
            }
            for column in profile.columns
        ],
    }


def _strict_object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    """Return the strict object shape required by the provider request."""

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


class OpenAILLMDecisionAgent:
    """LLM planner with a narrow declarative contract and deterministic guardrails.

    An LLM can classify the problem, recommend an allowlisted evaluation metric,
    order approved recipes, and react to summary metrics. The orchestrator still
    validates the resulting plan, owns all Python tool calls, enforces the run
    budget, evaluates the final holdout, and requires human model promotion.
    """

    _INSTRUCTIONS = """You are the decision layer of a controlled academic ML prototype.
You receive an objective, aggregate column metadata, and experiment summaries only.
Return only the requested JSON object. Never propose or include code, shell commands,
file paths, URLs, hyperparameters, tools, data rows, or model names outside the supplied
choices. Your output is advisory: deterministic policy controls enforce all execution,
acceptance, and deployment rules."""

    def __init__(
        self,
        client: StructuredLLMClient,
        fallback: HeuristicDecisionAgent | None = None,
    ) -> None:
        self.client = client
        self.fallback = fallback or HeuristicDecisionAgent()

    @classmethod
    def from_environment(cls) -> "OpenAILLMDecisionAgent":
        key = os.getenv("OPENAI_API_KEY", "")
        model = os.getenv("AUTODS_LLM_MODEL", "gpt-4.1-mini")
        base_url = os.getenv("OPENAI_BASE_URL")
        timeout_raw = os.getenv("AUTODS_LLM_TIMEOUT_SECONDS", "30")
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise LLMConfigurationError("AUTODS_LLM_TIMEOUT_SECONDS must be numeric") from exc
        if not 1 <= timeout_seconds <= 120:
            raise LLMConfigurationError("AUTODS_LLM_TIMEOUT_SECONDS must be between 1 and 120")
        return cls(
            OpenAIResponsesStructuredClient(
                api_key=key,
                model=model,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
        )

    def create_plan(
        self, profile: DatasetProfile, objective: str, config: RunConfig
    ) -> MLPlan:
        # The offline planner provides safe defaults and validates whether the
        # target/objective is even meaningful before a network call is made.
        baseline = self.fallback.create_plan(profile, objective, config)
        all_models = tuple(
            dict.fromkeys(
                model for task in TaskType for model in _available_candidates(task)
            )
        )
        target_choices = [config.target_column] if config.target_column else [
            column.name for column in profile.columns
        ]
        payload = {
            "objective": objective,
            "operator_target_override": config.target_column,
            "baseline_plan": {
                "target_column": baseline.target_column,
                "task_type": baseline.task_type.value,
                "metric": baseline.metric,
                "candidate_models": list(baseline.candidate_models),
            },
            "profile": _profile_metadata(profile),
        }
        response = self.client.generate(
            schema_name="autods_ml_plan",
            schema=_strict_object_schema(
                {
                    "target_column": {"type": "string", "enum": target_choices},
                    "task_type": {
                        "type": "string",
                        "enum": [task.value for task in TaskType],
                    },
                    "metric": {
                        "type": "string",
                        "enum": sorted({metric for values in ALLOWED_METRICS.values() for metric in values}),
                    },
                    "candidate_models": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(all_models)},
                        "minItems": 1,
                        "maxItems": len(all_models),
                    },
                    "reasoning": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 4,
                    },
                }
            ),
            instructions=self._INSTRUCTIONS,
            payload=payload,
        )
        try:
            target_column = str(response["target_column"])
            task_type = TaskType(str(response["task_type"]))
            metric = str(response["metric"])
            candidate_models = tuple(str(item) for item in response["candidate_models"])
            reasoning = tuple(_bounded_text(item) for item in response["reasoning"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMResponseError("OpenAI plan response did not match the requested contract") from exc
        if target_column not in target_choices:
            raise LLMResponseError("OpenAI selected a target outside the allowed choices")
        target = profile.column(target_column)
        if target.likely_identifier or target.missing_fraction > 0.50 or target.unique_count < 2:
            raise LLMResponseError("OpenAI selected an unsafe or invalid target column")
        if metric not in ALLOWED_METRICS[task_type]:
            raise LLMResponseError("OpenAI selected a metric incompatible with its task type")
        allowed_candidates = _available_candidates(task_type)
        if (
            not candidate_models
            or len(set(candidate_models)) != len(candidate_models)
            or any(model not in allowed_candidates for model in candidate_models)
        ):
            raise LLMResponseError("OpenAI selected invalid or unavailable candidate models")
        excluded = tuple(
            column.name
            for column in profile.columns
            if column.name != target_column and column.likely_identifier
        )
        return MLPlan(
            objective=objective,
            target_column=target_column,
            task_type=task_type,
            metric=metric,
            # The model cannot loosen the human/configured success criterion.
            acceptance_threshold=baseline.acceptance_threshold,
            candidate_models=candidate_models,
            preprocessing=PreprocessingSpec(excluded_columns=excluded),
            max_experiments=baseline.max_experiments,
            test_size=baseline.test_size,
            random_state=baseline.random_state,
            reasoning=(
                "OpenAI produced a strict structured planning recommendation; "
                "deterministic policy validation remains authoritative.",
                *reasoning,
            ),
        )

    def initial_decision(self, plan: MLPlan) -> AgentDecision:
        response = self.client.generate(
            schema_name="autods_initial_decision",
            schema=_strict_object_schema(
                {
                    "model_name": {"type": "string", "enum": list(plan.candidate_models)},
                    "reason": {"type": "string"},
                }
            ),
            instructions=self._INSTRUCTIONS,
            payload={
                "objective": plan.objective,
                "metric": plan.metric,
                "acceptance_threshold": plan.acceptance_threshold,
                "approved_candidate_models": list(plan.candidate_models),
                "instruction": "Choose the first approved model to evaluate.",
            },
        )
        model_name = str(response.get("model_name", ""))
        if model_name not in plan.candidate_models:
            raise LLMResponseError("OpenAI initial decision selected a model outside the plan")
        reason = _bounded_text(response.get("reason", ""))
        if not reason:
            raise LLMResponseError("OpenAI initial decision did not include a reason")
        return AgentDecision(
            action="run_experiment",
            model_name=model_name,
            strategy="llm_structured_baseline_selection",
            reason=reason,
        )

    def assess_results(
        self, plan: MLPlan, history: list[ExperimentResult]
    ) -> AgentDecision:
        response = self.client.generate(
            schema_name="autods_experiment_assessment",
            schema=_strict_object_schema(
                {
                    "action": {"type": "string", "enum": ["run_experiment", "accept", "stop"]},
                    "model_name": {"type": "string", "enum": list(plan.candidate_models)},
                    "reason": {"type": "string"},
                }
            ),
            instructions=self._INSTRUCTIONS,
            payload={
                "plan": {
                    "metric": plan.metric,
                    "acceptance_threshold": plan.acceptance_threshold,
                    "max_experiments": plan.max_experiments,
                    "approved_candidate_models": list(plan.candidate_models),
                },
                "experiments": [item.to_dict() for item in history],
                "instruction": (
                    "Choose one action. A run_experiment decision must select an untried approved "
                    "model. Accept means recommend the current best result; stop means end the run."
                ),
            },
        )
        action = str(response.get("action", ""))
        model_name = str(response.get("model_name", ""))
        reason = _bounded_text(response.get("reason", ""))
        if action not in {"run_experiment", "accept", "stop"}:
            raise LLMResponseError("OpenAI assessment returned an unsupported action")
        if model_name not in plan.candidate_models:
            raise LLMResponseError("OpenAI assessment selected a model outside the plan")
        if action == "run_experiment" and model_name in {item.model_name for item in history}:
            raise LLMResponseError("OpenAI assessment attempted a duplicate model recipe")
        if not reason:
            raise LLMResponseError("OpenAI assessment did not include a reason")
        return AgentDecision(
            action=action,
            model_name=model_name,
            strategy="llm_structured_experiment_assessment",
            reason=reason,
        )


def create_decision_agent(mode: str | None = None) -> DecisionAgent:
    """Select the offline default or an explicitly configured OpenAI agent."""

    selected = (mode or os.getenv("AUTODS_AGENT_MODE", "heuristic")).strip().lower()
    if selected in {"heuristic", "offline"}:
        return HeuristicDecisionAgent()
    if selected in {"openai", "llm"}:
        return OpenAILLMDecisionAgent.from_environment()
    raise LLMConfigurationError(
        "AUTODS_AGENT_MODE must be 'heuristic' or 'openai'"
    )
