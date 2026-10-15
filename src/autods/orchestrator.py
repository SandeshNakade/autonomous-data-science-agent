"""Readable end-to-end state machine for the controlled feedback loop."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent import DecisionAgent, create_decision_agent
from .monitoring import build_drift_baseline
from .policy import PolicyViolation, metric_higher_is_better, validate_plan
from .preprocessing import PreparationError, split_dataset
from .profiling import DatasetError, dataset_hash, profile_dataset, read_csv_dataset
from .registry import LocalModelRegistry
from .schemas import (
    ExperimentResult,
    MLPlan,
    RunConfig,
    RunStatus,
    RunSummary,
)
from .tracking import LocalExperimentTracker, create_tracker
from .training import train_and_evaluate


def _best_result(history: list[ExperimentResult], metric: str) -> ExperimentResult | None:
    valid = [
        item
        for item in history
        if item.status == "completed" and item.score is not None and math.isfinite(item.score)
    ]
    if not valid:
        return None
    if metric_higher_is_better(metric):
        return max(valid, key=lambda item: float(item.score))
    return min(valid, key=lambda item: float(item.score))


def _meets_acceptance(result: ExperimentResult | None, plan: MLPlan) -> bool:
    if result is None or result.score is None or not math.isfinite(result.score):
        return False
    if metric_higher_is_better(plan.metric):
        return result.score >= plan.acceptance_threshold
    return result.score <= plan.acceptance_threshold


class AutonomousDataScienceOrchestrator:
    """Coordinates agent decisions with a fixed collection of Python tools.

    Important boundary: the agent can only return a model name and strategy from
    the allowlisted schema.  This class selects and invokes the concrete Python
    functions; it never evaluates agent-supplied code or commands.
    """

    def __init__(
        self,
        artifact_root: str | Path = "artifacts",
        model_root: str | Path = "models",
        agent: DecisionAgent | None = None,
        tracker: LocalExperimentTracker | None = None,
        registry: LocalModelRegistry | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        # The default remains offline and reproducible. Setting
        # AUTODS_AGENT_MODE=openai opts every surface (CLI, API, dashboard)
        # into the constrained structured LLM decision adapter.
        self.agent = agent or create_decision_agent()
        self.tracker = tracker
        self.registry = registry or LocalModelRegistry(model_root)

    def run(
        self,
        dataset_path: str | Path,
        objective: str,
        config: RunConfig | None = None,
    ) -> RunSummary:
        config = config or RunConfig()
        profile = profile_dataset(dataset_path)
        tracker = self.tracker or create_tracker(
            self.artifact_root / "runs", use_mlflow=config.use_mlflow
        )
        run_id = tracker.start_run(objective, profile)
        tracker.log_event(
            run_id,
            "dataset_profiled",
            "Dataset profile created from the immutable input.",
            rows=profile.rows,
            columns=len(profile.columns),
        )
        plan: MLPlan | None = None
        history: list[ExperimentResult] = []
        trace: list[dict[str, Any]] = []
        final_evaluation: ExperimentResult | None = None

        try:
            agent_plan = self.agent.create_plan(profile, objective, config)
            # The agent may recommend a plan, but operator-supplied limits and
            # overrides are authoritative. This also protects future LLM-backed
            # DecisionAgent implementations from silently widening a run.
            plan = replace(
                agent_plan,
                target_column=config.target_column or agent_plan.target_column,
                acceptance_threshold=(
                    config.acceptance_threshold
                    if config.acceptance_threshold is not None
                    else agent_plan.acceptance_threshold
                ),
                max_experiments=config.max_experiments,
                test_size=config.test_size,
                random_state=config.random_state,
            )
            validate_plan(plan)
            tracker.log_plan(run_id, plan)
            tracker.log_event(
                run_id,
                "plan_validated",
                "Agent plan passed policy validation.",
                target_column=plan.target_column,
                metric=plan.metric,
                max_experiments=plan.max_experiments,
            )

            data = read_csv_dataset(dataset_path)
            if dataset_hash(dataset_path) != profile.data_hash:
                raise DatasetError("Dataset changed after profiling; run was stopped before training")
            prepared_data = split_dataset(data, plan)
            split = prepared_data.validation_split
            tracker.log_event(
                run_id,
                "data_prepared",
                "Created training, validation, and untouched holdout splits.",
                feature_count=len(split.feature_columns),
                training_rows=len(split.X_train),
                validation_rows=len(split.X_test),
                holdout_rows=len(prepared_data.holdout_split.X_test),
            )
            decision = self.agent.initial_decision(plan)
            stop_reason = ""
            accepted = False

            while True:
                decision_payload = decision.to_dict()
                trace.append(decision_payload)
                tracker.log_decision(run_id, decision_payload)
                if decision.action == "accept":
                    accepted = True
                    stop_reason = decision.reason
                    break
                if decision.action == "stop":
                    stop_reason = decision.reason
                    break
                if decision.action != "run_experiment" or not decision.model_name:
                    raise PolicyViolation("Agent returned an unsupported workflow action")
                if len(history) >= plan.max_experiments:
                    raise PolicyViolation("Agent attempted to exceed the configured experiment budget")
                if decision.model_name not in plan.candidate_models:
                    raise PolicyViolation("Agent selected a model outside the approved plan")
                if decision.model_name in {item.model_name for item in history}:
                    raise PolicyViolation("Agent attempted a duplicate model recipe")
                experiment_number = len(history) + 1
                experiment_id = f"{run_id[:8]}-trial-{experiment_number}"
                tracker.log_event(
                    run_id,
                    "experiment_started",
                    f"Started experiment {experiment_number} with {decision.model_name}.",
                    experiment_id=experiment_id,
                    model_name=decision.model_name,
                    strategy=decision.strategy,
                )
                result = train_and_evaluate(
                    split=split,
                    plan=plan,
                    model_name=decision.model_name,
                    experiment_id=experiment_id,
                    strategy=decision.strategy,
                )
                history.append(result)
                tracker.log_experiment(run_id, result)
                tracker.log_event(
                    run_id,
                    (
                        "experiment_completed"
                        if result.status == "completed"
                        else "experiment_failed"
                    ),
                    (
                        f"Experiment {experiment_number} completed."
                        if result.status == "completed"
                        else f"Experiment {experiment_number} did not complete."
                    ),
                    experiment_id=result.experiment_id,
                    model_name=result.model_name,
                    metric=result.metric,
                    score=result.score,
                    elapsed_seconds=result.elapsed_seconds,
                    error=result.error,
                )
                decision = self.agent.assess_results(plan, history)

            selected = _best_result(history, plan.metric)
            proposal_id: str | None = None
            quality_gate_met = _meets_acceptance(selected, plan)
            if accepted and not quality_gate_met:
                accepted = False
                stop_reason = (
                    "Agent acceptance was rejected by the deterministic quality gate: "
                    "the selected result did not meet the declared threshold."
                )
            if accepted and quality_gate_met and selected:
                # The agent saw validation metrics only. Refit the selected fixed
                # recipe on fit+validation data and touch the holdout exactly once.
                holdout_id = f"{run_id[:8]}-final-holdout"
                tracker.log_event(
                    run_id,
                    "holdout_evaluation_started",
                    "Refitting the selected recipe and evaluating it once on the untouched holdout.",
                    experiment_id=holdout_id,
                    model_name=selected.model_name,
                )
                final_evaluation = train_and_evaluate(
                    split=prepared_data.holdout_split,
                    plan=plan,
                    model_name=selected.model_name,
                    experiment_id=holdout_id,
                    strategy="final_holdout_evaluation",
                )
                tracker.log_final_evaluation(run_id, final_evaluation)
                tracker.log_event(
                    run_id,
                    (
                        "holdout_evaluation_completed"
                        if final_evaluation.status == "completed"
                        else "holdout_evaluation_failed"
                    ),
                    (
                        "Final holdout evaluation completed."
                        if final_evaluation.status == "completed"
                        else "Final holdout evaluation did not complete."
                    ),
                    experiment_id=final_evaluation.experiment_id,
                    metric=final_evaluation.metric,
                    score=final_evaluation.score,
                    elapsed_seconds=final_evaluation.elapsed_seconds,
                    error=final_evaluation.error,
                )
                if not _meets_acceptance(final_evaluation, plan):
                    accepted = False
                    stop_reason = (
                        "The selected validation winner did not meet the acceptance criterion "
                        "on the untouched final holdout."
                    )
            if (
                accepted
                and quality_gate_met
                and selected
                and final_evaluation
                and final_evaluation.model_bundle is not None
            ):
                monitoring_baseline = build_drift_baseline(
                    dataset_path,
                    excluded_columns=(plan.target_column, *plan.preprocessing.excluded_columns),
                )
                if monitoring_baseline.source_hash != profile.data_hash:
                    raise DatasetError(
                        "Dataset changed before its monitoring baseline could be saved; promotion was stopped"
                    )
                proposal_id = self.registry.create_proposal(
                    final_evaluation.model_bundle,
                    {
                        "run_id": run_id,
                        "experiment_id": selected.experiment_id,
                        "final_evaluation_id": final_evaluation.experiment_id,
                        "model_name": selected.model_name,
                        "metric": selected.metric,
                        "validation_score": selected.score,
                        "validation_metrics": selected.metrics,
                        "holdout_score": final_evaluation.score,
                        "holdout_metrics": final_evaluation.metrics,
                        "dataset_sha256": profile.data_hash,
                        "plan": plan.to_dict(),
                        "input_schema": [
                            {
                                "name": column.name,
                                "inferred_type": column.inferred_type,
                                "pandas_dtype": column.pandas_dtype,
                            }
                            for column in profile.columns
                            if column.name != plan.target_column
                            and column.name not in plan.preprocessing.excluded_columns
                        ],
                    },
                    monitoring_baseline=monitoring_baseline.to_dict(),
                )
                tracker.log_event(
                    run_id,
                    "candidate_submitted",
                    "Candidate model submitted for explicit human approval.",
                    proposal_id=proposal_id,
                    model_name=selected.model_name,
                )
                status = RunStatus.NEEDS_REVIEW
                stop_reason = (
                    f"{stop_reason} Model is pending explicit human approval before promotion."
                )
            elif selected:
                status = RunStatus.STOPPED
            else:
                status = RunStatus.FAILED
                stop_reason = stop_reason or "No candidate model completed successfully."

            summary = RunSummary(
                run_id=run_id,
                status=status,
                profile=profile,
                plan=plan,
                experiments=history,
                decision_trace=trace,
                selected_experiment_id=selected.experiment_id if selected else None,
                final_evaluation=final_evaluation,
                promotion_proposal_id=proposal_id,
                stop_reason=stop_reason,
            )
        except (DatasetError, PreparationError, PolicyViolation, ValueError) as exc:
            trace.append({"action": "stop", "strategy": "safety_or_input_failure", "reason": str(exc)})
            tracker.log_event(
                run_id,
                "workflow_failed",
                "Workflow stopped by a safety or input check.",
                reason=str(exc),
            )
            summary = RunSummary(
                run_id=run_id,
                status=RunStatus.FAILED,
                profile=profile,
                plan=plan,
                experiments=history,
                decision_trace=trace,
                selected_experiment_id=None,
                final_evaluation=final_evaluation,
                promotion_proposal_id=None,
                stop_reason=str(exc),
            )
        except Exception as exc:  # retain an audit record without exposing a raw traceback to API users
            trace.append(
                {
                    "action": "stop",
                    "strategy": "unexpected_failure",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            tracker.log_event(
                run_id,
                "workflow_failed",
                "Workflow stopped because an unexpected error was recorded.",
                reason=trace[-1]["reason"],
            )
            summary = RunSummary(
                run_id=run_id,
                status=RunStatus.FAILED,
                profile=profile,
                plan=plan,
                experiments=history,
                decision_trace=trace,
                selected_experiment_id=None,
                final_evaluation=final_evaluation,
                promotion_proposal_id=None,
                stop_reason=trace[-1]["reason"],
            )
        tracker.log_event(
            run_id,
            "run_completed",
            f"Workflow finished with status {summary.status.value}.",
            status=summary.status.value,
            stop_reason=summary.stop_reason,
        )
        tracker.complete(run_id, summary)
        return summary
