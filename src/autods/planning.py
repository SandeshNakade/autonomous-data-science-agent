"""Plan-only entry point for the controlled agent workflow.

The web planning page uses this module to inspect an already validated dataset
and return a declarative ``MLPlan``.  It intentionally does not construct a
tracker, train an estimator, write an experiment artifact, or promote a model.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .agent import DecisionAgent, PlanningError
from .policy import PolicyViolation, validate_plan
from .profiling import profile_dataset
from .schemas import AgentDecision, DatasetProfile, MLPlan, RunConfig


def create_controlled_plan(
    dataset_path: str | Path,
    objective: str,
    config: RunConfig,
    agent: DecisionAgent,
) -> tuple[DatasetProfile, MLPlan, AgentDecision]:
    """Return a policy-validated plan without starting a workflow.

    Operator settings remain authoritative over the recommendation returned by
    either the offline heuristic or the constrained structured LLM adapter.
    """

    profile = profile_dataset(dataset_path)
    recommended = agent.create_plan(profile, objective, config)
    plan = replace(
        recommended,
        target_column=config.target_column or recommended.target_column,
        acceptance_threshold=(
            config.acceptance_threshold
            if config.acceptance_threshold is not None
            else recommended.acceptance_threshold
        ),
        max_experiments=config.max_experiments,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    validate_plan(plan)
    decision = agent.initial_decision(plan)
    if (
        decision.action != "run_experiment"
        or decision.model_name not in plan.candidate_models
    ):
        raise PlanningError("The planning adapter returned an invalid first experiment decision")
    return profile, plan, decision


__all__ = ["PlanningError", "PolicyViolation", "create_controlled_plan"]
