from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from autods.agent import HeuristicDecisionAgent
from autods.monitoring import (
    build_drift_baseline,
    compare_to_baseline,
    retraining_request,
)
from autods.orchestrator import AutonomousDataScienceOrchestrator
from autods.policy import PolicyViolation, validate_plan
from autods.profiling import profile_dataset
from autods.registry import LocalModelRegistry, RegistryError
from autods.schemas import AgentDecision, MLPlan, PreprocessingSpec, RunConfig, RunStatus, TaskType


def _write_csv(tmp_path, name: str, frame: pd.DataFrame):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def test_binary_run_creates_pending_proposal_then_requires_manual_promotion(tmp_path):
    """A separable CSV reaches the quality gate, but is never auto-promoted."""

    # Repeated values ensure the feature is not profiled as a high-cardinality ID.
    values = np.tile(np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]), 20)
    dataset = _write_csv(
        tmp_path,
        "churn.csv",
        pd.DataFrame(
            {
                "customer_id": np.arange(1, 121),
                "signal": values,
                "tenure": np.arange(120) % 12,
                "segment": np.where(np.arange(120) % 2, "standard", "premium"),
                "churn": np.where(values > 0, "yes", "no"),
            }
        ),
    )
    registry = LocalModelRegistry(tmp_path / "models")
    summary = AutonomousDataScienceOrchestrator(
        artifact_root=tmp_path / "artifacts",
        registry=registry,
    ).run(
        dataset,
        "Classify churn and achieve F1 at least 0.90.",
        RunConfig(max_experiments=2, acceptance_threshold=0.90),
    )

    assert summary.status is RunStatus.NEEDS_REVIEW
    assert summary.plan is not None
    assert summary.plan.task_type is TaskType.BINARY_CLASSIFICATION
    assert summary.plan.metric == "f1"
    assert "customer_id" in summary.plan.preprocessing.excluded_columns
    assert len(summary.experiments) == 1
    assert summary.experiments[0].model_name == "logistic_regression"
    assert summary.experiments[0].score is not None
    assert summary.experiments[0].score >= 0.90
    assert summary.promotion_proposal_id is not None
    assert summary.decision_trace[-1]["action"] == "accept"

    # The successful run deliberately leaves the artifact pending until a person acts.
    with pytest.raises(RegistryError, match="No approved production model"):
        registry.load_production()

    promoted = registry.approve(summary.promotion_proposal_id, "academic-reviewer")
    assert promoted["status"] == "promoted"
    assert promoted["approved_by"] == "academic-reviewer"
    model = registry.load_production()
    baseline = registry.production_baseline()
    assert "signal" in baseline["numeric"]
    assert model.predict(
        [
            {"signal": -2.0, "tenure": 3, "segment": "standard"},
            {"signal": 2.0, "tenure": 8, "segment": "premium"},
        ]
    ) == ["no", "yes"]


def test_reviewer_approved_candidate_requires_separate_admin_deployment(tmp_path):
    """Review alone cannot serve a model; deployment creates an immutable release."""

    values = np.tile(np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]), 20)
    dataset = _write_csv(
        tmp_path,
        "deployment-churn.csv",
        pd.DataFrame(
            {
                "customer_id": np.arange(1, 121),
                "signal": values,
                "tenure": np.arange(120) % 12,
                "segment": np.where(np.arange(120) % 2, "standard", "premium"),
                "churn": np.where(values > 0, "yes", "no"),
            }
        ),
    )
    registry = LocalModelRegistry(tmp_path / "models")
    summary = AutonomousDataScienceOrchestrator(
        artifact_root=tmp_path / "artifacts",
        registry=registry,
    ).run(
        dataset,
        "Classify churn and achieve F1 at least 0.90.",
        RunConfig(max_experiments=2, acceptance_threshold=0.90),
    )

    assert summary.promotion_proposal_id is not None
    proposal_id = summary.promotion_proposal_id
    reviewed = registry.review(proposal_id, "approve", "academic-reviewer", "Holdout evidence is sufficient.")
    assert reviewed["status"] == "approved_for_deployment"
    with pytest.raises(RegistryError, match="No approved production model"):
        registry.load_production()

    deployed = registry.deploy(proposal_id, "mlops-admin", "Release the approved candidate for local serving.")

    assert deployed["status"] == "deployed"
    assert deployed["deployment"]["deployed_by"] == "mlops-admin"
    assert deployed["deployment"]["input_schema"]
    assert registry.deployment_status()["state"] == "ready"
    assert registry.list_deployments()[0]["proposal_id"] == proposal_id
    assert (tmp_path / "models" / "releases" / proposal_id / "model.joblib").is_file()
    assert registry.load_production().predict(
        [
            {"signal": -2.0, "tenure": 3, "segment": "standard"},
            {"signal": 2.0, "tenure": 8, "segment": "premium"},
        ]
    ) == ["no", "yes"]


@pytest.mark.parametrize(
    ("frame", "objective", "target", "expected_task", "expected_metric"),
    [
        (
            pd.DataFrame(
                {
                    "floor_area": np.linspace(30.0, 180.0, 40),
                    "bedrooms": np.tile([1, 2, 3, 4], 10),
                    "house_price": np.repeat(
                        [120_000.0, 280_000.0, 440_000.0, 600_000.0], 10
                    ),
                }
            ),
            "Use regression to predict house_price with R2 at least 0.70.",
            "house_price",
            TaskType.REGRESSION,
            "r2",
        ),
        (
            pd.DataFrame(
                {
                    "exam_score": np.tile([40, 55, 70, 85, 95], 12),
                    "attendance": np.linspace(0.5, 1.0, 60),
                    "segment": np.tile(["at_risk", "developing", "advanced"], 20),
                }
            ),
            "Classify segment into categories using macro F1.",
            "segment",
            TaskType.MULTICLASS_CLASSIFICATION,
            "f1_macro",
        ),
    ],
    ids=["regression", "multiclass"],
)
def test_agent_inferrs_supported_regression_and_multiclass_plans(
    tmp_path, frame, objective, target, expected_task, expected_metric
):
    dataset = _write_csv(tmp_path, f"{target}.csv", frame)

    plan = HeuristicDecisionAgent().create_plan(
        profile_dataset(dataset), objective, RunConfig(target_column=target)
    )

    assert plan.target_column == target
    assert plan.task_type is expected_task
    assert plan.metric == expected_metric
    assert plan.max_experiments == 3
    validate_plan(plan)
    assert set(plan.candidate_models).issubset(
        {
            "logistic_regression",
            "linear_regression",
            "ridge",
            "random_forest",
            "xgboost",
        }
    )


def test_policy_rejects_a_non_allowlisted_recipe_without_executing_it():
    unsafe_plan = MLPlan(
        objective="Classify churn",
        target_column="churn",
        task_type=TaskType.BINARY_CLASSIFICATION,
        metric="f1",
        acceptance_threshold=0.75,
        candidate_models=("__import__('os').system('should-not-run')",),
        preprocessing=PreprocessingSpec(),
    )

    with pytest.raises(PolicyViolation, match="Unsupported model request"):
        validate_plan(unsafe_plan)


def test_regression_run_accepts_a_continuous_high_cardinality_target(tmp_path):
    """Continuous targets are valid values, not automatically treated as IDs."""

    area = np.linspace(40.0, 180.0, 100)
    bedrooms = np.tile([1, 2, 3, 4], 25)
    dataset = _write_csv(
        tmp_path,
        "prices.csv",
        pd.DataFrame(
            {
                "floor_area": area,
                "bedrooms": bedrooms,
                "price": 2_000 * area + 15_000 * bedrooms + 5_000,
            }
        ),
    )
    summary = AutonomousDataScienceOrchestrator(
        artifact_root=tmp_path / "artifacts",
        model_root=tmp_path / "models",
    ).run(
        dataset,
        "Use regression to predict price with R2 at least 0.99.",
        RunConfig(target_column="price", acceptance_threshold=0.99, max_experiments=1),
    )

    assert summary.status is RunStatus.NEEDS_REVIEW
    assert summary.plan is not None
    assert summary.plan.task_type is TaskType.REGRESSION
    assert summary.experiments[0].model_name == "ridge"
    assert summary.experiments[0].metrics["r2"] > 0.99


def test_rmse_objective_parses_lower_is_better_threshold(tmp_path):
    dataset = _write_csv(
        tmp_path,
        "rmse.csv",
        pd.DataFrame(
            {
                "feature": np.linspace(0.0, 1.0, 30),
                "target": np.linspace(0.0, 1.0, 30),
            }
        ),
    )
    plan = HeuristicDecisionAgent().create_plan(
        profile_dataset(dataset),
        "Use regression to predict target with RMSE below 0.20.",
        RunConfig(target_column="target"),
    )

    assert plan.metric == "rmse"
    assert plan.acceptance_threshold == 0.20


def test_orchestrator_enforces_operator_quality_controls_over_agent_acceptance(tmp_path):
    class PrematureAcceptanceAgent:
        def create_plan(self, profile, objective, config):
            return MLPlan(
                objective=objective,
                target_column="churn",
                task_type=TaskType.BINARY_CLASSIFICATION,
                metric="f1",
                acceptance_threshold=0.0,  # must be replaced by operator input
                candidate_models=("logistic_regression",),
                preprocessing=PreprocessingSpec(),
                max_experiments=5,  # must be replaced by operator input
                test_size=0.4,
                random_state=999,
            )

        def initial_decision(self, plan):
            return AgentDecision("run_experiment", "logistic_regression", "baseline", "run")

        def assess_results(self, plan, history):
            return AgentDecision("accept", "logistic_regression", "unsafe_accept", "accept anyway")

    dataset = _write_csv(
        tmp_path,
        "noisy_churn.csv",
        pd.DataFrame(
            {
                "signal": np.zeros(100),
                "churn": np.tile(["yes", "no"], 50),
            }
        ),
    )
    summary = AutonomousDataScienceOrchestrator(
        artifact_root=tmp_path / "artifacts",
        model_root=tmp_path / "models",
        agent=PrematureAcceptanceAgent(),
    ).run(
        dataset,
        "Classify churn",
        RunConfig(acceptance_threshold=0.99, max_experiments=1, random_state=42),
    )

    assert summary.status is RunStatus.STOPPED
    assert summary.plan is not None
    assert summary.plan.acceptance_threshold == 0.99
    assert summary.plan.max_experiments == 1
    assert summary.promotion_proposal_id is None
    assert "deterministic quality gate" in summary.stop_reason


def test_drift_only_creates_a_retraining_request_not_automatic_training(tmp_path):
    reference = _write_csv(
        tmp_path,
        "reference.csv",
        pd.DataFrame(
            {
                "income": np.linspace(30_000.0, 80_000.0, 100),
                "segment": ["standard"] * 90 + ["premium"] * 10,
            }
        ),
    )
    current = _write_csv(
        tmp_path,
        "current.csv",
        pd.DataFrame(
            {
                "income": np.linspace(300_000.0, 500_000.0, 100),
                "segment": ["premium"] * 100,
            }
        ),
    )

    report = compare_to_baseline(build_drift_baseline(reference), current, threshold=0.05)
    request = retraining_request(report, source_run_id="a" * 32)

    assert report.drift_detected
    assert "segment" in report.drifted_columns
    assert request == {
        "state": "retraining_requested_pending_human_approval",
        "source_run_id": "a" * 32,
        "reason": "Drift detected in: " + ", ".join(report.drifted_columns),
        "drifted_columns": report.drifted_columns,
        "automatic_retraining": False,
    }
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "artifacts").exists()


def test_missing_monitored_column_is_json_safe_and_requests_review(tmp_path):
    reference = _write_csv(
        tmp_path,
        "reference_with_constant.csv",
        pd.DataFrame({"constant": [0.0] * 30, "segment": ["a"] * 30}),
    )
    current = _write_csv(
        tmp_path,
        "current_without_segment.csv",
        pd.DataFrame({"constant": [1.0] * 30, "other": ["new"] * 30}),
    )

    report = compare_to_baseline(build_drift_baseline(reference), current)

    assert {"constant", "segment"}.issubset(report.drifted_columns)
    assert report.categorical_tvd["segment"] is None
    assert "segment" in report.missing_columns
    json.dumps(report.to_dict(), allow_nan=False)


def test_high_cardinality_categorical_tail_is_stable_when_unchanged(tmp_path):
    values = [f"category-{index}" for index in range(100)]
    frame = pd.DataFrame({"category": values, "signal": np.arange(100) % 4})
    reference = _write_csv(tmp_path, "high_cardinality_reference.csv", frame)
    current = _write_csv(tmp_path, "high_cardinality_current.csv", frame)

    report = compare_to_baseline(build_drift_baseline(reference), current)

    assert report.categorical_tvd["category"] == 0.0
    assert "category" not in report.drifted_columns
