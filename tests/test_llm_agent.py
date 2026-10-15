from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autods.agent import (
    HeuristicDecisionAgent,
    LLMConfigurationError,
    LLMResponseError,
    OpenAILLMDecisionAgent,
    OpenAIResponsesStructuredClient,
    create_decision_agent,
)
from autods.profiling import profile_dataset
from autods.schemas import ExperimentResult, RunConfig, TaskType


class FakeStructuredClient:
    """Deterministic stand-in that proves no real provider is needed for tests."""

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _profile(tmp_path):
    path = tmp_path / "churn.csv"
    values = np.tile(np.array([-2.0, -1.0, 1.0, 2.0]), 20)
    pd.DataFrame(
        {
            "customer_id": [f"customer-{index}" for index in range(len(values))],
            "signal": values,
            "tenure": np.arange(len(values)) % 12,
            "churn": np.where(values > 0, "yes", "no"),
        }
    ).to_csv(path, index=False)
    return profile_dataset(path)


def test_openai_agent_uses_only_structured_metadata_and_allowlisted_choices(tmp_path):
    client = FakeStructuredClient(
        [
            {
                "target_column": "churn",
                "task_type": "binary_classification",
                "metric": "f1",
                "candidate_models": ["random_forest", "logistic_regression"],
                "reasoning": ["The target has two classes.", "F1 matches the objective."],
            },
            {"model_name": "random_forest", "reason": "Start with the nonlinear candidate."},
            {
                "action": "accept",
                "model_name": "random_forest",
                "reason": "The observed F1 meets the requested criterion.",
            },
        ]
    )
    agent = OpenAILLMDecisionAgent(client)
    profile = _profile(tmp_path)

    plan = agent.create_plan(
        profile,
        "Classify churn with F1 at least 0.75.",
        RunConfig(target_column="churn"),
    )

    assert plan.task_type is TaskType.BINARY_CLASSIFICATION
    assert plan.metric == "f1"
    assert plan.candidate_models == ("random_forest", "logistic_regression")
    assert "customer_id" in plan.preprocessing.excluded_columns
    assert "dataset_path" not in client.requests[0]["payload"]["profile"]
    assert "OpenAI produced" in plan.reasoning[0]

    initial = agent.initial_decision(plan)
    assert initial.model_name == "random_forest"
    assert initial.strategy == "llm_structured_baseline_selection"

    result = ExperimentResult(
        experiment_id="trial-1",
        model_name="random_forest",
        metric="f1",
        score=0.91,
        metrics={"f1": 0.91},
        status="completed",
        elapsed_seconds=0.01,
        strategy=initial.strategy,
    )
    assessment = agent.assess_results(plan, [result])
    assert assessment.action == "accept"
    assert assessment.strategy == "llm_structured_experiment_assessment"


def test_openai_agent_rejects_metric_incompatible_with_its_task(tmp_path):
    agent = OpenAILLMDecisionAgent(
        FakeStructuredClient(
            [
                {
                    "target_column": "churn",
                    "task_type": "binary_classification",
                    "metric": "r2",
                    "candidate_models": ["logistic_regression"],
                    "reasoning": ["invalid on purpose"],
                }
            ]
        )
    )

    with pytest.raises(LLMResponseError, match="metric incompatible"):
        agent.create_plan(
            _profile(tmp_path),
            "Classify churn with F1 at least 0.75.",
            RunConfig(target_column="churn"),
        )


def test_agent_factory_is_offline_by_default_and_rejects_unknown_modes(monkeypatch):
    monkeypatch.delenv("AUTODS_AGENT_MODE", raising=False)
    assert isinstance(create_decision_agent(), HeuristicDecisionAgent)

    with pytest.raises(LLMConfigurationError, match="heuristic.*openai"):
        create_decision_agent("not-a-provider")


@pytest.mark.parametrize("wrapped", [False, True])
def test_provider_error_retains_diagnostics_and_redacts_secrets(monkeypatch, wrapped):
    secret = "test-provider-secret-value"
    error = {
        "code": "invalid_json_schema",
        "param": "text.format.schema",
        "message": (
            "Invalid schema: unsupported constraint. "
            f"Credential {secret}; sk-another-test-key; Bearer another-token"
        ),
        "request": {"Authorization": f"Bearer {secret}"},
    }

    class BadRequestError(Exception):
        status_code = 400
        body = {"error": error} if wrapped else error

    def create(**kwargs):
        assert secret not in json.dumps(kwargs)
        assert kwargs["text"]["format"]["strict"] is True
        raise BadRequestError("Raw exception contains private request information")

    sdk_client = SimpleNamespace(api_key=secret, responses=SimpleNamespace(create=create))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw: sdk_client))
    client = OpenAIResponsesStructuredClient(api_key=secret, model="gpt-4.1-mini")

    with pytest.raises(LLMResponseError) as caught:
        client.generate(schema_name="test", schema={}, instructions="test", payload={})

    message = str(caught.value)
    assert "HTTP 400" in message
    assert "code: invalid_json_schema" in message
    assert "param: text.format.schema" in message
    assert "unsupported constraint" in message
    assert secret not in message
    assert "sk-another-test-key" not in message
    assert "another-token" not in message
    assert "Authorization" not in message
    assert "private request" not in message
    assert "[REDACTED]" in message


def test_non_json_provider_error_does_not_leak_raw_body(monkeypatch):
    class ProviderError(Exception):
        status_code = 502
        body = "<html>Private proxy configuration and credentials</html>"

    def create(**kwargs):
        raise ProviderError("raw request and headers")

    sdk_client = SimpleNamespace(api_key="test-secret", responses=SimpleNamespace(create=create))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw: sdk_client))
    client = OpenAIResponsesStructuredClient(api_key="test-secret", model="gpt-4.1-mini")
    with pytest.raises(LLMResponseError) as caught:
        client.generate(schema_name="test", schema={}, instructions="test", payload={})

    assert "HTTP 502" in str(caught.value)
    assert "HTML instead of a JSON API error" in str(caught.value)
    assert "Private proxy" not in str(caught.value)
    assert "raw request" not in str(caught.value)


@pytest.mark.parametrize(
    "body,expected",
    [
        ("Unsupported parameter: text.format", "Unsupported parameter: text.format"),
        ({"error": "Invalid request"}, "Invalid request"),
        ('{"error": {"message": "Invalid schema"}}', "Invalid schema"),
        (None, "SDK error message"),
        ("<html><title>Proxy rejected request</title>private data</html>", "Proxy rejected request"),
    ],
)
def test_diagnostic_supports_nonstandard_error_bodies(body, expected):
    from autods.agent import _safe_openai_error

    error = RuntimeError("private exception")
    error.body = body
    error.message = "SDK error message"
    message = _safe_openai_error(error)

    assert expected in message
    assert "private" not in message
