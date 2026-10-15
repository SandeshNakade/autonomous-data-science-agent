"""Stage 3 plan-preview API tests: controlled agent recommendations only."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autods.api import app
from autods.auth import create_user


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    import autods.api as module

    datasets = tmp_path / "datasets"
    datasets.mkdir()
    monkeypatch.setattr(module, "DATASET_ROOT", datasets)
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(module, "MODEL_ROOT", tmp_path / "models")
    monkeypatch.setenv("AUTODS_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("AUTODS_AGENT_MODE", "heuristic")
    return tmp_path


@pytest.fixture
def client(workspace):
    create_user("scientist@example.test", "Planning Tester", "scientist", "planning-test-password")
    with TestClient(app, headers={"X-AutoDS-Client": "web"}) as browser:
        response = browser.post(
            "/auth/login",
            json={"email": "scientist@example.test", "password": "planning-test-password"},
        )
        assert response.status_code == 200
        yield browser


def _upload(client: TestClient) -> str:
    response = client.post(
        "/datasets",
        files={
            "file": (
                "churn.csv",
                b"customer_id,tenure,contract,churn\nc1,3,month_to_month,yes\nc2,24,two_year,no\nc3,8,month_to_month,yes\nc4,40,one_year,no\n",
                "text/csv",
            )
        },
    )
    assert response.status_code == 200
    return response.json()["dataset_name"]


def test_plan_preview_returns_a_validated_plan_without_creating_a_run(client, workspace):
    dataset_name = _upload(client)

    response = client.post(
        "/plans/preview",
        json={
            "dataset_name": dataset_name,
            "objective": "Classify churn with F1 at least 0.85.",
            "target_column": "churn",
            "acceptance_threshold": 0.85,
            "max_experiments": 2,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["dataset"]["dataset_name"] == dataset_name
    assert payload["dataset"]["rows"] == 4
    assert payload["planning_adapter"] == "heuristic"
    assert payload["plan"]["target_column"] == "churn"
    assert payload["plan"]["task_type"] == "binary_classification"
    assert payload["plan"]["metric"] == "f1"
    assert payload["plan"]["acceptance_threshold"] == 0.85
    assert payload["plan"]["max_experiments"] == 2
    assert payload["plan"]["candidate_models"][0] == "logistic_regression"
    assert "customer_id" in payload["plan"]["preprocessing"]["excluded_columns"]
    assert payload["initial_decision"]["model_name"] == "logistic_regression"
    assert len(payload["guardrails"]) == 4
    assert "dataset_path" not in str(payload)
    assert not (workspace / "artifacts" / "runs").exists()
    assert client.get("/runs").json() == []


def test_plan_preview_rejects_an_identifier_target_and_keeps_the_dataset(client, workspace):
    dataset_name = _upload(client)
    response = client.post(
        "/plans/preview",
        json={
            "dataset_name": dataset_name,
            "objective": "Classify customer_id.",
            "target_column": "customer_id",
        },
    )

    assert response.status_code == 400
    assert "identifier" in response.json()["detail"]
    assert (workspace / "datasets" / dataset_name).is_file()
    assert not (workspace / "artifacts" / "runs").exists()


def test_plan_preview_requires_run_permission_and_a_signed_in_user(client):
    dataset_name = _upload(client)
    create_user("reviewer@example.test", "Plan Reviewer", "reviewer", "planning-test-password")
    assert client.post(
        "/auth/login",
        json={"email": "reviewer@example.test", "password": "planning-test-password"},
    ).status_code == 200

    response = client.post(
        "/plans/preview",
        json={"dataset_name": dataset_name, "objective": "Classify churn."},
    )
    assert response.status_code == 403
    with TestClient(app, headers={"X-AutoDS-Client": "web"}) as guest:
        assert guest.post(
            "/plans/preview",
            json={"dataset_name": dataset_name, "objective": "Classify churn."},
        ).status_code == 401
