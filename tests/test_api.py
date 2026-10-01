"""API integration tests using FastAPI TestClient."""

from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from fastapi.testclient import TestClient
except RuntimeError as exc:  # Some newer Starlette builds require optional httpx2.
    pytest.skip(f"FastAPI TestClient is unavailable in this environment: {exc}", allow_module_level=True)


@pytest.fixture(autouse=True)
def _isolate_api(tmp_path, monkeypatch):
    """Point every API directory at a temp folder so tests are isolated."""
    monkeypatch.setenv("AUTODS_HOME", str(tmp_path))
    monkeypatch.setenv("AUTODS_DATASET_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("AUTODS_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("AUTODS_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("AUTODS_APPROVAL_TOKEN", "test-secret-token")
    # Re-import to pick up the monkeypatched env vars
    import autods.api as api_module
    monkeypatch.setattr(api_module, "DATASET_ROOT", (tmp_path / "datasets").resolve())
    monkeypatch.setattr(api_module, "ARTIFACT_ROOT", (tmp_path / "artifacts").resolve())
    monkeypatch.setattr(api_module, "MODEL_ROOT", (tmp_path / "models").resolve())
    (tmp_path / "datasets").mkdir(exist_ok=True)
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    (tmp_path / "models").mkdir(exist_ok=True)


@pytest.fixture
def client():
    from autods.api import app
    return TestClient(app)


@pytest.fixture
def sample_csv(tmp_path):
    """Create a simple binary classification CSV."""
    values = np.tile(np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]), 20)
    df = pd.DataFrame({
        "signal": values,
        "tenure": np.arange(120) % 12,
        "churn": np.where(values > 0, "yes", "no"),
    })
    path = tmp_path / "test_dataset.csv"
    df.to_csv(path, index=False)
    return path


def _upload_csv(client, csv_path: Path) -> str:
    """Upload a CSV and return the dataset_name."""
    with open(csv_path, "rb") as f:
        response = client.post("/datasets", files={"file": ("test.csv", f, "text/csv")})
    assert response.status_code == 200
    return response.json()["dataset_name"]


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestDatasetUpload:
    def test_upload_csv_succeeds(self, client, sample_csv):
        dataset_name = _upload_csv(client, sample_csv)
        assert dataset_name.endswith(".csv")
        assert len(dataset_name) == 36  # 32 hex + ".csv"

    def test_upload_non_csv_rejected(self, client, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("not a csv")
        with open(path, "rb") as f:
            response = client.post("/datasets", files={"file": ("data.txt", f, "text/plain")})
        assert response.status_code == 400

    def test_upload_empty_rejected(self, client):
        from io import BytesIO
        response = client.post("/datasets", files={"file": ("empty.csv", BytesIO(b""), "text/csv")})
        assert response.status_code == 400


class TestRunEndpoints:
    def test_full_run_workflow(self, client, sample_csv):
        """Upload → Run → List → Get run by ID."""
        dataset_name = _upload_csv(client, sample_csv)
        run_response = client.post("/runs", json={
            "dataset_name": dataset_name,
            "objective": "Classify churn with F1 at least 0.90",
            "target_column": "churn",
            "acceptance_threshold": 0.90,
            "max_experiments": 2,
        })
        assert run_response.status_code == 200
        run_data = run_response.json()
        assert run_data["status"] in ["needs_review", "stopped", "failed"]
        assert run_data["run_id"]
        assert run_data["plan"] is not None
        assert len(run_data["experiments"]) >= 1

        # List runs
        list_response = client.get("/runs")
        assert list_response.status_code == 200
        runs = list_response.json()
        assert len(runs) >= 1
        assert any(r["run_id"] == run_data["run_id"] for r in runs)

        # Get specific run
        get_response = client.get(f"/runs/{run_data['run_id']}")
        assert get_response.status_code == 200

    def test_invalid_run_id_rejected(self, client):
        response = client.get("/runs/not-a-valid-id")
        assert response.status_code == 400

    def test_nonexistent_run_returns_404(self, client):
        response = client.get("/runs/" + "a" * 32)
        assert response.status_code == 404


class TestPredictionEndpoint:
    def test_predict_without_model_returns_503(self, client):
        response = client.post("/predict", json={
            "records": [{"signal": 1.0, "tenure": 5}],
        })
        assert response.status_code == 503


class TestPromotionEndpoint:
    def test_approve_without_token_returns_403(self, client):
        response = client.post(
            "/promotions/" + "a" * 32 + "/approve",
            json={"approved_by": "tester"},
        )
        assert response.status_code == 403

    def test_approve_nonexistent_proposal_returns_400(self, client):
        response = client.post(
            "/promotions/" + "a" * 32 + "/approve",
            json={"approved_by": "tester"},
            headers={"X-Approval-Token": "test-secret-token"},
        )
        assert response.status_code == 400


class TestModelEndpoints:
    def test_no_production_model_returns_404(self, client):
        response = client.get("/models/production")
        assert response.status_code == 404

    def test_empty_pending_list(self, client):
        response = client.get("/models/pending")
        assert response.status_code == 200
        assert response.json() == []


class TestMonitoringEndpoint:
    def test_drift_without_production_model_returns_503(self, client, sample_csv):
        dataset_name = _upload_csv(client, sample_csv)
        response = client.post("/monitor/drift", json={
            "current_dataset_name": dataset_name,
            "drift_threshold": 0.20,
        })
        assert response.status_code == 503


class TestRetrainEndpoint:
    def test_retrain_creates_new_run(self, client, sample_csv):
        dataset_name = _upload_csv(client, sample_csv)
        response = client.post("/retrain", json={
            "dataset_name": dataset_name,
            "objective": "Classify churn with F1 at least 0.90",
            "target_column": "churn",
            "acceptance_threshold": 0.90,
            "max_experiments": 1,
        }, headers={"X-Approval-Token": "test-secret-token"})
        assert response.status_code == 200
        data = response.json()
        assert data["retraining"] is True
        assert data["run_id"]
