"""Exercise real cookies and role enforcement, including direct API calls."""

import hashlib
import json
import time

import pytest
from fastapi.testclient import TestClient

from autods.api import app
from autods.auth import COOKIE, create_user, database


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTODS_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.delenv("AUTODS_COOKIE_SECURE", raising=False)
    import autods.api as module
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(module, "MODEL_ROOT", tmp_path / "models")
    monkeypatch.setattr(module, "DATASET_ROOT", tmp_path / "datasets")


@pytest.fixture
def client():
    return TestClient(app, headers={"X-AutoDS-Client": "web"})


def sign_in(client, role="scientist"):
    create_user(f"{role}@example.test", f"Test {role}", role, "a-long-test-password")
    return client.post("/auth/login", json={"email": f"{role}@example.test", "password": "a-long-test-password"})


def test_empty_workspace_requires_local_account_creation(client):
    assert client.get("/auth/status").json() == {"configured": False}
    assert client.post("/auth/register", json={"role": "admin"}).status_code in {404, 405}
    response = sign_in(client)
    assert response.status_code == 200
    assert client.get("/auth/status").json() == {"configured": True}


def test_cookie_session_and_logout(client):
    response = sign_in(client)
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and "Max-Age=28800" in cookie
    assert response.json()["role"] == "scientist"
    assert "password_hash" not in response.json()
    assert client.get("/auth/me").json()["name"] == "Test scientist"
    token = client.cookies.get(COOKIE)
    with database() as connection:
        session = connection.execute("SELECT * FROM sessions").fetchone()
        user = connection.execute("SELECT * FROM users").fetchone()
    assert session["token_hash"] == hashlib.sha256(token.encode()).hexdigest()
    assert "a-long-test-password" not in user["password_hash"]
    assert client.post("/auth/logout").status_code == 200
    client.cookies.set(COOKIE, token)
    assert client.get("/auth/me").status_code == 401


@pytest.mark.parametrize("path", ["/workspace/overview", "/runs", "/models/pending", "/models/production", "/monitoring/reports", "/alerts", "/audit-logs", "/auth/settings", "/auth/users"])
def test_reads_require_login(client, path):
    assert client.get(path).status_code == 401


def test_health_is_public(client):
    assert client.get("/health").status_code == 200


def test_wrong_password_and_unknown_email_use_same_error(client):
    sign_in(client)
    responses = [client.post("/auth/login", json={"email": email, "password": "wrong"})
                 for email in ["scientist@example.test", "missing@example.test"]]
    assert all(response.status_code == 401 for response in responses)
    assert responses[0].json() == responses[1].json()


def test_client_cannot_assign_its_own_role(client):
    sign_in(client)
    response = client.post("/auth/login", json={"email": "scientist@example.test", "password": "a-long-test-password", "role": "admin"})
    assert response.status_code == 422
    assert client.get("/auth/me").json()["role"] == "scientist"


@pytest.mark.parametrize("role", ["scientist", "reviewer"])
@pytest.mark.parametrize("path,payload", [
    ("/promotions/" + "a" * 32 + "/approve", {"approved_by": "Impersonated Admin"}),
    ("/deployments/" + "a" * 32, {"deployment_note": "Attempting to bypass deployment controls."}),
    ("/monitor/drift", {"current_dataset_name": "a" * 32 + ".csv"}),
    ("/retrain", {"dataset_name": "a" * 32 + ".csv", "objective": "Classify churn"}),
])
def test_restricted_roles_cannot_bypass_ui(client, role, path, payload, monkeypatch):
    sign_in(client, role)
    monkeypatch.setenv("AUTODS_APPROVAL_TOKEN", "valid-token")
    response = client.post(path, json=payload, headers={"X-Approval-Token": "valid-token"})
    assert response.status_code == 403


def test_reviewer_cannot_upload_or_train(client):
    sign_in(client, "reviewer")
    assert client.post("/datasets", files={"file": ("data.csv", b"x,y\n1,0", "text/csv")}).status_code == 403
    assert client.post("/runs", json={"dataset_name": "a" * 32 + ".csv", "objective": "Classify churn"}).status_code == 403
    assert client.get("/workspace/overview").status_code == 200


def test_scientist_can_upload_dataset(client):
    sign_in(client)
    assert client.post("/datasets", files={"file": ("data.csv", b"x,y\n1,0", "text/csv")}).status_code == 200


def test_admin_user_management_changes_role_and_revokes_existing_session(client):
    create_user("admin@example.test", "Test Admin", "admin", "a-long-test-password")
    scientist = create_user("scientist@example.test", "Test Scientist", "scientist", "a-long-test-password")
    scientist_client = TestClient(app, headers={"X-AutoDS-Client": "web"})
    assert scientist_client.post("/auth/login", json={"email": "scientist@example.test", "password": "a-long-test-password"}).status_code == 200
    assert scientist_client.get("/auth/users").status_code == 403

    assert client.post("/auth/login", json={"email": "admin@example.test", "password": "a-long-test-password"}).status_code == 200
    settings = client.get("/auth/settings")
    assert settings.status_code == 200
    assert settings.json()["user_provisioning"] == "local_cli_only"
    users = client.get("/auth/users")
    assert users.status_code == 200
    assert all("password_hash" not in user for user in users.json()["items"])

    response = client.patch(f"/auth/users/{scientist['id']}", json={"role": "reviewer"})

    assert response.status_code == 200
    assert response.json()["changed"] is True
    assert response.json()["user"]["role"] == "reviewer"
    assert scientist_client.get("/auth/me").status_code == 401
    audit_events = client.get("/audit-logs").json()["items"]
    assert any(
        event["source"] == "workspace"
        and event["action"] == "workspace_user_updated"
        and event["resource_id"] == scientist["id"]
        for event in audit_events
    )
    with database() as connection:
        event = connection.execute(
            "SELECT actor, action, reason FROM workspace_audit_events "
            "WHERE resource_id = ? AND action = 'workspace_user_updated'",
            (scientist["id"],),
        ).fetchone()
    assert event["actor"] == "admin@example.test"
    assert event["action"] == "workspace_user_updated"
    assert "scientist to reviewer" in event["reason"]


def test_admin_cannot_change_their_own_role_or_active_state(client):
    admin = sign_in(client, "admin").json()
    response = client.patch(f"/auth/users/{admin['id']}", json={"active": False})
    assert response.status_code == 400
    assert "own role" in response.json()["detail"]


def test_session_expires_and_disabled_user_is_rejected(client):
    sign_in(client)
    with database() as connection:
        connection.execute("UPDATE sessions SET expires_at = ?", (time.time() - 1,))
        connection.commit()
    assert client.get("/auth/me").status_code == 401
    client.post("/auth/login", json={"email": "scientist@example.test", "password": "a-long-test-password"})
    with database() as connection:
        connection.execute("UPDATE users SET active = 0")
        connection.commit()
    assert client.get("/auth/me").status_code == 401


def test_csrf_header_and_origin_check(client):
    sign_in(client)
    client.headers.pop("X-AutoDS-Client")
    assert client.post("/auth/logout").status_code == 403
    assert client.post("/auth/login", json={"email": "scientist@example.test", "password": "a-long-test-password"}).status_code == 403
    client.headers["X-AutoDS-Client"] = "web"
    assert client.post("/auth/logout", headers={"Origin": "https://untrusted.example"}).status_code == 403
    assert client.post("/auth/logout", headers={"Origin": "http://testserver"}).status_code == 200


def test_login_throttling(client):
    for _ in range(5):
        assert client.post("/auth/login", json={"email": "unknown@example.test", "password": "wrong"}).status_code == 401
    assert client.post("/auth/login", json={"email": "unknown@example.test", "password": "wrong"}).status_code == 429


def test_unique_accounts_and_minimum_password():
    with pytest.raises(ValueError):
        create_user("admin@example.test", "Admin", "admin", "short")
    create_user("admin@example.test", "Admin", "admin", "long-test-password")
    with pytest.raises(ValueError):
        create_user("ADMIN@example.test", "Another Admin", "admin", "long-test-password")


def test_production_secure_cookie(client, monkeypatch):
    monkeypatch.setenv("AUTODS_COOKIE_SECURE", "true")
    assert "Secure" in sign_in(client).headers["set-cookie"]


def test_overview_is_empty_not_simulated(client):
    sign_in(client)
    response = client.get("/workspace/overview")
    assert response.json() == {"runs_count": 0, "review_count": 0, "failed_count": 0,
                               "production_available": False, "recent_runs": []}
    assert response.headers["Cache-Control"] == "no-store"


def test_legacy_promotion_records_authenticated_identity(client, monkeypatch):
    sign_in(client, "admin")
    monkeypatch.setenv("AUTODS_APPROVAL_TOKEN", "valid-token")
    from autods.registry import LocalModelRegistry
    recorded = {}
    def approve(self, proposal_id, approved_by):
        recorded.update(proposal_id=proposal_id, approved_by=approved_by)
        return recorded
    monkeypatch.setattr(LocalModelRegistry, "approve", approve)
    response = client.post("/promotions/" + "a" * 32 + "/approve", json={"approved_by": "Forged identity"}, headers={"X-Approval-Token": "valid-token"})
    assert response.status_code == 200
    assert recorded["approved_by"] == "admin@example.test"


def test_deployment_records_authenticated_identity(client, monkeypatch):
    sign_in(client, "admin")
    from autods.registry import LocalModelRegistry
    recorded = {}

    def deploy(self, proposal_id, deployed_by, deployment_note):
        recorded.update(
            proposal_id=proposal_id,
            deployed_by=deployed_by,
            deployment_note=deployment_note,
        )
        return recorded

    monkeypatch.setattr(LocalModelRegistry, "deploy", deploy)
    response = client.post(
        "/deployments/" + "a" * 32,
        json={"deployment_note": "Reviewed candidate released for the local demo."},
    )
    assert response.status_code == 200
    assert recorded["deployed_by"] == "admin@example.test"


def test_monitoring_persists_a_safe_report_with_authenticated_actor(client, tmp_path, monkeypatch):
    sign_in(client, "admin")
    from autods.registry import LocalModelRegistry

    dataset_name = "d" * 32 + ".csv"
    dataset_path = tmp_path / "datasets" / dataset_name
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("amount,plan\n10,basic\n10,basic\n", encoding="utf-8")
    monkeypatch.setattr(
        LocalModelRegistry,
        "production_baseline",
        lambda self: {
            "source_hash": "reference-hash",
            "numeric": {"amount": {"constant_value": 10.0}},
            "categorical": {"plan": {"basic": 1.0, "__autods_other_categories__": 0.0}},
        },
    )
    monkeypatch.setattr(
        LocalModelRegistry,
        "production_metadata",
        lambda self: {
            "metadata": {"run_id": "e" * 32},
            "deployment": {"version": "release-demo"},
        },
    )

    response = client.post(
        "/monitor/drift",
        json={"current_dataset_name": dataset_name, "drift_threshold": 0.20},
    )

    assert response.status_code == 200
    report = response.json()
    assert report["report_id"]
    assert report["monitored_by"] == "admin@example.test"
    assert report["model_version"] == "release-demo"
    assert report["monitoring_scope"] == "feature_drift"
    assert "report_path" not in report
    assert report["report"]["drift_detected"] is False

    listing = client.get("/monitoring/reports")
    assert listing.status_code == 200
    assert listing.json()[0]["report_id"] == report["report_id"]
    assert "report_path" not in listing.json()[0]
    detail = client.get(f"/monitoring/reports/{report['report_id']}")
    assert detail.status_code == 200
    assert detail.json()["dataset_name"] == dataset_name


def test_alerts_only_derive_open_persisted_events(client, monkeypatch):
    sign_in(client)
    import autods.api as module
    from autods.registry import LocalModelRegistry

    monkeypatch.setattr(
        module,
        "list_monitoring_reports",
        lambda: [
            {
                "report_id": "a" * 32,
                "created_at": "2026-10-13T10:00:00+00:00",
                "drift_detected": True,
                "drifted_columns": ["monthly_charges"],
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "list_runs",
        lambda: [
            {
                "run_id": "b" * 32,
                "started_at": "2026-10-13T09:00:00+00:00",
                "status": "failed",
                "objective": "Classify churn with F1.",
            }
        ],
    )
    monkeypatch.setattr(
        LocalModelRegistry,
        "list_proposals",
        lambda self: [
            {
                "proposal_id": "c" * 32,
                "status": "pending_human_approval",
                "created_at": "2026-10-13T08:00:00+00:00",
                "metadata": {"model_name": "random_forest"},
            }
        ],
    )

    response = client.get("/alerts")

    assert response.status_code == 200
    alerts = response.json()
    assert [alert["source"] for alert in alerts] == ["monitoring", "workflow", "governance"]
    assert alerts[0]["destination"] == "#monitoring"
    assert alerts[1]["severity"] == "critical"
    assert alerts[2]["destination"] == "#approval"
    assert all("path" not in alert for alert in alerts)


def test_audit_logs_only_derive_safe_persisted_evidence(client, monkeypatch):
    sign_in(client)
    import autods.api as module
    from autods.registry import LocalModelRegistry

    monkeypatch.setattr(module, "list_workspace_audit_events", lambda limit: [])

    monkeypatch.setattr(
        module,
        "list_monitoring_reports",
        lambda: [
            {
                "report_id": "a" * 32,
                "created_at": "2026-10-13T10:00:00+00:00",
                "monitored_by": "admin@example.test",
                "drift_detected": True,
                "drifted_columns": ["monthly_charges"],
            }
        ],
    )
    monkeypatch.setattr(
        module,
        "list_runs",
        lambda: [
            {
                "run_id": "b" * 32,
                "completed_at": "2026-10-13T09:00:00+00:00",
                "status": "failed",
                "objective": "Classify churn with F1.",
            }
        ],
    )
    monkeypatch.setattr(
        LocalModelRegistry,
        "list_proposals",
        lambda self: [
            {
                "proposal_id": "c" * 32,
                "status": "deployed",
                "metadata": {"model_name": "random_forest"},
                "audit_events": [
                    {
                        "at": "2026-10-13T11:00:00+00:00",
                        "action": "deployed",
                        "actor": "admin@example.test",
                        "reason": "Approved release for the local demonstration.",
                    }
                ],
            }
        ],
    )

    response = client.get("/audit-logs?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert [event["source"] for event in payload["items"]] == ["governance", "monitoring"]
    assert payload["items"][0]["actor"] == "admin@example.test"
    assert payload["items"][1]["destination"] == "#monitoring"
    assert all("path" not in event for event in payload["items"])


def _write_reviewable_proposal(root, proposal_id="b" * 32):
    folder = root / "pending" / proposal_id
    folder.mkdir(parents=True)
    (folder / "proposal.json").write_text(
        json.dumps(
            {
                "proposal_id": proposal_id,
                "status": "pending_human_approval",
                "created_at": "2026-10-12T00:00:00+00:00",
                "metadata": {"model_name": "logistic_regression", "metric": "f1"},
                "audit_events": [],
            }
        ),
        encoding="utf-8",
    )
    return proposal_id


def test_reviewer_can_record_a_review_without_promoting_a_model(client, tmp_path):
    sign_in(client, "reviewer")
    proposal_id = _write_reviewable_proposal(tmp_path / "models")

    response = client.post(
        f"/model-reviews/{proposal_id}",
        json={"decision": "approve", "reason": "The holdout evidence meets the documented gate."},
    )

    assert response.status_code == 200
    proposal = response.json()
    assert proposal["status"] == "approved_for_deployment"
    assert proposal["review"]["reviewed_by"] == "reviewer@example.test"
    assert proposal["audit_events"][-1]["action"] == "approved_for_deployment"
    assert not (tmp_path / "models" / "production").exists()
    all_proposals = client.get("/models/proposals").json()
    assert all_proposals[0]["proposal_id"] == proposal_id


def test_reviewer_can_reject_a_model_with_a_recorded_reason(client, tmp_path):
    sign_in(client, "reviewer")
    proposal_id = _write_reviewable_proposal(tmp_path / "models", proposal_id="c" * 32)

    response = client.post(
        f"/model-reviews/{proposal_id}",
        json={"decision": "reject", "reason": "The holdout result needs a clearer error analysis."},
    )

    assert response.status_code == 200
    proposal = response.json()
    assert proposal["status"] == "rejected"
    assert proposal["review"]["reviewed_by"] == "reviewer@example.test"
    assert proposal["review"]["reason"] == "The holdout result needs a clearer error analysis."
    assert proposal["audit_events"][-1]["action"] == "rejected"
    assert not (tmp_path / "models" / "production").exists()


def test_scientist_cannot_record_a_model_review(client, tmp_path):
    sign_in(client, "scientist")
    proposal_id = _write_reviewable_proposal(tmp_path / "models")
    response = client.post(
        f"/model-reviews/{proposal_id}",
        json={"decision": "reject", "reason": "The validation evidence needs revision."},
    )
    assert response.status_code == 403
