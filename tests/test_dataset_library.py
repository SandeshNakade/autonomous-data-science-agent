"""Stage 2 API tests: real permissions, immutable files, validation, and previews."""

import hashlib

import pytest
from fastapi.testclient import TestClient

from autods.api import app
from autods.auth import create_user


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    import autods.api as module
    root = tmp_path / "datasets"
    root.mkdir()
    monkeypatch.setattr(module, "DATASET_ROOT", root)
    monkeypatch.setenv("AUTODS_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    monkeypatch.delenv("AUTODS_COOKIE_SECURE", raising=False)
    return root


@pytest.fixture
def client(workspace):
    create_user("scientist@example.test", "Dataset Tester", "scientist", "dataset-test-password")
    with TestClient(app, headers={"X-AutoDS-Client": "web"}) as browser:
        assert browser.post("/auth/login", json={"email": "scientist@example.test", "password": "dataset-test-password"}).status_code == 200
        yield browser


def upload(client, content=b"amount,churn\n10,yes\n20,no\n", name="churn.csv"):
    return client.post("/datasets", files={"file": (name, content, "text/csv")})


def test_upload_profiles_preserves_and_attributes(client, workspace):
    content = b"amount,churn\n10,yes\n20,no\n"
    response = upload(client, content)
    assert response.status_code == 200
    report = response.json()
    assert report["display_name"] == "churn.csv"
    assert report["uploaded_by"] == "scientist@example.test"
    assert report["uploaded_at"]
    assert report["profile"]["rows"] == 2
    assert len(report["profile"]["columns"]) == 2
    assert report["profile"]["data_hash"] == hashlib.sha256(content).hexdigest()
    assert report["profile"]["likely_targets"] == ["churn"]
    assert "dataset_path" not in report["profile"]
    assert report["quality"]["status"] == "passed"
    assert (workspace / report["dataset_name"]).read_bytes() == content
    assert client.get(f"/datasets/{report['dataset_name']}").json() == {key: value for key, value in report.items() if key != "message"}
    duplicate_upload = upload(client, content).json()
    assert duplicate_upload["dataset_name"] != report["dataset_name"]
    assert duplicate_upload["profile"]["data_hash"] == report["profile"]["data_hash"]
    assert len(list(workspace.glob("*.csv"))) == 2


@pytest.mark.parametrize("content", [
    b"a,a\n1,2\n", b",b\n1,2\n", b" ,b\n1,2\n", b"a,b\n",
    b"single\n1\n", b"a,b\n1,2,3\n", b"a,b\n1,2\n3,4,5\n",
    b'a,b\n"unclosed,2\n', b"a,b\n\xff,2\n",
])
def test_invalid_csv_never_published(client, workspace, content):
    response = upload(client, content)
    assert response.status_code == 400, response.text
    assert not list(workspace.iterdir())


def test_limits_reject_before_publication(client, workspace, monkeypatch):
    import autods.profiling as profiling
    monkeypatch.setattr(profiling, "MAX_DATASET_ROWS", 1)
    assert upload(client).status_code == 400
    assert not list(workspace.iterdir())
    monkeypatch.setattr(profiling, "MAX_DATASET_COLUMNS", 1)
    assert upload(client).status_code == 400
    import autods.api as module
    monkeypatch.setattr(module, "MAX_UPLOAD_BYTES", 5)
    assert upload(client).status_code == 413


def test_quality_counts_are_explained_not_an_arbitrary_score(client):
    content = b"customer_id,value,empty,constant,target\nc1,1,,same,yes\nc1,1,,same,yes\nc2,,,same,no\nc3,inf,,same,no\n"
    report = upload(client, content).json()
    quality = report["quality"]
    checks = {check["code"]: check for check in quality["checks"]}
    assert quality["status"] == "warnings"
    assert quality["missing_cells"] == 5
    assert quality["total_cells"] == 20
    assert quality["completeness"] == 75
    assert quality["duplicate_rows"] == 1
    assert checks["empty"]["columns"] == ["empty"]
    assert checks["constant"]["columns"] == ["constant"]
    assert checks["infinite"]["count"] == 1
    assert "customer_id" in checks["identifiers"]["columns"]
    assert checks["identifiers"]["status"] == "info"
    columns = {column["name"]: column for column in report["profile"]["columns"]}
    assert columns["empty"]["missing_count"] == 4


def test_preview_pagination_preserves_multiline_records(client):
    content = ('text,target\n' + ''.join(f'"line {i}\nsecond line",{i % 2}\n' for i in range(25))).encode()
    name = upload(client, content).json()["dataset_name"]
    preview = client.get(f"/datasets/{name}/preview?offset=20&limit=5").json()
    assert preview["columns"] == ["text", "target"]
    assert preview["rows"][0] == ["line 20\nsecond line", 0]
    assert len(preview["rows"]) == 5 and preview["total_rows"] == 25
    assert client.get(f"/datasets/{name}/preview?offset=100").json()["rows"] == []
    assert client.get(f"/datasets/{name}/preview?limit=51").status_code == 422
    assert client.get(f"/datasets/{name}/preview?offset=-1").status_code == 422


def test_preview_safe_nulls_long_values_large_ids_and_literal_text(client):
    text = "x" * 210
    report = upload(client, f"id,text,amount\n9007199254740993,{text},inf\n9007199254740994,<script>alert(1)</script>,\n".encode()).json()
    response = client.get(f"/datasets/{report['dataset_name']}/preview")
    assert response.status_code == 200
    preview = response.json()
    assert preview["rows"][0][0] == "9007199254740993"
    assert len(preview["rows"][0][1]) == 200
    assert preview["rows"][0][2] is None and preview["rows"][1][2] is None
    assert preview["rows"][1][1] == "<script>alert(1)</script>"


def test_list_search_and_pagination(client):
    upload(client, name="first.csv")
    upload(client, name="second.csv")
    listing = client.get("/datasets?limit=1").json()
    assert listing["total"] == 2 and len(listing["items"]) == 1
    assert listing["items"][0]["display_name"] == "second.csv"
    assert listing["items"][0]["rows"] == 2
    assert client.get("/datasets?offset=1&limit=1").json()["items"][0]["display_name"] == "first.csv"
    assert client.get("/datasets?search=FIRST").json()["total"] == 1
    assert client.get("/datasets?search=missing").json()["items"] == []
    assert client.get("/datasets?limit=10000").status_code == 422


def test_legacy_uuid_uploads_readable_without_modifying_them(client, workspace):
    name = "a" * 32 + ".csv"
    (workspace / name).write_bytes(b"feature,label\n1,0\n2,1\n")
    (workspace / "manually-copied.csv").write_bytes(b"feature,label\n1,0\n")
    item = client.get("/datasets").json()["items"][0]
    assert item["quality_status"] == "not_profiled"
    assert item["rows"] is None
    report = client.get(f"/datasets/{name}").json()
    assert report["profile"]["rows"] == 2
    assert report["uploaded_at"] is None
    assert not (workspace / name).with_suffix(".json").exists()
    assert client.get("/datasets/manually-copied.csv").status_code == 400


def test_modified_files_do_not_reuse_stale_profile(client, workspace):
    report = upload(client).json()
    (workspace / report["dataset_name"]).write_bytes(b"a,b\n1,0\n2,1\n3,0\n")
    detail = client.get(f"/datasets/{report['dataset_name']}").json()
    assert detail["profile"]["rows"] == 3
    assert detail["profile"]["data_hash"] != report["profile"]["data_hash"]
    assert client.get("/datasets").json()["items"][0]["quality_status"] == "not_profiled"


def test_reviewer_read_only_and_guest_blocked(client, workspace):
    name = upload(client).json()["dataset_name"]
    create_user("reviewer@example.test", "Reviewer", "reviewer", "dataset-test-password")
    assert client.post("/auth/login", json={"email": "reviewer@example.test", "password": "dataset-test-password"}).status_code == 200
    assert client.get("/datasets").status_code == 200
    assert client.get(f"/datasets/{name}").status_code == 200
    assert client.get(f"/datasets/{name}/preview").status_code == 200
    assert upload(client).status_code == 403
    with TestClient(app) as guest:
        for path in ["/datasets", f"/datasets/{name}", f"/datasets/{name}/preview"]:
            assert guest.get(path).status_code == 401


def test_missing_and_invalid_dataset_ids(client):
    assert client.get("/datasets/not-issued.csv").status_code == 400
    assert client.get("/datasets/" + "0" * 32 + ".csv").status_code == 404


def test_bom_header_and_windows_filename(client):
    response = upload(client, b"\xef\xbb\xbfa,b\n1,0\n2,1\n", name="C:\\fakepath\\sample.csv")
    assert response.status_code == 200
    assert response.json()["display_name"] == "sample.csv"
    assert response.json()["profile"]["columns"][0]["name"] == "a"
