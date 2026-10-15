"""A small local registry with explicit human approval as the only promotion path."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .training import ModelBundle

_SAFE_PROPOSAL_ID = re.compile(r"^[a-f0-9]{32}$")
_REGISTRY_LOCK = threading.RLock()
_REVIEWABLE_STATUSES = {"pending_human_approval"}
_DEPLOYABLE_STATUSES = {"approved_for_deployment"}


class RegistryError(ValueError):
    pass


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


class LocalModelRegistry:
    """Stores candidates as pending until a person explicitly approves one."""

    def __init__(self, root: str | Path = "models") -> None:
        self.root = Path(root)
        self.pending_root = self.root / "pending"
        self.production_root = self.root / "production"
        self.releases_root = self.root / "releases"

    def create_proposal(
        self,
        bundle: ModelBundle,
        metadata: dict[str, Any],
        monitoring_baseline: dict[str, Any] | None = None,
    ) -> str:
        proposal_id = uuid.uuid4().hex
        folder = self.pending_root / proposal_id
        folder.mkdir(parents=True, exist_ok=False)
        joblib.dump(bundle, folder / "model.joblib")
        _write_json(
            folder / "proposal.json",
            {
                "proposal_id": proposal_id,
                "status": "pending_human_approval",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata,
                "audit_events": [
                    {
                        "at": datetime.now(timezone.utc).isoformat(),
                        "action": "submitted_for_review",
                        "actor": "system",
                        "reason": "Candidate passed the controlled workflow quality gate.",
                    }
                ],
            },
        )
        if monitoring_baseline is not None:
            _write_json(folder / "baseline.json", monitoring_baseline)
        return proposal_id

    def list_proposals(self) -> list[dict[str, Any]]:
        """Return review metadata only; fitted candidate artifacts stay private."""

        if not self.pending_root.is_dir():
            return []
        proposals: list[dict[str, Any]] = []
        for folder in self.pending_root.iterdir():
            metadata_file = folder / "proposal.json"
            if not metadata_file.is_file():
                continue
            try:
                proposal = json.loads(metadata_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not _SAFE_PROPOSAL_ID.fullmatch(str(proposal.get("proposal_id", ""))):
                continue
            proposals.append(proposal)
        return sorted(proposals, key=lambda item: item.get("created_at", ""), reverse=True)

    def review(
        self,
        proposal_id: str,
        decision: str,
        reviewed_by: str,
        reason: str,
    ) -> dict[str, Any]:
        """Record a human review without promoting or deploying a model."""

        if not _SAFE_PROPOSAL_ID.fullmatch(proposal_id):
            raise RegistryError("Invalid proposal identifier")
        if decision not in {"approve", "reject"}:
            raise RegistryError("Review decision must be 'approve' or 'reject'")
        if not reviewed_by.strip():
            raise RegistryError("The reviewer identity is required")
        if not reason.strip():
            raise RegistryError("A review reason is required")
        with _REGISTRY_LOCK:
            metadata_file = self.pending_root / proposal_id / "proposal.json"
            if not metadata_file.is_file():
                raise RegistryError("Pending proposal does not exist")
            proposal = json.loads(metadata_file.read_text(encoding="utf-8"))
            if proposal.get("status") not in _REVIEWABLE_STATUSES:
                raise RegistryError(
                    f"Proposal cannot be reviewed from state '{proposal.get('status', 'unknown')}'"
                )
            reviewed_at = datetime.now(timezone.utc).isoformat()
            action = "approved_for_deployment" if decision == "approve" else "rejected"
            proposal["status"] = action
            proposal["review"] = {
                "decision": decision,
                "reviewed_by": reviewed_by.strip(),
                "reviewed_at": reviewed_at,
                "reason": reason.strip(),
            }
            audit_events = proposal.setdefault("audit_events", [])
            audit_events.append(
                {
                    "at": reviewed_at,
                    "action": action,
                    "actor": reviewed_by.strip(),
                    "reason": reason.strip(),
                }
            )
            _write_json(metadata_file, proposal)
            return proposal

    def approve(self, proposal_id: str, approved_by: str) -> dict[str, Any]:
        """Promote exactly one pending artifact after an explicit user action."""

        if not _SAFE_PROPOSAL_ID.fullmatch(proposal_id):
            raise RegistryError("Invalid proposal identifier")
        if not approved_by.strip():
            raise RegistryError("The approver identity is required")
        with _REGISTRY_LOCK:
            folder = self.pending_root / proposal_id
            metadata_file = folder / "proposal.json"
            model_file = folder / "model.joblib"
            if not metadata_file.is_file() or not model_file.is_file():
                raise RegistryError("Pending proposal does not exist")
            proposal = json.loads(metadata_file.read_text(encoding="utf-8"))
            if proposal["status"] != "pending_human_approval":
                raise RegistryError(f"Proposal cannot be approved from state '{proposal['status']}'")
            self.production_root.mkdir(parents=True, exist_ok=True)
            # The active model is replaced atomically, under a process-wide lock.
            # Candidate bytes are written to a staging file before the switch.
            _atomic_copy(model_file, self.production_root / "model.joblib")
            baseline_file = folder / "baseline.json"
            if baseline_file.is_file():
                _atomic_copy(baseline_file, self.production_root / "baseline.json")
            proposal["status"] = "promoted"
            proposal["approved_by"] = approved_by.strip()
            proposal["approved_at"] = datetime.now(timezone.utc).isoformat()
            _write_json(metadata_file, proposal)
            _write_json(self.production_root / "metadata.json", proposal)
            return proposal

    def deploy(
        self,
        proposal_id: str,
        deployed_by: str,
        deployment_note: str,
    ) -> dict[str, Any]:
        """Deploy exactly one reviewer-approved candidate and retain its release copy."""

        if not _SAFE_PROPOSAL_ID.fullmatch(proposal_id):
            raise RegistryError("Invalid proposal identifier")
        if not deployed_by.strip():
            raise RegistryError("The deployment actor is required")
        if len(deployment_note.strip()) < 3:
            raise RegistryError("A deployment note with at least 3 characters is required")
        with _REGISTRY_LOCK:
            folder = self.pending_root / proposal_id
            metadata_file = folder / "proposal.json"
            model_file = folder / "model.joblib"
            if not metadata_file.is_file() or not model_file.is_file():
                raise RegistryError("Approved candidate does not exist")
            proposal = json.loads(metadata_file.read_text(encoding="utf-8"))
            if proposal.get("status") not in _DEPLOYABLE_STATUSES:
                raise RegistryError(
                    "Only a reviewer-approved candidate can be deployed; "
                    f"current state is '{proposal.get('status', 'unknown')}'"
                )
            try:
                bundle: ModelBundle = joblib.load(model_file)
            except Exception as exc:
                raise RegistryError("Approved candidate artifact could not be loaded safely") from exc
            if not isinstance(bundle, ModelBundle):
                raise RegistryError("Approved candidate artifact has an unexpected format")

            deployed_at = datetime.now(timezone.utc).isoformat()
            version = f"release-{proposal_id[:8]}"
            class_labels = (
                [str(label) for label in bundle.target_encoder.classes_]
                if bundle.target_encoder is not None
                else []
            )
            deployment = {
                "version": version,
                "deployed_by": deployed_by.strip(),
                "deployed_at": deployed_at,
                "deployment_note": deployment_note.strip(),
                "feature_columns": list(bundle.feature_columns),
                "input_schema": proposal.get("metadata", {}).get("input_schema", []),
                "task_type": bundle.task_type,
                "metric": bundle.metric,
                "class_labels": class_labels,
            }
            proposal["status"] = "deployed"
            proposal["deployment"] = deployment
            proposal.setdefault("audit_events", []).append(
                {
                    "at": deployed_at,
                    "action": "deployed",
                    "actor": deployed_by.strip(),
                    "reason": deployment_note.strip(),
                }
            )

            # Each release remains immutable under releases/ while production is
            # atomically switched to the selected reviewer-approved artifact.
            release_root = self.releases_root / proposal_id
            if release_root.exists():
                raise RegistryError("This approved candidate already has a release record")
            _atomic_copy(model_file, release_root / "model.joblib")
            baseline_file = folder / "baseline.json"
            if baseline_file.is_file():
                _atomic_copy(baseline_file, release_root / "baseline.json")
                _atomic_copy(baseline_file, self.production_root / "baseline.json")
            _write_json(release_root / "metadata.json", proposal)
            _atomic_copy(release_root / "model.joblib", self.production_root / "model.joblib")
            _write_json(self.production_root / "metadata.json", proposal)
            _write_json(metadata_file, proposal)
            return proposal

    def list_deployments(self) -> list[dict[str, Any]]:
        """Return read-only release metadata, never fitted model bytes."""

        if not self.releases_root.is_dir():
            return []
        releases: list[dict[str, Any]] = []
        for folder in self.releases_root.iterdir():
            metadata_file = folder / "metadata.json"
            if not metadata_file.is_file():
                continue
            try:
                release = json.loads(metadata_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not _SAFE_PROPOSAL_ID.fullmatch(str(release.get("proposal_id", ""))):
                continue
            releases.append(release)
        return sorted(
            releases,
            key=lambda item: item.get("deployment", {}).get("deployed_at", ""),
            reverse=True,
        )

    def deployment_status(self) -> dict[str, Any]:
        """Describe the local serving state without executing predictions."""

        releases = self.list_deployments()
        try:
            active = self.production_metadata()
        except RegistryError:
            active = None
        return {
            "endpoint": "/predict",
            "method": "POST",
            "state": "ready" if active is not None else "not_deployed",
            "active": active,
            "releases": releases,
        }

    def load_production(self) -> ModelBundle:
        with _REGISTRY_LOCK:
            model_path = self.production_root / "model.joblib"
            if not model_path.is_file():
                raise RegistryError("No approved production model is available")
            return joblib.load(model_path)

    def production_metadata(self) -> dict[str, Any]:
        with _REGISTRY_LOCK:
            path = self.production_root / "metadata.json"
            if not path.is_file():
                raise RegistryError("No approved production model metadata is available")
            return json.loads(path.read_text(encoding="utf-8"))

    def production_baseline(self) -> dict[str, Any]:
        with _REGISTRY_LOCK:
            path = self.production_root / "baseline.json"
            if not path.is_file():
                raise RegistryError("No monitoring baseline is associated with the production model")
            return json.loads(path.read_text(encoding="utf-8"))
