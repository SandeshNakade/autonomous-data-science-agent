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
            },
        )
        if monitoring_baseline is not None:
            _write_json(folder / "baseline.json", monitoring_baseline)
        return proposal_id

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
