"""Small file-backed dataset library; immutable CSVs with derived metadata."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .profiling import DatasetError, profile_frame, read_csv_dataset

DATASET_ID = re.compile(r"[a-f0-9]{32}\.csv")


def _quality(data: pd.DataFrame, profile: dict) -> dict:
    missing = int(data.isna().sum().sum())
    duplicates = int(data.duplicated().sum())
    empty = [c["name"] for c in profile["columns"] if c["unique_count"] == 0]
    constant = [c["name"] for c in profile["columns"] if c["unique_count"] == 1]
    identifiers = [c["name"] for c in profile["columns"] if c["likely_identifier"]]
    infinite = int(np.isinf(data.select_dtypes(include="number").to_numpy(dtype=float)).sum())
    checks = [
        {"code": "structure", "label": "CSV structure", "status": "passed",
         "message": "Valid UTF-8 CSV within the row, column, and file-size limits.", "count": 0, "columns": []},
    ]
    for code, label, count, columns, message in [
        ("missing", "Missing values", missing, [c["name"] for c in profile["columns"] if c["missing_fraction"] > 0], "Missing cells need a target-aware imputation or exclusion strategy."),
        ("duplicates", "Duplicate rows", duplicates, [], "Repeated rows after the first occurrence; review before choosing the split strategy."),
        ("empty", "Empty columns", len(empty), empty, "These columns have no observed values."),
        ("constant", "Constant columns", len(constant), constant, "These columns contain only one distinct non-missing value."),
        ("infinite", "Non-finite numeric values", infinite, [], "Infinite numeric values need handling before model training."),
    ]:
        checks.append({"code": code, "label": label, "count": count, "columns": columns,
                       "status": "warning" if count else "passed",
                       "message": message if count else "No issues detected in this check."})
    checks.append({"code": "identifiers", "label": "Potential identifiers", "count": len(identifiers),
                   "columns": identifiers, "status": "info" if identifiers else "passed",
                   "message": "Heuristic suggestions, not confirmed leakage. Review these columns during planning." if identifiers else "No likely identifier columns detected."})
    total_cells = int(data.size)
    return {"status": "warnings" if any(c["status"] == "warning" for c in checks) else "passed",
            "missing_cells": missing, "total_cells": total_cells,
            "completeness": round(100 * (total_cells - missing) / total_cells, 2),
            "duplicate_rows": duplicates, "checks": checks}


class DatasetLibrary:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def path(self, name: str) -> Path:
        if not DATASET_ID.fullmatch(name):
            raise DatasetError("Use a dataset identifier issued by the upload API")
        path = (self.root / name).resolve()
        if path.parent != self.root or not path.is_file():
            raise FileNotFoundError("Dataset was not found")
        return path

    def _metadata(self, path: Path) -> dict | None:
        metadata_path = path.with_suffix(".json")
        # Do not follow a manually placed metadata symlink outside the library.
        if metadata_path.resolve().parent != self.root:
            return None
        try:
            result = json.loads(metadata_path.read_text(encoding="utf-8"))
            if result.get("modified_ns") == path.stat().st_mtime_ns and result.get("dataset_name") == path.name:
                return result
        except (OSError, ValueError, AttributeError):
            pass
        return None

    def add(self, content: bytes, original_name: str, uploaded_by: str) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.csv"
        display_name = re.split(r"[/\\]", original_name)[-1][:200]
        with tempfile.TemporaryDirectory(prefix=".upload-", dir=self.root) as temporary:
            staged = Path(temporary) / name
            staged.write_bytes(content)
            report = self._inspect(staged)
            report.update(dataset_name=name, display_name=display_name,
                          uploaded_at=datetime.now(timezone.utc).isoformat(), uploaded_by=uploaded_by)
            metadata = Path(temporary) / f"{Path(name).stem}.json"
            metadata.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
            # CSV is published last: invalid/incomplete uploads never appear in
            # the library and existing datasets are never overwritten.
            os.replace(metadata, self.root / metadata.name)
            os.replace(staged, self.root / name)
        return report

    def _inspect(self, path: Path) -> dict:
        data = read_csv_dataset(path)
        profile = profile_frame(data, path).to_dict()
        profile.pop("dataset_path", None)
        for column in profile["columns"]:
            column["missing_count"] = int(data[column["name"]].isna().sum())
        return {"dataset_name": path.name, "display_name": f"Existing dataset · {path.stem[:8]}",
                "uploaded_at": None, "uploaded_by": None,
                "size_bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns,
                "profile": profile, "quality": _quality(data, profile)}

    def inspect(self, name: str) -> dict:
        path = self.path(name)
        return self._metadata(path) or self._inspect(path)

    def list(self, search: str = "", offset: int = 0, limit: int = 20) -> dict:
        items = []
        for path in self.root.glob("*.csv"):
            if not DATASET_ID.fullmatch(path.name) or path.resolve().parent != self.root:
                continue
            metadata = self._metadata(path)
            item = {"dataset_name": path.name,
                    "display_name": metadata["display_name"] if metadata else f"Existing dataset · {path.stem[:8]}",
                    "uploaded_at": metadata.get("uploaded_at") if metadata else None,
                    "size_bytes": path.stat().st_size,
                    "rows": metadata["profile"]["rows"] if metadata else None,
                    "columns": len(metadata["profile"]["columns"]) if metadata else None,
                    "quality_status": metadata["quality"]["status"] if metadata else "not_profiled"}
            if search.casefold() in (item["display_name"] + " " + path.name).casefold():
                items.append((path.stat().st_mtime_ns, item))
        items.sort(key=lambda pair: (pair[0], pair[1]["dataset_name"]), reverse=True)
        return {"items": [item for _, item in items[offset:offset + limit]], "total": len(items), "offset": offset, "limit": limit}

    def preview(self, name: str, offset: int, limit: int) -> dict:
        path = self.path(name)
        report = self.inspect(name)
        # Chunk iteration preserves quoted/multiline CSV fields. Never use line
        # offsets or skiprows for pagination because records can span lines.
        records = []
        seen = 0
        with pd.read_csv(path, encoding="utf-8-sig", chunksize=1000) as reader:
            for chunk in reader:
                if seen + len(chunk) > offset:
                    start = max(0, offset - seen)
                    for row in chunk.iloc[start:start + limit - len(records)].itertuples(index=False, name=None):
                        records.append([_preview_cell(value) for value in row])
                seen += len(chunk)
                if len(records) == limit:
                    break
        return {"columns": [c["name"] for c in report["profile"]["columns"]],
                "rows": records, "offset": offset, "limit": limit, "total_rows": report["profile"]["rows"],
                "cell_character_limit": 200}


def _preview_cell(value):
    if pd.isna(value) or (isinstance(value, (float, np.floating)) and not math.isfinite(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        # Preserve large IDs exactly; JavaScript numbers cannot represent them.
        return int(value) if abs(value) <= 2**53 - 1 else str(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    text = str(value)
    return text if len(text) <= 200 else text[:199] + "…"
