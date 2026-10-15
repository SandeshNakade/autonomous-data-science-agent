"""Deterministic, bounded CSV profiling used as the agent's factual context."""

from __future__ import annotations

import hashlib
import re
import csv
import warnings
from pathlib import Path

import pandas as pd

from .schemas import ColumnProfile, DatasetProfile

MAX_DATASET_BYTES = 200 * 1024 * 1024
MAX_DATASET_ROWS = 1_000_000
MAX_DATASET_COLUMNS = 500
TARGET_HINTS = {"target", "label", "class", "outcome", "response", "y", "churn"}
IDENTIFIER_HINTS = {"id", "uuid", "identifier", "email", "phone", "name"}


class DatasetError(ValueError):
    """A CSV does not satisfy the prototype's safe input contract."""


def read_csv_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise DatasetError("Only .csv datasets are accepted")
    if not path.is_file():
        raise DatasetError(f"Dataset does not exist: {path}")
    if path.stat().st_size > MAX_DATASET_BYTES:
        raise DatasetError("Dataset exceeds the 200 MB prototype limit")
    try:
        # Check the original header before pandas can rename duplicate/empty
        # names. Keep names intact so the profile and training use the same schema.
        with path.open(encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source, strict=True), [])
        if len(header) < 2:
            raise DatasetError("Dataset must have at least one row and two columns")
        if len(header) > MAX_DATASET_COLUMNS:
            raise DatasetError("Dataset exceeds the prototype limit of 500 columns")
        if any(not name.strip() for name in header):
            raise DatasetError("Every column must have a non-empty name")
        if len(set(header)) != len(header):
            raise DatasetError("Dataset contains duplicate column names")
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            data = pd.read_csv(path, encoding="utf-8-sig", nrows=MAX_DATASET_ROWS + 1, index_col=False)
    except DatasetError:
        raise
    except Exception as exc:  # pandas presents useful parser diagnostics
        raise DatasetError(f"Could not read CSV: {exc}") from exc
    if data.empty or data.shape[1] < 2:
        raise DatasetError("Dataset must have at least one row and two columns")
    if len(data) > MAX_DATASET_ROWS or data.shape[1] > MAX_DATASET_COLUMNS:
        raise DatasetError("Dataset exceeds the prototype limit of 1,000,000 rows or 500 columns")
    if data.columns.duplicated().any():
        raise DatasetError("Dataset contains duplicate column names")
    return data


def dataset_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inferred_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def _likely_identifier(name: str, unique_fraction: float, inferred_type: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
    # Do not mistake a continuous numerical feature (price, age, sensor value)
    # for an ID solely because nearly every value is distinct. Name hints are
    # always respected; high cardinality is an ID signal only for non-numeric
    # columns in this small prototype.
    return bool(tokens & IDENTIFIER_HINTS) or (
        inferred_type in {"categorical", "datetime"} and unique_fraction > 0.98
    )


def profile_dataset(path: str | Path) -> DatasetProfile:
    """Return compact, serialisable facts; no model or code is executed here."""

    data = read_csv_dataset(path)
    return profile_frame(data, path)


def profile_frame(data: pd.DataFrame, path: str | Path) -> DatasetProfile:
    """Profile an already validated frame without parsing the CSV a second time."""
    rows = len(data)
    columns: list[ColumnProfile] = []
    target_candidates: list[str] = []
    for name in data.columns:
        series = data[name]
        unique_count = int(series.nunique(dropna=True))
        unique_fraction = unique_count / rows if rows else 0.0
        inferred_type = _inferred_type(series)
        profile = ColumnProfile(
            name=str(name),
            inferred_type=inferred_type,
            pandas_dtype=str(series.dtype),
            missing_fraction=round(float(series.isna().mean()), 6),
            unique_count=unique_count,
            unique_fraction=round(unique_fraction, 6),
            likely_identifier=_likely_identifier(str(name), unique_fraction, inferred_type),
        )
        columns.append(profile)
        if str(name).lower() in TARGET_HINTS:
            target_candidates.append(str(name))
    return DatasetProfile(
        dataset_path=str(Path(path).resolve()),
        data_hash=dataset_hash(path),
        rows=rows,
        columns=columns,
        likely_targets=target_candidates,
    )
