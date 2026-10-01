"""Batch drift checks and a retraining gate that can only request approval."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .profiling import dataset_hash, read_csv_dataset

OTHER_CATEGORY = "__autods_other_categories__"
MAX_TRACKED_CATEGORIES = 24


@dataclass(frozen=True)
class DriftBaseline:
    source_hash: str
    numeric: dict[str, dict[str, Any]]
    categorical: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriftReport:
    current_hash: str
    numeric_psi: dict[str, float | None]
    categorical_tvd: dict[str, float | None]
    drifted_columns: list[str]
    missing_columns: list[str]
    threshold: float

    @property
    def drift_detected(self) -> bool:
        return bool(self.drifted_columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "drift_detected": self.drift_detected,
        }


def _distribution(values: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Histogram with open-ended outer bins, so out-of-range shifts count."""

    indices = np.searchsorted(cuts, values, side="right")
    counts = np.bincount(indices, minlength=len(cuts) + 1)
    probabilities = counts.astype(float) + 1e-6
    return probabilities / probabilities.sum()


def build_drift_baseline(
    dataset_path: str | Path,
    excluded_columns: tuple[str, ...] = (),
) -> DriftBaseline:
    """Persist compact feature distributions rather than raw reference records."""

    data = read_csv_dataset(dataset_path).drop(columns=list(excluded_columns), errors="ignore")
    numeric: dict[str, dict[str, Any]] = {}
    categorical: dict[str, dict[str, float]] = {}
    for column in data.columns:
        series = data[column]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            if np.nanmin(values) == np.nanmax(values):
                numeric[column] = {"constant_value": float(values[0])}
                continue
            # Internal deciles give one open-ended bin on each side.  Unlike a
            # min/max histogram, a production batch beyond the old range is
            # therefore visible instead of being silently discarded.
            cuts = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
            if len(cuts) < 1:
                continue
            numeric[column] = {
                "cuts": cuts.tolist(),
                "distribution": _distribution(values, cuts).tolist(),
            }
        else:
            values = series.fillna("__missing__").astype(str)
            distribution = values.value_counts(normalize=True)
            retained = distribution.head(MAX_TRACKED_CATEGORIES)
            payload = {str(key): float(value) for key, value in retained.items()}
            payload[OTHER_CATEGORY] = float(1.0 - retained.sum())
            categorical[column] = payload
    return DriftBaseline(source_hash=dataset_hash(dataset_path), numeric=numeric, categorical=categorical)


def compare_to_baseline(
    baseline: DriftBaseline,
    dataset_path: str | Path,
    threshold: float = 0.20,
) -> DriftReport:
    """Compare a batch to saved feature summaries using PSI/TVD, not retraining."""

    if not 0 < threshold <= 1:
        raise ValueError("Drift threshold must be between 0 and 1")
    current = read_csv_dataset(dataset_path)
    numeric_psi: dict[str, float | None] = {}
    categorical_tvd: dict[str, float | None] = {}
    drifted: list[str] = []
    missing: list[str] = []
    for column, reference in baseline.numeric.items():
        if column not in current.columns:
            numeric_psi[column] = None
            drifted.append(column)
            missing.append(column)
            continue
        values = pd.to_numeric(current[column], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(values):
            numeric_psi[column] = None
            drifted.append(column)
            missing.append(column)
            continue
        if "constant_value" in reference:
            changed = float(
                np.mean(~np.isclose(values, float(reference["constant_value"]), rtol=1e-9, atol=1e-12))
            )
            numeric_psi[column] = round(changed, 6)
            if changed >= threshold:
                drifted.append(column)
            continue
        expected = np.asarray(reference["distribution"], dtype=float)
        actual = _distribution(values, np.asarray(reference["cuts"], dtype=float))
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        numeric_psi[column] = round(psi, 6)
        if psi >= threshold:
            drifted.append(column)
    for column, reference in baseline.categorical.items():
        if column not in current.columns:
            categorical_tvd[column] = None
            drifted.append(column)
            missing.append(column)
            continue
        values = current[column].fillna("__missing__").astype(str)
        observed = values.value_counts(normalize=True)
        tracked = set(reference) - {OTHER_CATEGORY}
        mapped = {key: float(observed.get(key, 0.0)) for key in tracked}
        mapped[OTHER_CATEGORY] = float(
            sum(float(value) for key, value in observed.items() if str(key) not in tracked)
        )
        distance = 0.5 * sum(
            abs(reference.get(key, 0.0) - mapped.get(key, 0.0))
            for key in set(reference) | set(mapped)
        )
        categorical_tvd[column] = round(distance, 6)
        if distance >= threshold:
            drifted.append(column)
    return DriftReport(
        current_hash=dataset_hash(dataset_path),
        numeric_psi=numeric_psi,
        categorical_tvd=categorical_tvd,
        drifted_columns=sorted(set(drifted)),
        missing_columns=sorted(set(missing)),
        threshold=threshold,
    )


def save_baseline(baseline: DriftBaseline, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(baseline.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_baseline(path: str | Path) -> DriftBaseline:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DriftBaseline(**payload)


def retraining_request(report: DriftReport, source_run_id: str | None = None) -> dict[str, Any]:
    """Return an auditable request; this function intentionally never runs training."""

    if not report.drift_detected:
        return {
            "state": "monitoring_normal",
            "source_run_id": source_run_id,
            "reason": "No monitored feature exceeded the drift threshold.",
        }
    return {
        "state": "retraining_requested_pending_human_approval",
        "source_run_id": source_run_id,
        "reason": "Drift detected in: " + ", ".join(report.drifted_columns),
        "drifted_columns": report.drifted_columns,
        "automatic_retraining": False,
    }


def try_write_evidently_report(
    reference_path: str | Path,
    current_path: str | Path,
    output_html: str | Path,
) -> bool:
    """Optionally write an Evidently HTML report when the extra is installed.

    The lightweight PSI/TVD report remains the deterministic baseline when the
    optional dependency is absent or its API is unavailable.
    """

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=read_csv_dataset(reference_path),
            current_data=read_csv_dataset(current_path),
        )
        report.save_html(str(output_html))
        return True
    except (ImportError, AttributeError, TypeError):
        return False
